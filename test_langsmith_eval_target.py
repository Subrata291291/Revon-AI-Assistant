import os
import unittest
import uuid
from unittest.mock import patch

from langsmith import traceable
from langsmith.run_helpers import ensure_traceable, get_current_run_tree, get_tracing_context, tracing_context

import retrieval_service
from conversation_service import ConversationManager
from evaluation.langsmith_eval.capture import parse_context_chunks, retrieval_capture
from evaluation.langsmith_eval.target import (
    PRODUCTION_PROJECT,
    langsmith_target,
    run_rag_target,
)

# Tests run in an isolated non-production process/context but perform global
# module-state swaps; explicitly allow in-process risk for this test module.
os.environ.setdefault("REVON_EVAL_ALLOW_IN_PROCESS_RISK", "1")


class TestLangsmithEvalTarget(unittest.TestCase):
    def test_parse_context_chunks_extracts_structured_fields(self):
        context = (
            "[Source 1]\nFilename: handbook.pdf\nPage: 2\nContent: Probation period is 3 months.\n\n"
            "[Source 2]\nFilename: handbook.pdf\nPage: 5\nContent: 20 paid annual leave days per year."
        )
        sources = [
            {"filename": "handbook.pdf", "page_number": 2, "score": 0.91},
            {"filename": "handbook.pdf", "page_number": 5, "score": 0.88},
        ]
        chunks = parse_context_chunks(context, sources)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["filename"], "handbook.pdf")
        self.assertEqual(chunks[0]["page_number"], 2)
        self.assertAlmostEqual(chunks[0]["score"], 0.91)
        self.assertIn("Probation period", chunks[0]["text"])
        self.assertEqual(chunks[1]["page_number"], 5)

    def test_context_local_retrieval_capture_and_safe_restore(self):
        module_original = retrieval_service.retrieve_context

        def fake_retrieve(question, top_k=5):
            return {
                "context": f"[Source 1]\nFilename: doc.pdf\nPage: 1\nContent: Evidence for {question}",
                "sources": [{"filename": "doc.pdf", "page_number": 1, "score": 0.9}],
                "total_matches": 1,
            }

        retrieval_service.retrieve_context = fake_retrieve  # type: ignore[assignment]
        try:
            before_capture = retrieval_service.retrieve_context
            with retrieval_capture() as sink:
                res = retrieval_service.retrieve_context("q1", top_k=3)
                self.assertEqual(res["total_matches"], 1)
            self.assertIsNotNone(sink.last)
            self.assertEqual(sink.last.question, "q1")
            self.assertEqual(sink.last.top_k, 3)
            self.assertEqual(len(sink.last.chunks), 1)
            self.assertEqual(sink.last.chunks[0]["filename"], "doc.pdf")
            # hook must be restored to the exact function present before capture
            self.assertIs(retrieval_service.retrieve_context, before_capture)
        finally:
            retrieval_service.retrieve_context = module_original  # type: ignore[assignment]

    def test_run_rag_target_captures_all_fields_uses_unique_eval_scope_and_restores(self):
        prod_manager = retrieval_service.conversation_manager
        self.assertIsInstance(prod_manager, ConversationManager)
        original_prod_sessions = dict(getattr(prod_manager, "_sessions", {}))

        retrieval_payload = {
            "context": "[Source 1]\nFilename: profile.pdf\nPage: 4\nContent: Python and FastAPI skills listed.",
            "sources": [{"filename": "profile.pdf", "page_number": 4, "score": 0.93}],
            "total_matches": 1,
        }
        answer_payload = {
            "conversation_id": "will-be-overridden",
            "answer": "Skills include Python and FastAPI [Source 1].",
            "sources": [{"filename": "profile.pdf", "page_number": 4, "score": 0.93}],
            "total_matches": 1,
            "is_policy_query": True,
            "intent": "document_question",
        }

        captured_call = {}

        def fake_answer(question, conversation_id=None, client_history=None, top_k=5):
            captured_call["question"] = question
            captured_call["conversation_id"] = conversation_id
            captured_call["client_history"] = client_history
            captured_call["top_k"] = top_k
            # simulate workflow retrieval call so capture hook records it
            retrieval_service.retrieve_context(question, top_k=top_k)
            out = dict(answer_payload)
            out["conversation_id"] = conversation_id
            return out

        with patch("retrieval_service.answer_conversational_question", side_effect=fake_answer), \
             patch("retrieval_service.retrieve_context", return_value=retrieval_payload):
            out1 = run_rag_target({"question": "What skills are listed?", "example_id": "ex1"}, top_k=5, redact=False)
            out2 = run_rag_target({"question": "What skills are listed?", "example_id": "ex1"}, top_k=5, redact=False)

        # required captured fields
        self.assertEqual(out1["question"], "What skills are listed?")
        self.assertEqual(out1["final_answer"], answer_payload["answer"])
        self.assertEqual(len(out1["retrieved_chunks"]), 1)
        self.assertEqual(out1["retrieved_chunks"][0]["filename"], "profile.pdf")
        self.assertEqual(out1["retrieved_chunks"][0]["page_number"], 4)
        self.assertEqual(out1["retrieval_scores"], [0.93])
        self.assertEqual(out1["retrieved_sources"][0]["filename"], "profile.pdf")

        # unique conversation_id per example/invocation and no client history leakage
        self.assertTrue(out1["conversation_id"].startswith("eval-ex1-"))
        self.assertTrue(out2["conversation_id"].startswith("eval-ex1-"))
        self.assertNotEqual(out1["conversation_id"], out2["conversation_id"])
        self.assertIsNone(captured_call["client_history"])

        # production conversation manager reference and its history must remain untouched
        self.assertIs(retrieval_service.conversation_manager, prod_manager)
        self.assertEqual(dict(getattr(prod_manager, "_sessions", {})), original_prod_sessions)

    def test_hooks_restored_after_exception(self):
        prod_manager = retrieval_service.conversation_manager
        original_retrieve = retrieval_service.retrieve_context

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        with patch("retrieval_service.answer_conversational_question", side_effect=boom), \
             patch("retrieval_service.retrieve_context", return_value={"context": "", "sources": [], "total_matches": 0}):
            with self.assertRaises(RuntimeError):
                run_rag_target({"question": "q", "example_id": "ex"}, redact=False)

        self.assertIs(retrieval_service.conversation_manager, prod_manager)
        # retrieve_context should be restored to the state before run_rag_target (patched value here)
        self.assertIsNotNone(retrieval_service.retrieve_context)

    def test_explicit_guard_blocks_unsafe_in_process_execution_by_default(self):
        with patch.dict(os.environ, {"REVON_EVAL_ALLOW_IN_PROCESS_RISK": "0"}, clear=False):
            # ensure guard sees disabled state
            os.environ["REVON_EVAL_ALLOW_IN_PROCESS_RISK"] = "0"
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    run_rag_target({"question": "q", "example_id": "ex"}, redact=False, allow_in_process_risk=False)
                self.assertIn("dedicated", str(ctx.exception).lower())
            finally:
                os.environ["REVON_EVAL_ALLOW_IN_PROCESS_RISK"] = "1"

    def test_preserves_sdk_experiment_context_and_child_trace_association(self):
        """Simulate SDK 0.12.6 _forward: traceable target invoked with langsmith_extra."""
        experiment_project = "revon-policy-chatbot-eval-abc12345"
        reference_example_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
        captured = {}

        @traceable(name="eval_child_op")
        def child_op():
            rt = get_current_run_tree()
            return {
                "project": rt.session_name if rt else None,
                "parent_run_id": str(rt.parent_run_id) if rt and rt.parent_run_id else None,
                "trace_id": str(rt.trace_id) if rt else None,
            }

        answer_payload = {
            "answer": "ok [Source 1]",
            "sources": [{"filename": "doc.pdf", "page_number": 1}],
            "total_matches": 1,
            "is_policy_query": True,
            "intent": "document_question",
        }

        def fake_answer(question, conversation_id=None, client_history=None, top_k=5):
            rt = get_current_run_tree()
            captured["inside_project"] = rt.session_name if rt else None
            captured["inside_ref"] = str(rt.reference_example_id) if rt and rt.reference_example_id else None
            captured["inside_run_id"] = str(rt.id) if rt else None
            captured["child"] = child_op()
            retrieval_service.retrieve_context(question, top_k=top_k)
            return dict(answer_payload, conversation_id=conversation_id)

        env_before = {k: os.environ.get(k) for k in list(os.environ.keys())}
        wrapped = ensure_traceable(langsmith_target)
        with patch("langsmith.run_trees.RunTree.post"), patch("langsmith.run_trees.RunTree.patch"), \
             patch("langsmith.client.Client.create_run", create=True), patch("langsmith.client.Client.update_run", create=True), \
             patch("retrieval_service.answer_conversational_question", side_effect=fake_answer), \
             patch("retrieval_service.retrieve_context", return_value={"context": "", "sources": [], "total_matches": 0}):
            # Match _forward: enabled="local" tracing_context + langsmith_extra with experiment context
            with tracing_context(enabled="local"):
                out = wrapped(
                    {"question": "q", "example_id": "ex-sdk"},
                    langsmith_extra={
                        "project_name": experiment_project,
                        "reference_example_id": reference_example_id,
                        "metadata": {"example_version": "v1"},
                    },
                )

        env_after = {k: os.environ.get(k) for k in list(os.environ.keys())}
        self.assertEqual(env_after, env_before)

        self.assertEqual(out["trace_project_name"], experiment_project)
        self.assertNotEqual(out["trace_project_name"], PRODUCTION_PROJECT)
        # SDK experiment context preserved for target/children
        self.assertEqual(captured.get("inside_project"), experiment_project)
        self.assertNotEqual(captured.get("inside_project"), PRODUCTION_PROJECT)
        self.assertEqual(captured.get("inside_ref"), str(reference_example_id))
        # Child trace inherits experiment project and is associated as child of target run
        self.assertEqual((captured.get("child") or {}).get("project"), experiment_project)
        self.assertNotEqual((captured.get("child") or {}).get("project"), PRODUCTION_PROJECT)
        self.assertIsNotNone(captured.get("inside_run_id"))
        self.assertEqual((captured.get("child") or {}).get("parent_run_id"), captured.get("inside_run_id"))

    def test_fallback_trace_project_isolated_from_production_without_sdk_context(self):
        captured = {}
        fallback = "revon-policy-chatbot-eval-fallback-test"

        @traceable(name="fallback_child_op")
        def child_op():
            rt = get_current_run_tree()
            ctx = get_tracing_context()
            return {"project": rt.session_name if rt else ctx.get("project_name")}

        answer_payload = {
            "answer": "ok",
            "sources": [],
            "total_matches": 0,
            "is_policy_query": True,
            "intent": "document_question",
        }

        def fake_answer(question, conversation_id=None, client_history=None, top_k=5):
            captured["ctx_project"] = get_tracing_context().get("project_name")
            captured["child"] = child_op()
            return dict(answer_payload, conversation_id=conversation_id)

        env_before = {k: os.environ.get(k) for k in list(os.environ.keys())}
        with patch("langsmith.run_trees.RunTree.post"), patch("langsmith.run_trees.RunTree.patch"), \
             patch("retrieval_service.answer_conversational_question", side_effect=fake_answer), \
             patch("retrieval_service.retrieve_context", return_value={"context": "", "sources": [], "total_matches": 0}):
            # Direct call (no SDK experiment context): must use safe fallback, never production
            out = run_rag_target(
                {"question": "q", "example_id": "ex-fallback"},
                redact=False,
                trace_project_name=fallback,
            )

        env_after = {k: os.environ.get(k) for k in list(os.environ.keys())}
        self.assertEqual(env_after, env_before)
        self.assertEqual(out["trace_project_name"], fallback)
        self.assertNotEqual(out["trace_project_name"], PRODUCTION_PROJECT)
        self.assertEqual(captured.get("ctx_project"), fallback)
        self.assertNotEqual(captured.get("ctx_project"), PRODUCTION_PROJECT)
        self.assertEqual((captured.get("child") or {}).get("project"), fallback)

    def test_production_project_context_is_never_used(self):
        # Even if caller/env supplies production project, target must isolate to safe fallback.
        answer_payload = {"answer": "ok", "sources": [], "total_matches": 0, "is_policy_query": True, "intent": "document_question"}
        captured = {}

        def fake_answer(question, conversation_id=None, client_history=None, top_k=5):
            captured["ctx_project"] = get_tracing_context().get("project_name")
            return dict(answer_payload, conversation_id=conversation_id)

        with patch.dict(os.environ, {"REVON_EVAL_TRACE_PROJECT": PRODUCTION_PROJECT}, clear=False), \
             patch("langsmith.run_trees.RunTree.post"), patch("langsmith.run_trees.RunTree.patch"), \
             patch("retrieval_service.answer_conversational_question", side_effect=fake_answer), \
             patch("retrieval_service.retrieve_context", return_value={"context": "", "sources": [], "total_matches": 0}):
            out = run_rag_target({"question": "q", "example_id": "ex-prod"}, redact=False)

        self.assertNotEqual(out["trace_project_name"], PRODUCTION_PROJECT)
        self.assertTrue(str(out["trace_project_name"]).startswith("revon-policy-chatbot-eval"))
        self.assertEqual(captured.get("ctx_project"), out["trace_project_name"])
        self.assertNotEqual(captured.get("ctx_project"), PRODUCTION_PROJECT)


if __name__ == "__main__":
    unittest.main()
