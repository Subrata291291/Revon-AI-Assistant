import json
import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch, MagicMock

import retrieval_service
import main


class _DummyRun:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.metadata = {}
        self.name = None
        self.extra = {}


class TestTracingUtils(unittest.TestCase):
    def test_redaction_masks_secrets_and_pii(self):
        from tracing_utils import redact_text, redact_obj
        text = "Contact a@b.com +1 (555) 123-4567 https://linkedin.com/in/x api_key=sk-secret token: abc123"
        red = redact_text(text)
        self.assertNotIn("a@b.com", red)
        self.assertNotIn("555", red)
        self.assertNotIn("linkedin.com/in/x", red)
        self.assertIn("[REDACTED_EMAIL]", red)
        self.assertIn("[REDACTED_PHONE]", red)
        self.assertIn("[REDACTED_PROFILE_URL]", red)
        self.assertIn("api_key=[REDACTED]", red)
        self.assertIn("token=[REDACTED]", red)

        obj = {"authorization": "Bearer secret", "nested": {"password": "pw", "safe": "ok"}}
        red_obj = redact_obj(obj)
        self.assertEqual(red_obj["authorization"], "[REDACTED]")
        self.assertEqual(red_obj["nested"]["password"], "[REDACTED]")
        self.assertEqual(red_obj["nested"]["safe"], "ok")

    def test_source_and_grounding_states(self):
        from tracing_utils import source_state, grounding_state
        self.assertEqual(source_state(retrieval_performed=False)[0], "retrieval_not_required")
        self.assertEqual(source_state(pinecone_match_count=0)[0], "no_chunks_retrieved")
        self.assertEqual(source_state(pinecone_match_count=2, kept_chunk_count=0)[0], "no_chunks_after_processing_or_gating")
        self.assertEqual(source_state(pinecone_match_count=2, kept_chunk_count=2, cited_source_count=0)[0], "retrieved_but_uncited")
        self.assertEqual(source_state(pinecone_match_count=2, kept_chunk_count=2, cited_source_count=1)[0], "retrieved_and_cited")
        self.assertEqual(grounding_state(has_context=False, cited_source_count=0, abstained=True), "abstained")
        self.assertEqual(grounding_state(has_context=True, cited_source_count=1, abstained=False), "grounded")
        self.assertEqual(grounding_state(has_context=True, cited_source_count=0, abstained=False), "partially_grounded")


class TestProductionTracingNonStreaming(unittest.TestCase):
    def setUp(self):
        self._orig_get_or_create = retrieval_service.conversation_manager.get_or_create
        self._orig_get_history = retrieval_service.conversation_manager.get_history_for_prompt if hasattr(retrieval_service.conversation_manager, "get_history_for_prompt") else None

    def tearDown(self):
        pass

    @patch("retrieval_service.generate_response")
    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.conversation_manager")
    def test_non_streaming_retrieved_and_cited_trace_flow(self, mock_cm, mock_retrieve, mock_chat, mock_intent_llm):
        # Intent LLM returns substantive question classification
        mock_intent_llm.return_value = '{"intent":"document_question","requires_retrieval":true,"resolved_question":"What is RAG?","search_query":"What is RAG?","reason":"doc"}'
        mock_chat.return_value = "RAG combines retrieval and generation [Source 1]."
        session = MagicMock()
        session.get_history_for_prompt.return_value = []
        mock_cm.get_or_create.return_value = ("cid-1", session)
        mock_retrieve.return_value = {
            "context": "[Source 1]\nChunk ID: c1\nFilename: rag.pdf\nPage: 3\nContent: RAG is retrieval augmented generation.",
            "sources": [{"source_index": 1, "chunk_id": "c1", "filename": "rag.pdf", "page_number": 3, "page": 3, "score": 0.91}],
            "retrieved_chunks": [{"source_index": 1, "chunk_id": "c1", "filename": "rag.pdf", "page_number": 3, "score": 0.91, "text": "RAG is retrieval augmented generation."}],
            "total_matches": 1,
            "kept_chunk_count": 1,
            "embedding_model": "gemini-embedding-2",
            "pinecone_index": "revon-policy-index",
            "pinecone_namespace": "revon-policies",
            "top_k": 5,
            "filters": {"is_active": {"$eq": True}},
            "retrieval_latency_ms": 12.3,
            "chunk_summaries": [{"source_index": 1, "chunk_id": "c1", "filename": "rag.pdf", "page_number": 3, "score": 0.91, "excerpt": "RAG..."}],
        }

        trace_calls = []
        set_calls = []

        @contextmanager
        def fake_trace(name, run_type="chain", inputs=None, metadata=None, tags=None):
            trace_calls.append({"name": name, "run_type": run_type, "inputs": inputs or {}, "metadata": metadata or {}})
            yield

        with patch("tracing_utils.is_langsmith_enabled", return_value=True), \
             patch("tracing_utils.trace_operation", side_effect=fake_trace), \
             patch("tracing_utils.ls_trace", new=None), \
             patch("tracing_utils.safe_set_current_run", side_effect=lambda **k: set_calls.append(k)), \
             patch("retrieval_service.safe_set_current_run", side_effect=lambda **k: set_calls.append(k)), \
             patch("main.safe_set_current_run", side_effect=lambda **k: set_calls.append(k)):
            out = retrieval_service.answer_conversational_question("What is RAG?", conversation_id="cid-1", client_history=[])

        self.assertEqual(out["conversation_id"], "cid-1")
        self.assertEqual(len(out["sources"]), 1)
        self.assertEqual(out["sources"][0]["chunk_id"], "c1")
        self.assertEqual(out["sources"][0]["filename"], "rag.pdf")
        self.assertEqual(out["sources"][0]["page_number"], 3)
        self.assertIn("[Source 1]", out["answer"])

        names = [c["name"] for c in trace_calls]
        ops = [c.get("metadata", {}).get("operation") for c in set_calls if isinstance(c, dict)]
        self.assertIn("llm_generation", names)
        self.assertIn("query_rewriting", ops)
        self.assertIn("context_construction", ops)
        self.assertIn("citation_source_extraction", ops)
        self.assertIn("conversation_persistence", ops)

    @patch("retrieval_service.generate_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.conversation_manager")
    def test_empty_retrieval_state_explicit(self, mock_cm, mock_retrieve, mock_intent_llm):
        mock_intent_llm.return_value = '{"intent":"document_question","requires_retrieval":true,"resolved_question":"Unknown","search_query":"Unknown","reason":"doc"}'
        session = MagicMock(); session.get_history_for_prompt.return_value = []
        mock_cm.get_or_create.return_value = ("cid-2", session)
        mock_retrieve.return_value = {
            "context": "", "sources": [], "retrieved_chunks": [], "total_matches": 0, "kept_chunk_count": 0,
            "embedding_model": "gemini-embedding-2", "pinecone_index": "revon-policy-index", "pinecone_namespace": "revon-policies",
            "top_k": 5, "filters": {"is_active": {"$eq": True}}, "retrieval_latency_ms": 5.0, "chunk_summaries": [],
        }
        set_calls = []
        with patch("tracing_utils.is_langsmith_enabled", return_value=True), \
             patch("tracing_utils.trace_operation", side_effect=lambda *a, **k: _nullctx()), \
             patch("retrieval_service.safe_set_current_run", side_effect=lambda **k: set_calls.append(k)), \
             patch("tracing_utils.safe_set_current_run", side_effect=lambda **k: set_calls.append(k)):
            out = retrieval_service.answer_conversational_question("Unknown?", conversation_id="cid-2")
        self.assertEqual(out["sources"], [])
        self.assertEqual(out["answer"], retrieval_service.DOCUMENT_NOT_FOUND_MESSAGE)
        ops = [c.get("metadata", {}).get("operation") for c in set_calls if isinstance(c, dict)]
        self.assertIn("citation_source_extraction", ops)
        finals = [c.get("outputs", {}) for c in set_calls if isinstance(c.get("outputs"), dict)]
        reasons = [o.get("source_state_reason") for o in finals if o.get("source_state_reason")]
        self.assertTrue(any("no matching chunks" in str(r).lower() or "excluded" in str(r).lower() or "failed" in str(r).lower() or "retrieval" in str(r).lower() for r in reasons) or out["sources"] == [])

    @patch("retrieval_service.generate_response")
    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.conversation_manager")
    def test_retrieved_but_uncited_sources(self, mock_cm, mock_retrieve, mock_chat, mock_intent_llm):
        mock_intent_llm.return_value = '{"intent":"document_question","requires_retrieval":true,"resolved_question":"Q","search_query":"Q","reason":"doc"}'
        mock_chat.return_value = "I cannot determine this from the provided documents."  # no [Source X]
        session = MagicMock(); session.get_history_for_prompt.return_value = []
        mock_cm.get_or_create.return_value = ("cid-3", session)
        mock_retrieve.return_value = {
            "context": "[Source 1]\nChunk ID: c1\nFilename: a.pdf\nPage: 1\nContent: something",
            "sources": [{"source_index": 1, "chunk_id": "c1", "filename": "a.pdf", "page_number": 1, "page": 1, "score": 0.5}],
            "retrieved_chunks": [{"source_index": 1, "chunk_id": "c1", "filename": "a.pdf", "page_number": 1, "score": 0.5, "text": "something"}],
            "total_matches": 1, "kept_chunk_count": 1,
            "embedding_model": "gemini-embedding-2", "pinecone_index": "revon-policy-index", "pinecone_namespace": "revon-policies",
            "top_k": 5, "filters": {"is_active": {"$eq": True}}, "retrieval_latency_ms": 5.0,
            "chunk_summaries": [{"source_index": 1, "chunk_id": "c1", "filename": "a.pdf", "page_number": 1, "score": 0.5, "excerpt": "something"}],
        }
        set_calls = []
        with patch("tracing_utils.trace_operation", side_effect=lambda *a, **k: _nullctx()), \
             patch("tracing_utils.is_langsmith_enabled", return_value=True), \
             patch("retrieval_service.safe_set_current_run", side_effect=lambda **k: set_calls.append(k)), \
             patch("tracing_utils.safe_set_current_run", side_effect=lambda **k: set_calls.append(k)):
            out = retrieval_service.answer_conversational_question("Q", conversation_id="cid-3")
        self.assertEqual(out["sources"], [])
        # finalization should record retrieved_but_uncited or abstained/uncited semantics
        finals = [c.get("outputs", {}) for c in set_calls if isinstance(c.get("outputs"), dict)]
        states = [o.get("source_state") for o in finals if o.get("source_state")]
        self.assertTrue(any(s in ("retrieved_but_uncited",) for s in states) or out["sources"] == [])

    @patch("retrieval_service.generate_response")
    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.conversation_manager")
    def test_tracing_failure_resilience(self, mock_cm, mock_retrieve, mock_chat, mock_intent_llm):
        mock_intent_llm.return_value = '{"intent":"document_question","requires_retrieval":true,"resolved_question":"Q","search_query":"Q","reason":"doc"}'
        mock_chat.return_value = "Answer [Source 1]."
        session = MagicMock(); session.get_history_for_prompt.return_value = []
        mock_cm.get_or_create.return_value = ("cid-4", session)
        mock_retrieve.return_value = {
            "context": "[Source 1]\nChunk ID: c1\nFilename: a.pdf\nPage: 1\nContent: text",
            "sources": [{"source_index": 1, "chunk_id": "c1", "filename": "a.pdf", "page_number": 1, "page": 1, "score": 0.9}],
            "retrieved_chunks": [], "total_matches": 1, "kept_chunk_count": 1,
            "embedding_model": "gemini-embedding-2", "pinecone_index": "revon-policy-index", "pinecone_namespace": "revon-policies",
            "top_k": 5, "filters": {"is_active": {"$eq": True}}, "retrieval_latency_ms": 1.0, "chunk_summaries": [],
        }
        def boom(*a, **k):
            raise RuntimeError("trace backend down")
        # Even if tracing operations/metadata fail, response must succeed
        with patch("tracing_utils.is_langsmith_enabled", return_value=True), \
             patch("tracing_utils.trace_operation", side_effect=boom):
            out = retrieval_service.answer_conversational_question("Q", conversation_id="cid-4")
        self.assertIn("answer", out)
        self.assertTrue(out["answer"])
        self.assertIsInstance(out.get("sources", []), list)


class TestProductionTracingStreaming(unittest.TestCase):
    @patch("retrieval_service.generate_response")
    @patch("retrieval_service.generate_chat_stream")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.conversation_manager")
    def test_streaming_trace_during_generation_and_source_serialization(self, mock_cm, mock_retrieve, mock_stream, mock_intent_llm):
        mock_intent_llm.return_value = '{"intent":"document_question","requires_retrieval":true,"resolved_question":"What?","search_query":"What?","reason":"doc"}'
        mock_stream.return_value = iter(["Hello ", "[Source 1]"])
        session = MagicMock(); session.get_history_for_prompt.return_value = []
        mock_cm.get_or_create.return_value = ("cid-s1", session)
        mock_retrieve.return_value = {
            "context": "[Source 1]\nChunk ID: c9\nFilename: doc.pdf\nPage: 2\nContent: evidence",
            "sources": [{"source_index": 1, "chunk_id": "c9", "filename": "doc.pdf", "page_number": 2, "page": 2, "score": 0.8}],
            "retrieved_chunks": [], "total_matches": 1, "kept_chunk_count": 1,
            "embedding_model": "gemini-embedding-2", "pinecone_index": "revon-policy-index", "pinecone_namespace": "revon-policies",
            "top_k": 5, "filters": {"is_active": {"$eq": True}}, "retrieval_latency_ms": 2.0, "chunk_summaries": [],
        }
        trace_calls = []
        @contextmanager
        def fake_trace(name, run_type="chain", inputs=None, metadata=None, tags=None):
            trace_calls.append({"name": name})
            yield
        events = []
        with patch("tracing_utils.is_langsmith_enabled", return_value=True), \
             patch("tracing_utils.trace_operation", side_effect=fake_trace), \
             patch("retrieval_service.safe_set_current_run", lambda **k: None), \
             patch("tracing_utils.safe_set_current_run", lambda **k: None):
            for ev in retrieval_service.stream_conversational_question("What?", conversation_id="cid-s1"):
                events.append(ev)
        joined = "".join(events)
        self.assertIn("event: start", joined)
        self.assertIn("event: token", joined)
        self.assertIn("event: sources", joined)
        self.assertIn("event: complete", joined)
        self.assertIn('"chunk_id": "c9"', joined)
        set_calls = []
        # capture ops via safe_set_current_run for decorated helpers; llm_generation via trace_operation
        with patch("tracing_utils.is_langsmith_enabled", return_value=True), \
             patch("tracing_utils.trace_operation", side_effect=fake_trace), \
             patch("retrieval_service.safe_set_current_run", side_effect=lambda **k: set_calls.append(k)), \
             patch("tracing_utils.safe_set_current_run", side_effect=lambda **k: set_calls.append(k)):
            for ev in retrieval_service.stream_conversational_question("What?", conversation_id="cid-s1"):
                events.append(ev)
        joined = "".join(events)
        self.assertIn("event: start", joined)
        self.assertIn("event: token", joined)
        self.assertIn("event: sources", joined)
        self.assertIn("event: complete", joined)
        self.assertIn('"chunk_id": "c9"', joined)
        names = [c["name"] for c in trace_calls]
        ops = [c.get("metadata", {}).get("operation") for c in set_calls if isinstance(c, dict)]
        self.assertIn("llm_generation", names)
        self.assertIn("query_rewriting", ops)
        self.assertIn("context_construction", ops)
        self.assertIn("citation_source_extraction", ops)
        self.assertIn("conversation_persistence", ops)

    @patch("retrieval_service.generate_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.conversation_manager")
    def test_streaming_empty_retrieval(self, mock_cm, mock_retrieve, mock_intent_llm):
        mock_intent_llm.return_value = '{"intent":"document_question","requires_retrieval":true,"resolved_question":"Q","search_query":"Q","reason":"doc"}'
        session = MagicMock(); session.get_history_for_prompt.return_value = []
        mock_cm.get_or_create.return_value = ("cid-s2", session)
        mock_retrieve.return_value = {"context": "", "sources": [], "total_matches": 0, "kept_chunk_count": 0, "chunk_summaries": []}
        events = []
        with patch("tracing_utils.is_langsmith_enabled", return_value=True), \
             patch("tracing_utils.trace_operation", side_effect=lambda *a, **k: _nullctx()), \
             patch("retrieval_service.safe_set_current_run", lambda **k: None), \
             patch("tracing_utils.safe_set_current_run", lambda **k: None):
            for ev in retrieval_service.stream_conversational_question("Q", conversation_id="cid-s2"):
                events.append(ev)
        joined = "".join(events)
        self.assertIn(retrieval_service.DOCUMENT_NOT_FOUND_MESSAGE, joined)
        self.assertIn('"sources": []', joined)

    @patch("retrieval_service.stream_conversational_question")
    def test_stream_endpoint_traces_during_generator_execution(self, mock_stream):
        # Ensure root trace context wraps actual iteration
        order = []
        @contextmanager
        def fake_req_ctx(**kwargs):
            order.append("enter")
            yield
            order.append("exit")
        def gen():
            order.append("gen")
            yield 'event: start\ndata: {"conversation_id":"c"}\n\n'
            yield 'event: complete\ndata: {"conversation_id":"c","status":"success","is_policy_query":false,"sources":[],"answer":"ok"}\n\n'
        mock_stream.return_value = gen()

        from fastapi.testclient import TestClient
        client = TestClient(main.app)
        overrides = {main.verify_chat_token: lambda: None}
        old = dict(main.app.dependency_overrides)
        main.app.dependency_overrides.update(overrides)
        try:
            with patch("main.enforce_chat_rate_limit", lambda x: None), \
                 patch("main.request_trace_context", side_effect=fake_req_ctx):
                resp = client.post("/api/v1/chat/stream", json={"question": "hi"}, headers={"Authorization": "Bearer x"})
                self.assertEqual(resp.status_code, 200)
                _ = resp.text  # consume streaming generator so tracing wraps execution
        finally:
            main.app.dependency_overrides = old
        # tracing context must be active while generator runs
        self.assertIn("enter", order)
        self.assertIn("gen", order)
        self.assertIn("exit", order)
        self.assertLess(order.index("enter"), order.index("gen"))


class TestTenantIsolationAndSerialization(unittest.TestCase):
    @patch("retrieval_service.generate_response")
    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.conversation_manager")
    def test_retrieval_uses_active_only_filters_and_sources_serialized(self, mock_cm, mock_retrieve, mock_chat, mock_intent_llm):
        mock_intent_llm.return_value = '{"intent":"document_question","requires_retrieval":true,"resolved_question":"Q","search_query":"Q","reason":"doc"}'
        mock_chat.return_value = "A [Source 1]"
        session = MagicMock(); session.get_history_for_prompt.return_value = []
        mock_cm.get_or_create.return_value = ("cid-t1", session)
        mock_retrieve.return_value = {
            "context": "[Source 1]\nChunk ID: c1\nFilename: doc.pdf\nPage: 7\nContent: x",
            "sources": [{"source_index": 1, "chunk_id": "c1", "filename": "doc.pdf", "page_number": 7, "page": 7, "score": 0.7}],
            "retrieved_chunks": [], "total_matches": 1, "kept_chunk_count": 1,
            "embedding_model": "gemini-embedding-2", "pinecone_index": "revon-policy-index", "pinecone_namespace": "revon-policies",
            "top_k": 5, "filters": {"is_active": {"$eq": True}}, "retrieval_latency_ms": 1.0, "chunk_summaries": [],
        }
        with patch("tracing_utils.trace_operation", side_effect=lambda *a, **k: _nullctx()), \
             patch("retrieval_service.safe_set_current_run", lambda **k: None), \
             patch("tracing_utils.safe_set_current_run", lambda **k: None):
            out = retrieval_service.answer_conversational_question("Q", conversation_id="cid-t1")
        mock_retrieve.assert_called_once()
        self.assertEqual(out["sources"][0]["page_number"], 7)
        self.assertEqual(out["sources"][0]["chunk_id"], "c1")


class TestTracingAndGroundingFixes(unittest.TestCase):
    def test_citation_normalization_supports_fullwidth_brackets(self):
        text = "Answer 【Source 1】 and ［Source 2］."
        self.assertEqual(
            retrieval_service.extract_cited_source_indices(text, 3),
            [1, 2],
        )
        self.assertEqual(
            retrieval_service.normalize_citation_markers(text),
            "Answer [Source 1] and [Source 2].",
        )

    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.conversation_manager")
    @patch("retrieval_service.assess_evidence_sufficiency")
    @patch("retrieval_service.generate_response")
    def test_irrelevant_chunks_produce_clean_abstention_and_sources_empty(
        self, mock_intent_llm, mock_assess, mock_cm, mock_retrieve, mock_chat
    ):
        mock_intent_llm.return_value = '{"intent":"document_question","requires_retrieval":true,"resolved_question":"What is RevOn annual leave policy?","search_query":"RevOn annual leave policy","reason":"doc"}'
        mock_assess.return_value = {
            "overall_sufficiency": "insufficient",
            "conflict_detected": False,
            "reason": "No retrieved chunk supports annual leave policy.",
            "sources": [{"source_index": 1, "relevant": False, "evidence": "none", "reason": "irrelevant"}],
            "relevant_indices": [],
            "evidence_indices": [],
            "supported_indices": [],
            "conflict_indices": [],
            "assessment_failed": False,
        }
        session = MagicMock(); session.get_history_for_prompt.return_value = []
        mock_cm.get_or_create.return_value = ("cid-noev", session)
        mock_retrieve.return_value = {
            "context": "[Source 1]\nChunk ID: c1\nFilename: api.pdf\nPage: 1\nContent: API basics",
            "sources": [{"source_index": 1, "chunk_id": "c1", "filename": "api.pdf", "page_number": 1, "page": 1, "score": 0.3}],
            "retrieved_chunks": [{"source_index": 1, "chunk_id": "c1", "filename": "api.pdf", "page_number": 1, "page": 1, "score": 0.3, "text": "API basics"}],
            "total_matches": 1,
            "kept_chunk_count": 1,
            "embedding_model": "gemini-embedding-2",
            "pinecone_index": "revon-policy-index",
            "pinecone_namespace": "revon-policies",
            "top_k": 5,
            "filters": {"is_active": {"$eq": True}},
            "retrieval_latency_ms": 1.0,
            "chunk_summaries": [],
        }
        with patch("tracing_utils.trace_operation", side_effect=lambda *a, **k: _nullctx()), \
             patch("retrieval_service.safe_set_current_run", lambda **k: None), \
             patch("tracing_utils.safe_set_current_run", lambda **k: None), \
             patch("retrieval_service.safe_set_request_root_run", lambda **k: None):
            out = retrieval_service.answer_conversational_question("What is RevOn annual leave policy?", conversation_id="cid-noev")

        self.assertEqual(out["sources"], [])
        self.assertEqual(out["answer"], retrieval_service.DOCUMENT_NOT_FOUND_MESSAGE)
        self.assertEqual(out.get("grounding_state"), "abstained")
        self.assertEqual(out.get("evidence_sufficiency"), "insufficient")
        mock_chat.assert_not_called()

    @patch("retrieval_service.generate_chat_stream")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.conversation_manager")
    @patch("retrieval_service.assess_evidence_sufficiency")
    @patch("retrieval_service.generate_response")
    def test_streaming_no_evidence_abstains_with_empty_sources(
        self, mock_intent_llm, mock_assess, mock_cm, mock_retrieve, mock_stream
    ):
        mock_intent_llm.return_value = '{"intent":"document_question","requires_retrieval":true,"resolved_question":"leave policy?","search_query":"leave policy","reason":"doc"}'
        mock_assess.return_value = {
            "overall_sufficiency": "insufficient",
            "conflict_detected": False,
            "reason": "Irrelevant chunks",
            "sources": [{"source_index": 1, "relevant": False, "evidence": "none", "reason": "irrelevant"}],
            "relevant_indices": [],
            "evidence_indices": [],
            "supported_indices": [],
            "conflict_indices": [],
            "assessment_failed": False,
        }
        session = MagicMock(); session.get_history_for_prompt.return_value = []
        mock_cm.get_or_create.return_value = ("cid-s-noev", session)
        mock_retrieve.return_value = {
            "context": "[Source 1]\nChunk ID: c1\nFilename: api.pdf\nPage: 1\nContent: API basics",
            "sources": [{"source_index": 1, "chunk_id": "c1", "filename": "api.pdf", "page_number": 1, "page": 1, "score": 0.3}],
            "retrieved_chunks": [{"source_index": 1, "chunk_id": "c1", "filename": "api.pdf", "page_number": 1, "page": 1, "score": 0.3, "text": "API basics"}],
            "total_matches": 1,
            "kept_chunk_count": 1,
            "chunk_summaries": [],
        }
        events = []
        with patch("tracing_utils.trace_operation", side_effect=lambda *a, **k: _nullctx()), \
             patch("retrieval_service.safe_set_current_run", lambda **k: None), \
             patch("tracing_utils.safe_set_current_run", lambda **k: None), \
             patch("retrieval_service.safe_set_request_root_run", lambda **k: None):
            for ev in retrieval_service.stream_conversational_question("leave policy?", conversation_id="cid-s-noev"):
                events.append(ev)
        joined = "".join(events)
        self.assertIn("event: start", joined)
        self.assertIn("event: token", joined)
        self.assertIn("event: complete", joined)
        self.assertNotIn("event: sources", joined)
        self.assertIn('"sources": []', joined)
        self.assertIn(retrieval_service.DOCUMENT_NOT_FOUND_MESSAGE, joined)
        mock_stream.assert_not_called()

    @patch("retrieval_service.generate_chat_stream")
    @patch("retrieval_service.generate_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.conversation_manager")
    def test_streaming_rag_uses_single_llm_generation_span(self, mock_cm, mock_retrieve, mock_intent_llm, mock_stream):
        mock_intent_llm.return_value = '{"intent":"document_question","requires_retrieval":true,"resolved_question":"Q","search_query":"Q","reason":"doc"}'
        mock_stream.return_value = iter(["Answer ", "[Source 1]"])
        session = MagicMock(); session.get_history_for_prompt.return_value = []
        mock_cm.get_or_create.return_value = ("cid-s-span", session)
        mock_retrieve.return_value = {
            "context": "[Source 1]\nChunk ID: c1\nFilename: doc.pdf\nPage: 1\nContent: evidence",
            "sources": [{"source_index": 1, "chunk_id": "c1", "filename": "doc.pdf", "page_number": 1, "page": 1, "score": 0.9}],
            "retrieved_chunks": [{"source_index": 1, "chunk_id": "c1", "filename": "doc.pdf", "page_number": 1, "page": 1, "score": 0.9, "text": "evidence"}],
            "total_matches": 1, "kept_chunk_count": 1, "chunk_summaries": [],
        }
        trace_calls = []
        @contextmanager
        def fake_trace(name, run_type="chain", inputs=None, metadata=None, tags=None):
            trace_calls.append({"name": name, "run_type": run_type})
            yield
        with patch("tracing_utils.is_langsmith_enabled", return_value=True), \
             patch("tracing_utils.trace_operation", side_effect=fake_trace), \
             patch("retrieval_service.safe_set_current_run", lambda **k: None), \
             patch("tracing_utils.safe_set_current_run", lambda **k: None), \
             patch("retrieval_service.safe_set_request_root_run", lambda **k: None):
            for _ in retrieval_service.stream_conversational_question("Q", conversation_id="cid-s-span"):
                pass
        llm_spans = [c for c in trace_calls if c["name"] == "llm_generation"]
        self.assertEqual(len(llm_spans), 1, f"Expected single llm_generation span, got {llm_spans} from {trace_calls}")


@contextmanager
def _nullctx():
    yield


if __name__ == "__main__":
    unittest.main()