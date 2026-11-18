import unittest

from langsmith.evaluation import EvaluationResult

from evaluation.langsmith_eval.evaluators import (
    evaluate_answer_correctness,
    evaluate_faithfulness,
    evaluate_retrieval_relevance,
)
from retrieval_service import DOCUMENT_NOT_FOUND_MESSAGE


def _info(res):
    return getattr(res, "evaluator_info", None) or {}


def _assert_sdk_compatible(testcase: unittest.TestCase, res: EvaluationResult, expected_key: str):
    testcase.assertIsInstance(res, EvaluationResult)
    testcase.assertEqual(res.key, expected_key)
    testcase.assertIsInstance(res.score, (int, float))
    testcase.assertGreaterEqual(float(res.score), 0.0)
    testcase.assertLessEqual(float(res.score), 1.0)
    # Must pass actual SDK validation / supported-field model behavior
    dumped = res.model_dump()
    testcase.assertEqual(
        set(dumped.keys()),
        {
            "key",
            "score",
            "value",
            "metadata",
            "comment",
            "correction",
            "evaluator_info",
            "feedback_config",
            "source_run_id",
            "target_run_id",
            "extra",
        },
    )
    EvaluationResult(**dumped)  # should not raise (no unsupported fields)
    testcase.assertIsInstance(res.comment, (str, type(None)))
    testcase.assertIsInstance(_info(res), dict)
    if res.comment is not None:
        testcase.assertEqual(_info(res).get("explanation"), res.comment)


class TestLangsmithEvalEvaluators(unittest.TestCase):
    def _retrieved(self, chunks, context=None):
        chunks = chunks or []
        if context is None:
            parts = []
            for c in chunks:
                parts.append(
                    f"[Source {c.get('source_index')}]\nFilename: {c.get('filename')}\nPage: {c.get('page_number')}\nContent: {c.get('text')}"
                )
            context = "\n\n".join(parts)
        return {
            "retrieved_chunks": chunks,
            "retrieved_context": context,
            "retrieved_sources": [
                {"filename": c.get("filename"), "page_number": c.get("page_number"), "score": c.get("score")}
                for c in chunks
            ],
        }

    def test_retrieval_relevant_vs_irrelevant_and_precision_recall(self):
        inputs = {"question": "What is the probation period?"}
        ref = {
            "expected_key_facts": ["Probation period is 3 months"],
            "expected_supporting_information": ["Handbook states probation period is 3 months"],
            "expected_relevant_chunks": [
                {"evidence_id": "ev1", "filename": "handbook.pdf", "page_number": 2, "text_must_contain": "probation period is 3 months"}
            ],
            "expected_abstention": {"should_abstain": False, "acceptable_responses": [], "rationale": ""},
        }
        relevant_out = self._retrieved([
            {"source_index": 1, "filename": "handbook.pdf", "page_number": 2, "score": 0.9, "text": "The probation period is 3 months for new employees."}
        ])
        irrelevant_out = self._retrieved([
            {"source_index": 1, "filename": "handbook.pdf", "page_number": 9, "score": 0.4, "text": "Office cafeteria opens at 9 AM."}
        ])

        rel = evaluate_retrieval_relevance(inputs, relevant_out, ref)
        irr = evaluate_retrieval_relevance(inputs, irrelevant_out, ref)

        _assert_sdk_compatible(self, rel, "retrieval_relevance")
        _assert_sdk_compatible(self, irr, "retrieval_relevance")
        self.assertGreater(rel.score, irr.score)
        self.assertTrue(_info(rel)["pass"])
        self.assertFalse(_info(irr)["pass"])
        self.assertEqual(_info(rel)["retrieval_precision"], 1.0)
        self.assertEqual(_info(rel)["retrieval_recall"], 1.0)
        self.assertEqual(_info(rel)["retrieval_f1"], 1.0)
        self.assertEqual(_info(rel)["relevant_retrieved_count"], 1)
        self.assertEqual(_info(rel)["retrieved_count"], 1)
        self.assertEqual(_info(rel)["ground_truth_relevant_count"], 1)
        self.assertEqual(_info(irr)["retrieval_precision"], 0.0)
        self.assertEqual(_info(irr)["retrieval_recall"], 0.0)
        self.assertEqual(_info(irr)["retrieval_f1"], 0.0)
        self.assertEqual(_info(irr)["relevant_retrieved_count"], 0)
        self.assertGreater(_info(rel)["evidence_coverage"], _info(irr)["evidence_coverage"])

    def test_missing_info_irrelevant_retrieved_chunks_not_high_due_to_abstention(self):
        inputs = {"question": "What is the pet insurance benefit?"}
        ref = {
            "expected_key_facts": [],
            "expected_supporting_information": [],
            "expected_relevant_chunks": [],
            "expected_abstention": {
                "should_abstain": True,
                "acceptable_responses": [DOCUMENT_NOT_FOUND_MESSAGE],
                "rationale": "No evidence exists.",
            },
        }
        # Retrieval returned irrelevant chunks (bad retrieval) but answer may abstain correctly elsewhere.
        outputs = self._retrieved([
            {"source_index": 1, "filename": "handbook.pdf", "page_number": 5, "score": 0.3, "text": "Annual leave is 20 days."},
            {"source_index": 2, "filename": "handbook.pdf", "page_number": 7, "score": 0.2, "text": "Remote work up to 2 days per week."},
        ])
        res = evaluate_retrieval_relevance(inputs, outputs, ref)
        _assert_sdk_compatible(self, res, "retrieval_relevance")
        # Must not be high: irrelevant chunks with empty GT relevant set => low precision/relevance
        self.assertLessEqual(res.score, 0.5)
        self.assertFalse(_info(res)["pass"])
        self.assertEqual(_info(res)["retrieval_precision"], 0.0)
        # gt_count == 0 => recall is vacuously 1.0 by existing metric definition
        self.assertEqual(_info(res)["retrieval_recall"], 1.0)
        self.assertEqual(_info(res)["retrieval_f1"], 0.0)
        self.assertEqual(_info(res)["relevant_retrieved_count"], 0)
        self.assertEqual(_info(res)["retrieved_count"], 2)
        self.assertEqual(_info(res)["ground_truth_relevant_count"], 0)
        self.assertEqual(_info(res)["evidence_coverage"], 0.0)

    def test_faithfulness_correct_vs_unsupported_and_fabricated_citation(self):
        chunks = [
            {"source_index": 1, "filename": "handbook.pdf", "page_number": 2, "score": 0.9, "text": "The probation period is 3 months for new employees."}
        ]
        outputs_ctx = self._retrieved(chunks)
        inputs = {"question": "What is the probation period?"}
        ref = {"expected_abstention": {"should_abstain": False, "acceptable_responses": [], "rationale": ""}}

        correct = {**outputs_ctx, "final_answer": "The probation period is 3 months [Source 1]."}
        unsupported = {**outputs_ctx, "final_answer": "The probation period is 6 months and includes a signing bonus [Source 1]."}
        fabricated = {**outputs_ctx, "final_answer": "The probation period is 3 months [Source 2]."}

        ok = evaluate_faithfulness(inputs, correct, ref)
        bad = evaluate_faithfulness(inputs, unsupported, ref)
        fab = evaluate_faithfulness(inputs, fabricated, ref)

        _assert_sdk_compatible(self, ok, "faithfulness_groundedness")
        _assert_sdk_compatible(self, bad, "faithfulness_groundedness")
        _assert_sdk_compatible(self, fab, "faithfulness_groundedness")
        self.assertTrue(_info(ok)["pass"])
        self.assertEqual(_info(ok)["unsupported_claims"], [])
        self.assertEqual(_info(ok)["fabricated_citations"], [])

        self.assertFalse(_info(bad)["pass"])
        self.assertGreater(len(_info(bad)["unsupported_claims"]), 0)

        self.assertFalse(_info(fab)["pass"])
        self.assertIn("[Source 2]", _info(fab)["fabricated_citations"])

    def test_faithfulness_missing_info_abstention_and_does_not_use_reference_as_evidence(self):
        inputs = {"question": "What is the pet insurance benefit?"}
        # Reference contains info that is NOT in retrieved context; faithfulness must ignore it.
        ref = {
            "reference_answer": "Pet insurance is 1000 USD coverage.",
            "expected_abstention": {"should_abstain": True, "acceptable_responses": [DOCUMENT_NOT_FOUND_MESSAGE], "rationale": ""},
        }
        no_ctx = self._retrieved([], context="")
        abstain = {**no_ctx, "final_answer": DOCUMENT_NOT_FOUND_MESSAGE}
        hallucinated = {**no_ctx, "final_answer": "Pet insurance is 1000 USD coverage [Source 1]."}

        ok = evaluate_faithfulness(inputs, abstain, ref)
        bad = evaluate_faithfulness(inputs, hallucinated, ref)

        _assert_sdk_compatible(self, ok, "faithfulness_groundedness")
        _assert_sdk_compatible(self, bad, "faithfulness_groundedness")
        self.assertTrue(_info(ok)["pass"])
        self.assertEqual(_info(ok)["unsupported_claims"], [])
        self.assertEqual(_info(ok)["fabricated_citations"], [])
        self.assertFalse(_info(bad)["pass"])
        self.assertTrue(_info(bad)["unsupported_claims"] or _info(bad)["fabricated_citations"])

    def test_answer_correctness_correct_incorrect_and_abstention_separately(self):
        inputs = {"question": "What is the probation period?"}
        ref_non_abstain = {
            "reference_answer": "The probation period is 3 months.",
            "expected_key_facts": ["Probation period is 3 months"],
            "expected_abstention": {"should_abstain": False, "acceptable_responses": [], "rationale": ""},
        }
        ref_abstain = {
            "reference_answer": DOCUMENT_NOT_FOUND_MESSAGE,
            "expected_key_facts": [],
            "expected_abstention": {"should_abstain": True, "acceptable_responses": [DOCUMENT_NOT_FOUND_MESSAGE], "rationale": "No info"},
        }

        correct = evaluate_answer_correctness(inputs, {"final_answer": "New employees have a 3 months probation period."}, ref_non_abstain)
        incorrect = evaluate_answer_correctness(inputs, {"final_answer": "The probation period is 6 months."}, ref_non_abstain)
        abstain_ok = evaluate_answer_correctness(inputs, {"final_answer": DOCUMENT_NOT_FOUND_MESSAGE}, ref_abstain)
        abstain_bad = evaluate_answer_correctness(inputs, {"final_answer": "Pet insurance is provided."}, ref_abstain)

        _assert_sdk_compatible(self, correct, "answer_correctness")
        _assert_sdk_compatible(self, incorrect, "answer_correctness")
        _assert_sdk_compatible(self, abstain_ok, "answer_correctness")
        _assert_sdk_compatible(self, abstain_bad, "answer_correctness")
        self.assertTrue(_info(correct)["pass"])
        self.assertFalse(_info(incorrect)["pass"])
        self.assertTrue(_info(abstain_ok)["pass"])
        self.assertTrue(_info(abstain_ok)["abstention_expected"])
        self.assertTrue(_info(abstain_ok)["abstention_correct"])
        self.assertFalse(_info(abstain_bad)["pass"])
        self.assertFalse(_info(abstain_bad)["abstention_correct"])

        # not exact-string-only: semantically equivalent phrasing should pass
        paraphrase = evaluate_answer_correctness(inputs, {"final_answer": "Probation lasts 3 months."}, ref_non_abstain)
        _assert_sdk_compatible(self, paraphrase, "answer_correctness")
        self.assertTrue(_info(paraphrase)["pass"])
        self.assertIn("key_fact_coverage", (paraphrase.comment or ""))

    def test_evaluator_failure_returns_safe_sdk_compatible_result(self):
        # Passing malformed types should not raise; evaluators must fail safely
        res_r = evaluate_retrieval_relevance(None, None, None)  # type: ignore[arg-type]
        res_f = evaluate_faithfulness(None, None, None)  # type: ignore[arg-type]
        res_c = evaluate_answer_correctness(None, None, None)  # type: ignore[arg-type]
        for res, key in [
            (res_r, "retrieval_relevance"),
            (res_f, "faithfulness_groundedness"),
            (res_c, "answer_correctness"),
        ]:
            _assert_sdk_compatible(self, res, key)
            self.assertEqual(res.score, 0.0)
            self.assertFalse(_info(res)["pass"])
            self.assertTrue(str(res.comment).startswith("evaluator_error"))
            self.assertTrue(str(_info(res).get("explanation", "")).startswith("evaluator_error"))

        # Judge exception path: faithfulness/correctness/relevance should fallback/fail safely
        def exploding_judge(system, user):
            raise RuntimeError("judge boom")

        inputs = {"question": "q"}
        ref = {
            "reference_answer": "a",
            "expected_key_facts": ["a"],
            "expected_supporting_information": ["a"],
            "expected_relevant_chunks": [],
            "expected_abstention": {"should_abstain": False, "acceptable_responses": [], "rationale": ""},
        }
        out = {"final_answer": "a", "retrieved_context": "a", "retrieved_chunks": [], "retrieved_sources": []}
        # relevance falls back to deterministic on judge failure; others also safe
        rr = evaluate_retrieval_relevance(inputs, out, ref, judge=exploding_judge)
        ff = evaluate_faithfulness(inputs, out, ref, judge=exploding_judge)
        cc = evaluate_answer_correctness(inputs, out, ref, judge=exploding_judge)
        _assert_sdk_compatible(self, rr, "retrieval_relevance")
        _assert_sdk_compatible(self, ff, "faithfulness_groundedness")
        _assert_sdk_compatible(self, cc, "answer_correctness")
        self.assertIn("judge_failed_fallback_deterministic", rr.comment or "")


if __name__ == "__main__":
    unittest.main()
