import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

from langsmith.evaluation import EvaluationResult

from evaluation.langsmith_eval import run_experiment as run_mod


class DummyResults(list):
    def __init__(self, rows=None, experiment_name="revon-policy-chatbot-eval-deadbeef"):
        super().__init__(rows or [])
        self.experiment_name = experiment_name


class TestLangsmithEvalRunExperiment(unittest.TestCase):
    def _run(self, list_examples_return, **kwargs):
        client = MagicMock()
        client.list_examples.return_value = list(list_examples_return)
        fake_results = DummyResults([])

        with patch.object(run_mod, "create_langsmith_client", return_value=client), \
             patch.object(run_mod, "evaluate", return_value=fake_results) as mock_eval, \
             patch.object(run_mod, "_write_local_report", return_value=None):
            report = run_mod.run_experiment(
                dataset_name="revon-rag-qa-eval",
                experiment_prefix="revon-policy-chatbot-eval",
                trace_project="revon-policy-chatbot-eval",
                max_concurrency=1,
                judge_enabled=False,
                sync=False,
                allow_in_process_risk=True,
                **kwargs,
            )
        return client, mock_eval, report

    def test_limits_to_3_examples_and_passes_limited_actual_input(self):
        examples = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]
        client, mock_eval, report = self._run(examples, max_examples=3)

        client.list_examples.assert_called_once()
        # preserve ordering: first 3 in existing order
        self.assertEqual(list(client.list_examples.call_args.kwargs.get("dataset_name") and [client.list_examples.call_args.kwargs["dataset_name"]] or []), ["revon-rag-qa-eval"])
        self.assertEqual(mock_eval.call_args.kwargs["data"], examples[:3])
        self.assertEqual(report["max_examples"], 3)
        self.assertEqual(mock_eval.call_args.kwargs["max_concurrency"], 1)
        self.assertEqual(mock_eval.call_args.kwargs["experiment_prefix"], "revon-policy-chatbot-eval")

    def test_limits_to_1_example(self):
        examples = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        client, mock_eval, report = self._run(examples, max_examples=1)

        self.assertEqual(mock_eval.call_args.kwargs["data"], [examples[0]])
        self.assertEqual(report["max_examples"], 1)

    def test_requesting_more_examples_than_available_uses_all(self):
        examples = [{"id": "x"}, {"id": "y"}]
        client, mock_eval, report = self._run(examples, max_examples=10)

        # actual evaluation input limited to available examples, preserving order
        self.assertEqual(mock_eval.call_args.kwargs["data"], examples)
        self.assertEqual(len(mock_eval.call_args.kwargs["data"]), 2)
        self.assertEqual(report["max_examples"], 10)

    def test_rejects_zero_and_negative_values(self):
        for bad in (0, -1, -5):
            with self.assertRaises(ValueError):
                run_mod.run_experiment(
                    dataset_name="revon-rag-qa-eval",
                    allow_in_process_risk=True,
                    max_examples=bad,
                )

    def test_cli_rejects_zero_and_negative_and_dry_run_shows_selected_count(self):
        for bad in ("0", "-2"):
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = run_mod.main(["--dry-run", "--max-examples", bad])
            self.assertNotEqual(code, 0)
            payload = json.loads((err.getvalue() or out.getvalue()).strip())
            self.assertFalse(payload.get("ok", True))

        out = io.StringIO()
        with redirect_stdout(out):
            code = run_mod.main(["--dry-run", "--max-examples", "3"])
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["max_examples"], 3)
        self.assertEqual(payload["sync"], False)
        self.assertEqual(payload["max_concurrency"], 1)
        self.assertEqual(payload["experiment_prefix"], "revon-policy-chatbot-eval")
        self.assertTrue(payload["isolation"]["sdk_experiment_context_preserved_for_association"])
        self.assertTrue(payload["isolation"]["child_traces_follow_experiment_project_when_available"])

    def test_default_dry_run_includes_max_examples_none(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = run_mod.main(["--dry-run"])
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertIn("max_examples", payload)
        self.assertIsNone(payload["max_examples"])

    def _run_with_results(self, fake_results):
        client = MagicMock()
        client.list_examples.return_value = [{"id": 1}]
        with patch.object(run_mod, "create_langsmith_client", return_value=client), \
             patch.object(run_mod, "evaluate", return_value=fake_results), \
             patch.object(run_mod, "_write_local_report", return_value=None):
            return run_mod.run_experiment(
                dataset_name="revon-rag-qa-eval",
                experiment_prefix="revon-policy-chatbot-eval",
                trace_project="revon-policy-chatbot-eval",
                max_concurrency=1,
                judge_enabled=False,
                sync=False,
                allow_in_process_risk=True,
            )

    def test_evaluator_failures_mark_report_unsuccessful_and_expose_errors(self):
        # Simulate SDK 0.12.6 evaluator exception handling: error feedback results
        error_rows = [
            {
                "evaluation_results": {
                    "results": [
                        EvaluationResult(key="retrieval_relevance", score=None, comment="ValueError('bad')", extra={"error": True}),
                        EvaluationResult(key="faithfulness_groundedness", score=None, comment="ValueError('bad')", extra={"error": True}),
                        EvaluationResult(key="answer_correctness", score=None, comment="ValueError('bad')", extra={"error": True}),
                    ]
                },
                "example": {"id": "ex-1"},
            }
        ]
        report = self._run_with_results(DummyResults(error_rows))
        self.assertFalse(report["ok"])
        self.assertFalse(report["evaluation_ok"])
        self.assertGreaterEqual(len(report["summary"]["evaluator_errors"]), 3)
        self.assertTrue(all(e.get("error") for e in report["summary"]["evaluator_errors"]))
        self.assertEqual(report["summary"]["metrics"], {})

    def test_missing_expected_metrics_cannot_be_reported_successful(self):
        # Only one metric produced -> missing other expected metrics
        rows = [
            {
                "evaluation_results": {
                    "results": [
                        EvaluationResult(key="retrieval_relevance", score=0.9, comment="ok", evaluator_info={"pass": True}),
                    ]
                },
                "example": {"id": "ex-1"},
            }
        ]
        report = self._run_with_results(DummyResults(rows))
        self.assertFalse(report["ok"])
        self.assertFalse(report["evaluation_ok"])
        self.assertIn("faithfulness_groundedness", report["summary"]["missing_expected_metrics"])
        self.assertIn("answer_correctness", report["summary"]["missing_expected_metrics"])

    def test_empty_metric_summaries_cannot_be_reported_successful(self):
        # No evaluation result items at all
        rows = [{"evaluation_results": {"results": []}, "example": {"id": "ex-1"}}]
        report = self._run_with_results(DummyResults(rows))
        self.assertFalse(report["ok"])
        self.assertFalse(report["evaluation_ok"])
        self.assertEqual(report["summary"]["metrics"], {})
        self.assertTrue(report["summary"]["missing_expected_metrics"])

    def test_successful_report_when_all_expected_metrics_present_without_errors(self):
        def ok_result(key, score):
            return EvaluationResult(key=key, score=score, comment="ok", evaluator_info={"pass": True, "explanation": "ok"})

        rows = [
            {
                "evaluation_results": {
                    "results": [
                        ok_result("retrieval_relevance", 0.9),
                        ok_result("faithfulness_groundedness", 0.85),
                        ok_result("answer_correctness", 0.95),
                    ]
                },
                "example": {"id": "ex-1"},
            }
        ]
        report = self._run_with_results(DummyResults(rows))
        self.assertTrue(report["ok"])
        self.assertTrue(report["evaluation_ok"])
        self.assertEqual(report["summary"]["evaluator_errors"], [])
        self.assertEqual(report["summary"]["missing_expected_metrics"], [])
        self.assertEqual(set(report["summary"]["metrics"].keys()), {"retrieval_relevance", "faithfulness_groundedness", "answer_correctness"})


if __name__ == "__main__":
    unittest.main()
