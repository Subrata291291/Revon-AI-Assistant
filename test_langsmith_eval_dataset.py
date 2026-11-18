import json
import unittest
from pathlib import Path

from evaluation.langsmith_eval.dataset_schema import (
    MIN_EXAMPLES,
    REQUIRED_CASE_TYPES,
    load_dataset,
    validate_dataset,
    validate_dataset_file,
)
from evaluation.langsmith_eval.redaction import redact_obj, redact_text


DATASET_PATH = Path(__file__).resolve().parent / "evaluation" / "langsmith_eval" / "datasets" / "revon_qa_eval_v1.json"


class TestLangsmithEvalDataset(unittest.TestCase):
    def test_v1_dataset_is_valid_and_covers_requirements(self):
        result = validate_dataset_file(DATASET_PATH)
        self.assertTrue(result.ok, msg=f"errors={result.errors} warnings={result.warnings}")
        self.assertGreaterEqual(result.example_count, MIN_EXAMPLES)
        self.assertEqual(result.case_types_present, REQUIRED_CASE_TYPES)

        data = load_dataset(DATASET_PATH)
        self.assertIsInstance(data.get("corpus_snapshot"), dict)

        ids = set()
        for ex in data["examples"]:
            self.assertIn(ex["case_type"], REQUIRED_CASE_TYPES)
            ids.add(ex["id"])
            abst = ex["outputs"]["expected_abstention"]
            self.assertIn("should_abstain", abst)
            self.assertIn("acceptable_responses", abst)
            self.assertIn("rationale", abst)
            self.assertIsInstance(ex["outputs"]["expected_key_facts"], list)
            self.assertIsInstance(ex["outputs"]["expected_supporting_information"], list)
            self.assertIsInstance(ex["outputs"]["expected_relevant_chunks"], list)
            # precise annotations / stable evidence IDs for chunk-level metrics
            for chunk in ex["outputs"]["expected_relevant_chunks"]:
                self.assertTrue(chunk.get("evidence_id"))
                self.assertTrue(
                    any(k in chunk for k in ("text_must_contain", "text_excerpt_hint", "chunk_text_hash", "content_contains")),
                    msg=f"chunk annotation not precise enough: {chunk}",
                )
        self.assertEqual(len(ids), len(data["examples"]))

    def test_validator_rejects_missing_case_types_and_insufficient_count(self):
        bad = {
            "dataset_version": "bad",
            "corpus_snapshot": {"index": "i", "namespace": "n", "embedding": {}, "description": "d"},
            "examples": [
                {
                    "id": "x1",
                    "case_type": "direct_factual",
                    "inputs": {"question": "q"},
                    "outputs": {
                        "reference_answer": "a",
                        "expected_key_facts": ["f"],
                        "expected_supporting_information": ["s"],
                        "expected_relevant_chunks": [],
                        "expected_abstention": {"should_abstain": False, "acceptable_responses": [], "rationale": "r"},
                    },
                    "metadata": {"case_type": "direct_factual"},
                }
            ],
        }
        result = validate_dataset(bad)
        self.assertFalse(result.ok)
        self.assertTrue(any("at least" in e for e in result.errors))
        self.assertTrue(any("missing required case types" in e for e in result.errors))

    def test_redaction_removes_pii_and_secrets_but_preserves_critical_facts(self):
        text = (
            "Contact jane.doe@example.com or +1 415-555-2671. "
            "Employee ID: EMP123456789. "
            "Senior Software Engineer at RevOn Technologies worked on Project Atlas "
            "using Python and FastAPI from January 2021 to March 2024. "
            "api_key=sk-secret-ABCDEF1234567890"
        )
        red = redact_text(text)
        self.assertNotIn("jane.doe@example.com", red)
        self.assertNotIn("415-555-2671", red)
        self.assertNotIn("EMP123456789", red)
        self.assertIn("[REDACTED_EMAIL]", red)
        self.assertIn("RevOn Technologies", red)
        self.assertIn("Project Atlas", red)
        self.assertIn("Python", red)
        self.assertIn("FastAPI", red)
        self.assertIn("January 2021", red)
        self.assertIn("March 2024", red)

        obj = {"token": "secret", "nested": {"email": "a@b.com", "company": "RevOn Technologies"}}
        red_obj = redact_obj(obj)
        self.assertEqual(red_obj["token"], "[REDACTED_SECRET]")
        self.assertIn("[REDACTED_EMAIL]", json.dumps(red_obj))
        self.assertEqual(red_obj["nested"]["company"], "RevOn Technologies")


if __name__ == "__main__":
    unittest.main()
