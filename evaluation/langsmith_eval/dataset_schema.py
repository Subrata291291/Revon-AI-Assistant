"""Dataset schema, validation, and case taxonomy for RevOn RAG QA evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_CASE_TYPES = {
    "direct_factual",
    "technical_skills",
    "employment_history",
    "multi_chunk",
    "missing_info",
    "hallucination_prevention",
}

MIN_EXAMPLES = 15

REQUIRED_EXAMPLE_FIELDS = {
    "id",
    "case_type",
    "inputs",
    "outputs",
    "metadata",
}

REQUIRED_OUTPUT_FIELDS = {
    "reference_answer",
    "expected_key_facts",
    "expected_supporting_information",
    "expected_relevant_chunks",
    "expected_abstention",
}

REQUIRED_ABSTENTION_FIELDS = {
    "should_abstain",
    "acceptable_responses",
    "rationale",
}


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    example_count: int
    case_types_present: set[str]


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def validate_example(example: dict[str, Any], index: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    prefix = f"examples[{index}]({example.get('id', '<missing-id>')})"

    missing = REQUIRED_EXAMPLE_FIELDS - set(example.keys())
    if missing:
        errors.append(f"{prefix}: missing fields {sorted(missing)}")
        return errors, warnings

    if not _is_nonempty_str(example.get("id")):
        errors.append(f"{prefix}: id must be a non-empty string")

    case_type = example.get("case_type")
    if case_type not in REQUIRED_CASE_TYPES:
        errors.append(f"{prefix}: case_type must be one of {sorted(REQUIRED_CASE_TYPES)}")

    inputs = example.get("inputs")
    if not isinstance(inputs, dict) or not _is_nonempty_str(inputs.get("question")):
        errors.append(f"{prefix}: inputs.question must be a non-empty string")

    outputs = example.get("outputs")
    if not isinstance(outputs, dict):
        errors.append(f"{prefix}: outputs must be an object")
        return errors, warnings

    missing_out = REQUIRED_OUTPUT_FIELDS - set(outputs.keys())
    if missing_out:
        errors.append(f"{prefix}: outputs missing fields {sorted(missing_out)}")
    else:
        if not isinstance(outputs.get("reference_answer"), str) or not outputs.get("reference_answer").strip():
            errors.append(f"{prefix}: outputs.reference_answer must be a non-empty string")
        if not isinstance(outputs.get("expected_key_facts"), list):
            errors.append(f"{prefix}: outputs.expected_key_facts must be a list")
        if not isinstance(outputs.get("expected_supporting_information"), list):
            errors.append(f"{prefix}: outputs.expected_supporting_information must be a list")
        if not isinstance(outputs.get("expected_relevant_chunks"), list):
            errors.append(f"{prefix}: outputs.expected_relevant_chunks must be a list")

        abst = outputs.get("expected_abstention")
        if not isinstance(abst, dict):
            errors.append(f"{prefix}: outputs.expected_abstention must be an object")
        else:
            missing_abst = REQUIRED_ABSTENTION_FIELDS - set(abst.keys())
            if missing_abst:
                errors.append(f"{prefix}: expected_abstention missing fields {sorted(missing_abst)}")
            if not isinstance(abst.get("should_abstain"), bool):
                errors.append(f"{prefix}: expected_abstention.should_abstain must be boolean")
            if not isinstance(abst.get("acceptable_responses"), list):
                errors.append(f"{prefix}: expected_abstention.acceptable_responses must be a list")
            if not isinstance(abst.get("rationale"), str) or not abst.get("rationale").strip():
                errors.append(f"{prefix}: expected_abstention.rationale must be a non-empty string")

            if abst.get("should_abstain") is True:
                if len(abst.get("acceptable_responses") or []) < 1:
                    errors.append(f"{prefix}: abstention cases must include >=1 acceptable_responses")
            else:
                # Non-abstention: key facts/supporting info should generally be present
                if len(outputs.get("expected_key_facts") or []) == 0:
                    warnings.append(f"{prefix}: non-abstention example has empty expected_key_facts")

        # Chunk annotations quality (for precision/recall)
        for j, chunk in enumerate(outputs.get("expected_relevant_chunks") or []):
            if not isinstance(chunk, dict):
                errors.append(f"{prefix}: expected_relevant_chunks[{j}] must be an object")
                continue
            if not _is_nonempty_str(chunk.get("evidence_id")):
                errors.append(
                    f"{prefix}: expected_relevant_chunks[{j}] must include stable non-empty evidence_id"
                )
            # Require sufficiently precise annotation beyond filename/page alone
            precise_keys = {"text_must_contain", "text_excerpt_hint", "chunk_text_hash", "content_contains"}
            if not (precise_keys & set(chunk.keys())):
                errors.append(
                    f"{prefix}: expected_relevant_chunks[{j}] must include a precise evidence annotation "
                    f"(one of {sorted(precise_keys)}), not filename/page alone"
                )

    metadata = example.get("metadata")
    if not isinstance(metadata, dict):
        errors.append(f"{prefix}: metadata must be an object")
    else:
        if metadata.get("case_type") and metadata.get("case_type") != case_type:
            errors.append(f"{prefix}: metadata.case_type must match case_type")
        if metadata.get("sensitivity") not in (None, "synthetic", "redacted", "none"):
            warnings.append(f"{prefix}: metadata.sensitivity should be synthetic|redacted|none")

    return errors, warnings


def validate_dataset(data: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    examples = data.get("examples")
    if not isinstance(examples, list):
        return ValidationResult(False, ["dataset.examples must be a list"], [], 0, set())

    if len(examples) < MIN_EXAMPLES:
        errors.append(f"dataset must contain at least {MIN_EXAMPLES} examples (found {len(examples)})")

    seen_ids: set[str] = set()
    case_types_present: set[str] = set()

    for i, ex in enumerate(examples):
        if not isinstance(ex, dict):
            errors.append(f"examples[{i}] must be an object")
            continue
        ex_errors, ex_warnings = validate_example(ex, i)
        errors.extend(ex_errors)
        warnings.extend(ex_warnings)
        ex_id = ex.get("id")
        if isinstance(ex_id, str):
            if ex_id in seen_ids:
                errors.append(f"duplicate example id: {ex_id}")
            seen_ids.add(ex_id)
        ct = ex.get("case_type")
        if isinstance(ct, str):
            case_types_present.add(ct)

    missing_cases = REQUIRED_CASE_TYPES - case_types_present
    if missing_cases:
        errors.append(f"dataset missing required case types: {sorted(missing_cases)}")

    corpus_snapshot = data.get("corpus_snapshot")
    if not isinstance(corpus_snapshot, dict):
        errors.append("dataset.corpus_snapshot must be an object documenting the corpus/index snapshot")
    else:
        for key in ("index", "namespace", "embedding", "description"):
            if key not in corpus_snapshot:
                warnings.append(f"corpus_snapshot missing recommended field: {key}")

    dataset_version = data.get("dataset_version")
    if not _is_nonempty_str(dataset_version):
        warnings.append("dataset_version should be a non-empty string")

    return ValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        example_count=len(examples),
        case_types_present=case_types_present,
    )


def load_dataset(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_dataset_file(path: str | Path) -> ValidationResult:
    data = load_dataset(path)
    return validate_dataset(data)
