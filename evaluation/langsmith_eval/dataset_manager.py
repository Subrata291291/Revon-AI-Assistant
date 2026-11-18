"""Dataset validation, redaction, and LangSmith sync utilities."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from langsmith import Client

from .client import create_langsmith_client
from .dataset_schema import load_dataset, validate_dataset
from .redaction import redact_obj

DEFAULT_DATASET_NAME = "revon-rag-qa-eval"
# Dedicated evaluation experiment/trace project prefix (not production project).
DEFAULT_PROJECT = "revon-policy-chatbot-eval"

EXAMPLE_ID_NAMESPACE = uuid.UUID(
    "a9f0d8e2-6c41-4b75-9a13-8d2e7f5c1b60"
)


def _langsmith_example_uuid(local_id: str) -> str:
    """Generate a stable UUID for each local evaluation example."""
    return str(
        uuid.uuid5(EXAMPLE_ID_NAMESPACE, str(local_id))
    )


def _example_payload(example: dict[str, Any], dataset_version: str | None) -> dict[str, Any]:
    ex_id = str(example["id"])
    inputs = redact_obj(example.get("inputs") or {})
    # keep question and example_id in inputs for target convenience
    inputs["example_id"] = ex_id
    outputs = redact_obj(example.get("outputs") or {})
    metadata = redact_obj(example.get("metadata") or {})
    metadata["example_id"] = ex_id
    if dataset_version:
        metadata["dataset_version"] = dataset_version
    metadata.setdefault("source", "revon_qa_eval_v1")
    return {
        "id": ex_id,
        "example_id": ex_id,
        "inputs": inputs,
        "outputs": outputs,
        "metadata": metadata,
    }


def ensure_dataset(client: Client, name: str, description: str, metadata: dict[str, Any] | None = None):
    if client.has_dataset(dataset_name=name):
        return client.read_dataset(dataset_name=name)
    return client.create_dataset(
        dataset_name=name,
        description=description,
        metadata=metadata or {},
    )


def sync_dataset(
    file_path: str | Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    project_name: str | None = None,
) -> dict[str, Any]:
    """Validate local dataset and idempotently sync examples to LangSmith.

    Redacts payloads before upload. Requires LANGSMITH_API_KEY in environment.
    `project_name` is accepted for backward compatibility/metadata only; this
    function does not mutate process environment or production tracing config.
    """
    data = load_dataset(file_path)
    result = validate_dataset(data)
    if not result.ok:
        raise ValueError("Dataset validation failed: " + "; ".join(result.errors))

    dataset_version = data.get("dataset_version")
    description = (
        f"RevOn RAG QA evaluation dataset (version={dataset_version}). "
        "Synthetic/redacted; includes expected evidence and abstention behavior."
    )
    ds_metadata = redact_obj(
        {
            "dataset_version": dataset_version,
            "corpus_snapshot": data.get("corpus_snapshot", {}),
            "case_types_present": sorted(list(result.case_types_present)),
            "example_count": result.example_count,
        }
    )

    client = create_langsmith_client()
    dataset = ensure_dataset(client, dataset_name, description=description, metadata=ds_metadata)

    # Map existing examples by metadata/example_id or id
    # Map existing examples by stable local example ID.
    existing_by_id: dict[str, Any] = {}

    for ex in client.list_examples(dataset_name=dataset_name):
        md = getattr(ex, "metadata", None) or {}

        if isinstance(md, dict):
            local_id = md.get("example_id") or md.get("id")

            if local_id:
                existing_by_id[str(local_id)] = ex

    to_create = []
    to_update = []

    for example in data.get("examples", []):
        payload = _example_payload(example, dataset_version)
        local_id = payload["id"]

        existing = existing_by_id.get(local_id)

        if existing is None:
            # LangSmith requires UUID for ExampleCreate.id.
            payload["id"] = _langsmith_example_uuid(local_id)
            to_create.append(payload)

        else:
            to_update.append(
                {
                    "id": getattr(existing, "id"),
                    "inputs": payload["inputs"],
                    "outputs": payload["outputs"],
                    "metadata": payload["metadata"],
                }
            )

    created = 0
    updated = 0
    if to_create:
        # create_examples may accept id/example_id depending on SDK; pass both-safe dicts
        client.create_examples(dataset_name=dataset_name, examples=to_create)
        created = len(to_create)
    if to_update:
        # update_examples signature supports updates list
        try:
            client.update_examples(updates=to_update)
            updated = len(to_update)
        except Exception:
            # fallback: attempt create for any that failed update semantics is risky;
            # surface error for explicit handling
            raise

    return {
        "dataset_name": dataset_name,
        "dataset_id": str(getattr(dataset, "id", "")),
        "dataset_version": dataset_version,
        "validated_examples": result.example_count,
        "created": created,
        "updated": updated,
        "warnings": result.warnings,
        "project_name": project_name or DEFAULT_PROJECT,
        "note": "Dataset sync uses explicit LangSmith client; no process env mutation performed.",
    }


def validate_file(file_path: str | Path) -> dict[str, Any]:
    data = load_dataset(file_path)
    result = validate_dataset(data)
    return {
        "ok": result.ok,
        "errors": result.errors,
        "warnings": result.warnings,
        "example_count": result.example_count,
        "case_types_present": sorted(list(result.case_types_present)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RevOn LangSmith evaluation dataset manager")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Validate a local dataset JSON file")
    p_val.add_argument("--file", required=True, help="Path to dataset JSON")

    p_sync = sub.add_parser("sync", help="Validate and sync dataset to LangSmith")
    p_sync.add_argument("--file", required=True, help="Path to dataset JSON")
    p_sync.add_argument("--dataset", default=DEFAULT_DATASET_NAME, help="LangSmith dataset name")
    p_sync.add_argument("--project", default=DEFAULT_PROJECT, help="Dedicated LangSmith project name")

    args = parser.parse_args(argv)

    if args.command == "validate":
        report = validate_file(args.file)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 1

    if args.command == "sync":
        try:
            report = sync_dataset(args.file, dataset_name=args.dataset, project_name=args.project)
            print(json.dumps(redact_obj(report), indent=2, ensure_ascii=False))
            return 0
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}, indent=2), file=sys.stderr)
            return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
