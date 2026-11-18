"""Run LangSmith evaluation experiments for RevOn RAG (real workflow target).

Defaults are safe: no live experiment is executed automatically, concurrency is
forced to 1, and judge is disabled by default.

Isolation notes:
- Evaluation experiment/project names are prefixed via `experiment_prefix`
  (LangSmith SDK appends a UUID suffix for the experiment project).
- Under LangSmith evaluate (SDK 0.12.6), target/child traces preserve the
  SDK-provided experiment project/parent context for correct example/run
  association, while remaining isolated from the production project.
- `trace_project` is a non-production fallback tracing project used only when no
  safe SDK experiment context is available (e.g. direct/non-experiment calls);
  it must never be the production project.
- No global environment mutation is used to redirect production requests; eval
  process-scoped env is set only within this dedicated CLI/process execution.
- Global module-state swaps in target require a dedicated evaluation process
  (explicit guard). `max_concurrency=1` alone does not isolate from production
  traffic in the same process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langsmith.evaluation import evaluate

from .client import create_langsmith_client
from .dataset_manager import DEFAULT_DATASET_NAME, DEFAULT_PROJECT, sync_dataset
from .evaluators import (
    evaluate_answer_correctness,
    evaluate_faithfulness,
    evaluate_retrieval_relevance,
)
from .judge import create_judge, load_judge_config_from_env
from .redaction import redact_obj, sanitize_for_logging
from .target import langsmith_target

DEFAULT_EXPERIMENT_PREFIX = "revon-policy-chatbot-eval"
DEFAULT_TRACE_PROJECT = "revon-policy-chatbot-eval"


def _git_sha() -> str | None:
    try:
        root = Path(__file__).resolve().parents[3]
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True)
        return out.strip() or None
    except Exception:
        return None


def _build_evaluators(judge_enabled: bool = False):
    judge = create_judge() if judge_enabled else None

    def retrieval_relevance(inputs, outputs, reference_outputs):
        return evaluate_retrieval_relevance(inputs, outputs, reference_outputs, judge=judge)

    def faithfulness_groundedness(inputs, outputs, reference_outputs):
        return evaluate_faithfulness(inputs, outputs, reference_outputs, judge=judge)

    def answer_correctness(inputs, outputs, reference_outputs):
        return evaluate_answer_correctness(inputs, outputs, reference_outputs, judge=judge)

    retrieval_relevance.__name__ = "retrieval_relevance"
    faithfulness_groundedness.__name__ = "faithfulness_groundedness"
    answer_correctness.__name__ = "answer_correctness"
    return [retrieval_relevance, faithfulness_groundedness, answer_correctness], judge


def run_experiment(
    dataset_name: str = DEFAULT_DATASET_NAME,
    experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX,
    trace_project: str | None = None,
    project_name: str | None = None,
    max_concurrency: int = 1,
    judge_enabled: bool = False,
    dataset_file: str | None = None,
    sync: bool = False,
    metadata_extra: dict[str, Any] | None = None,
    allow_in_process_risk: bool = False,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Run a LangSmith evaluation experiment in a dedicated evaluation process.

    `experiment_prefix` controls the LangSmith evaluation experiment/project name
    prefix (SDK appends a UUID suffix). During SDK-driven evaluation, target/child
    traces preserve the SDK-provided experiment project/context for association.
    `trace_project` is the isolated non-production fallback tracing project used
    only when no safe SDK experiment context is available.

    If `max_examples` is provided, it must be > 0 and limits the actual examples
    passed to `evaluate()` (preserving existing dataset order) without modifying
    or uploading the dataset.

    No global env mutation is performed to redirect production requests. Process-
    scoped env vars are set for this evaluation execution context only.
    """
    # Concurrency must be 1 initially / when using capture hooks.
    max_concurrency = 1 if max_concurrency is None else int(max_concurrency)
    if max_concurrency != 1:
        max_concurrency = 1

    if max_examples is not None:
        if not isinstance(max_examples, int) or isinstance(max_examples, bool) or max_examples <= 0:
            raise ValueError("--max-examples must be an integer greater than zero when provided.")

    # Backward-compatible alias: older callers passed project_name.
    if trace_project is None and project_name:
        trace_project = project_name
    if trace_project is None:
        trace_project = DEFAULT_TRACE_PROJECT
    if trace_project == "revon-policy-chatbot":
        raise RuntimeError(
            "trace_project must not be the production project 'revon-policy-chatbot'."
        )

    # Explicit guard: global module-state swaps in target are unsafe if this runs
    # in a process that may also serve production traffic.
    if not allow_in_process_risk and os.getenv("REVON_EVAL_ALLOW_IN_PROCESS_RISK") != "1":
        raise RuntimeError(
            "Evaluation must run in a dedicated process. Use the CLI "
            "(`python -m evaluation.langsmith_eval.run_experiment`) in a separate "
            "process, or pass allow_in_process_risk=True / set "
            "REVON_EVAL_ALLOW_IN_PROCESS_RISK=1 only for explicitly isolated "
            "non-production contexts."
        )

    # Process-scoped evaluation execution context (dedicated process): provide an
    # isolated fallback trace project for non-experiment contexts; do not mutate
    # production .env or force this fallback over SDK experiment context.
    os.environ["REVON_EVAL_TRACE_PROJECT"] = trace_project
    os.environ["REVON_EVAL_ALLOW_IN_PROCESS_RISK"] = (
        "1" if allow_in_process_risk else os.environ.get("REVON_EVAL_ALLOW_IN_PROCESS_RISK", "1")
    )

    client = create_langsmith_client()

    sync_report = None
    if sync:
        if not dataset_file:
            raise ValueError("--sync requires --dataset-file")
        sync_report = sync_dataset(file_path=dataset_file, dataset_name=dataset_name)

    evaluators, judge = _build_evaluators(judge_enabled=judge_enabled)
    judge_cfg = load_judge_config_from_env().safe_metadata()

    metadata = {
        "dataset_name": dataset_name,
        "experiment_prefix": experiment_prefix,
        "trace_project_fallback": trace_project,
        "production_project": "revon-policy-chatbot",
        "max_concurrency": max_concurrency,
        "max_examples": max_examples,
        "git_sha": _git_sha(),
        "judge": judge_cfg,
        "thresholds": {
            "retrieval_relevance": 0.7,
            "faithfulness_groundedness": 0.8,
            "answer_correctness": 0.7,
        },
        "workflow": "answer_conversational_question",
        "isolation": {
            "dedicated_process_required": True,
            "experiment_prefix_controls_experiment_name": True,
            "sdk_experiment_context_preserved": True,
            "child_traces_follow_experiment_project_when_available": True,
            "fallback_trace_project_non_experiment_only": trace_project,
            "production_env_unmodified": True,
        },
        "note": (
            "Real RAG workflow target; retrieval capture via context-local hook; "
            "concurrency forced to 1. Child traces preserve SDK experiment context "
            "for association and never route to production."
        ),
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    # Limit actual examples passed to evaluate() without modifying/uploading dataset.
    eval_data: Any = dataset_name
    if max_examples is not None:
        # Preserve existing dataset/example ordering: take first N as returned.
        eval_data = list(client.list_examples(dataset_name=dataset_name))[:max_examples]

    results = evaluate(
        langsmith_target,
        data=eval_data,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        metadata=redact_obj(metadata),
        max_concurrency=max_concurrency,
        client=client,
    )

    expected_metric_keys = [
        "retrieval_relevance",
        "faithfulness_groundedness",
        "answer_correctness",
    ]
    summary = _summarize_results(results, expected_metric_keys=expected_metric_keys)
    evaluation_ok = bool(summary.get("evaluation_ok"))
    report = {
        "ok": evaluation_ok,
        "evaluation_ok": evaluation_ok,
        "experiment_name": getattr(results, "experiment_name", None),
        "dataset_name": dataset_name,
        "experiment_prefix": experiment_prefix,
        "trace_project_fallback": trace_project,
        "max_examples": max_examples,
        "summary": summary,
        "metadata": redact_obj(metadata),
        "sync": redact_obj(sync_report) if sync_report else None,
    }
    _write_local_report(experiment_prefix, report)
    return report


def _result_field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _result_extra(item: Any) -> dict[str, Any]:
    extra = _result_field(item, "extra", None)
    return extra if isinstance(extra, dict) else {}


def _summarize_results(results: Any, expected_metric_keys: list[str] | None = None) -> dict[str, Any]:
    rows = []
    try:
        # ExperimentResults is iterable in recent SDKs
        for row in results:
            rows.append(row)
    except Exception:
        pass

    metrics: dict[str, list[float]] = {}
    pass_counts: dict[str, list[bool]] = {}
    evaluator_errors: list[dict[str, Any]] = []
    expected = [str(k) for k in (expected_metric_keys or []) if k]
    per_example_keys: list[set[str]] = []

    for idx, row in enumerate(rows):
        eval_results = None
        example = None
        if isinstance(row, dict):
            eval_results = row.get("evaluation_results") or row.get("feedback")
            example = row.get("example")
        else:
            eval_results = getattr(row, "evaluation_results", None)
            example = getattr(row, "example", None)
        items = []
        if isinstance(eval_results, dict):
            items = eval_results.get("results") or []
        elif isinstance(eval_results, list):
            items = eval_results

        example_id = None
        if isinstance(example, dict):
            example_id = example.get("id") or example.get("example_id")
        else:
            example_id = getattr(example, "id", None)
        row_keys: set[str] = set()

        for item in items:
            key = _result_field(item, "key")
            score = _result_field(item, "score")
            passed = _result_field(item, "pass")
            comment = _result_field(item, "comment")
            evaluator_info = _result_field(item, "evaluator_info") or {}
            if not isinstance(evaluator_info, dict):
                evaluator_info = {}
            if passed is None and "pass" in evaluator_info:
                passed = evaluator_info.get("pass")

            extra = _result_extra(item)
            is_error = bool(extra.get("error"))

            if not key:
                continue
            key_s = str(key)

            if is_error:
                evaluator_errors.append(
                    {
                        "example_index": idx,
                        "example_id": str(example_id) if example_id is not None else None,
                        "key": key_s,
                        "error": comment,
                        "source_run_id": str(_result_field(item, "source_run_id") or "") or None,
                    }
                )
                # SDK evaluator failures do not produce valid metric scores.
                continue

            row_keys.add(key_s)
            if score is not None:
                try:
                    metrics.setdefault(key_s, []).append(float(score))
                except Exception:
                    pass
            if passed is not None:
                pass_counts.setdefault(key_s, []).append(bool(passed))

        per_example_keys.append(row_keys)

    summary_metrics: dict[str, Any] = {}
    for key, vals in metrics.items():
        passes = pass_counts.get(key, [])
        summary_metrics[key] = {
            "mean_score": sum(vals) / len(vals) if vals else None,
            "count": len(vals),
            "pass_rate": (sum(1 for p in passes if p) / len(passes)) if passes else None,
        }

    missing_expected_metrics: list[str] = []
    if expected:
        for key in expected:
            if not metrics.get(key):
                missing_expected_metrics.append(key)
            else:
                # ensure every example produced this metric
                for row_keys in per_example_keys:
                    if key not in row_keys:
                        missing_expected_metrics.append(key)
                        break

    evaluation_ok = len(rows) > 0 and len(evaluator_errors) == 0 and len(missing_expected_metrics) == 0 and bool(summary_metrics)

    return {
        "examples": len(rows),
        "metrics": summary_metrics,
        "expected_metric_keys": expected,
        "missing_expected_metrics": missing_expected_metrics,
        "evaluator_errors": evaluator_errors,
        "evaluation_ok": evaluation_ok,
    }


def _write_local_report(prefix: str, report: dict[str, Any]) -> Path | None:
    try:
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = results_dir / f"{prefix}-{ts}-{uuid.uuid4().hex[:8]}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(redact_obj(report), f, indent=2, ensure_ascii=False)
        return path
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RevOn LangSmith RAG evaluation experiment")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--experiment-prefix",
        default=DEFAULT_EXPERIMENT_PREFIX,
        help="LangSmith evaluation experiment/project name prefix (SDK appends UUID suffix)",
    )
    parser.add_argument(
        "--trace-project",
        default=DEFAULT_TRACE_PROJECT,
        help=(
            "Fallback dedicated LangSmith tracing project for non-experiment contexts "
            "(isolated from production). SDK experiment context is preserved during evaluate()."
        ),
    )
    parser.add_argument(
        "--project",
        dest="trace_project_alias",
        default=None,
        help="Backward-compatible alias for --trace-project",
    )
    parser.add_argument("--max-concurrency", type=int, default=1, help="Forced to 1 for safe capture")
    parser.add_argument("--judge", action="store_true", help="Enable isolated LLM-as-judge if configured")
    parser.add_argument("--sync", action="store_true", help="Sync dataset before running")
    parser.add_argument("--dataset-file", help="Local dataset JSON path (required with --sync)")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional: limit evaluation to the first N dataset examples (> 0). Preserves dataset order.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config only; do not execute evaluate()")
    args = parser.parse_args(argv)

    trace_project = args.trace_project_alias or args.trace_project or DEFAULT_TRACE_PROJECT
    max_examples = args.max_examples
    if max_examples is not None and max_examples <= 0:
        print(
            json.dumps({"ok": False, "error": "--max-examples must be an integer greater than zero when provided."}, indent=2),
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        cfg = {
            "dataset": args.dataset,
            "experiment_prefix": args.experiment_prefix,
            "trace_project_fallback": trace_project,
            "production_project": "revon-policy-chatbot",
            "max_concurrency": 1,
            "max_examples": max_examples,
            "judge_enabled": bool(args.judge),
            "sync": bool(args.sync),
            "dataset_file": args.dataset_file,
            "isolation": {
                "experiment_prefix_controls_experiment_name": True,
                "sdk_experiment_context_preserved_for_association": True,
                "child_traces_follow_experiment_project_when_available": True,
                "fallback_trace_project_non_experiment_only": trace_project,
                "dedicated_process_required_for_live_run": True,
                "global_env_mutation_for_production_redirect": False,
            },
            "note": "No live experiment executed in dry-run.",
        }
        print(json.dumps(redact_obj(cfg), indent=2))
        return 0

    try:
        # CLI execution is a dedicated process; allow in-process global swaps for
        # this isolated evaluation process only.
        report = run_experiment(
            dataset_name=args.dataset,
            experiment_prefix=args.experiment_prefix,
            trace_project=trace_project,
            max_concurrency=1,
            judge_enabled=bool(args.judge),
            dataset_file=args.dataset_file,
            sync=bool(args.sync),
            allow_in_process_risk=True,
            max_examples=max_examples,
        )
        print(json.dumps(sanitize_for_logging(report), indent=2, ensure_ascii=False))
        return 0 if report.get("evaluation_ok") or report.get("ok") else 2
    except Exception as e:
        print(json.dumps({"ok": False, "evaluation_ok": False, "error": str(e)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
