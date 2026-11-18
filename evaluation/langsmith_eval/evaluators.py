"""LangSmith evaluators for RevOn RAG QA evaluation.

Metrics:
- retrieval_relevance: evidence/fact coverage + chunk relevance, with
  precision@k/recall@k/F1@k when ground-truth chunk annotations are available.
  Independent of answer correctness and abstention correctness.
- faithfulness_groundedness: claims/citations supported by retrieved context only
  (reference answer never used as evidence). Isolated LLM-as-judge for real eval;
  deterministic rule-based checks for unit tests.
- answer_correctness: semantic/key-fact correctness vs reference/key facts with
  separate abstention handling (not exact-string-only).

Compatible with langsmith==0.12.6 simplified evaluator signatures:
  (inputs, outputs, reference_outputs) -> dict/EvaluationResult
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

from langsmith.evaluation import EvaluationResult

from .judge import JudgeFn

try:
    from retrieval_service import DOCUMENT_NOT_FOUND_MESSAGE
except Exception:  # pragma: no cover - fallback for isolated test envs
    DOCUMENT_NOT_FOUND_MESSAGE = "The uploaded documents do not contain the requested information."


def _eval_result(
    key: str,
    score: float,
    *,
    comment: str,
    passed: bool | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Build an SDK 0.12.6-compatible evaluation result.

    Preserves all diagnostic details in supported fields (`comment` and
    `evaluator_info`) because `EvaluationResult` forbids extra top-level fields.
    """
    info: dict[str, Any] = {}
    if diagnostics:
        info.update(diagnostics)
    if passed is not None:
        info["pass"] = bool(passed)
    # Keep explanation mirrored in diagnostics for reliable programmatic access.
    info.setdefault("explanation", comment)
    return EvaluationResult(
        key=key,
        score=float(_clamp01(_safe_float(score))),
        comment=comment,
        evaluator_info=info,
    )


# ----------------------------
# Shared helpers
# ----------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9_\-/]{1,}", _normalize_text(text)))


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "for", "of", "to", "at", "in", "on", "by", "with", "and", "or", "as",
    "per", "from", "that", "this", "it", "its",
}


def _content_tokens(text: str) -> set[str]:
    toks = _tokenize(text)
    return {t for t in toks if t not in _STOPWORDS}


def _number_date_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d{1,4}(?:[/-]\d{1,4})*\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", _normalize_text(text)))


def _contains_phrase(haystack: str, needle: str) -> bool:
    h = _normalize_text(haystack)
    n = _normalize_text(needle)
    if not n:
        return False
    if n in h:
        return True
    nt = _content_tokens(n)
    if not nt:
        nt = _tokenize(n)
    if not nt:
        return False
    ht = _content_tokens(h)
    if not ht:
        ht = _tokenize(h)
    overlap = len(nt & ht) / max(1, len(nt))
    # all numeric/date tokens in needle must appear for factual phrases
    nnums = _number_date_tokens(n)
    nums_ok = True
    if nnums:
        nums_ok = all(num in h for num in nnums)
        if not nums_ok:
            return False
    if overlap >= 0.75:
        return True
    if nnums and nums_ok and overlap >= 0.6:
        return True
    # require all key content tokens for short facts
    if len(nt) <= 4 and nt.issubset(ht):
        return True
    return False


def _evidence_supported_in_text(evidence: Any, text: str) -> bool:
    if isinstance(evidence, str):
        return _contains_phrase(text, evidence)
    if isinstance(evidence, dict):
        # support either explicit text hints or fact strings
        for key in ("text_must_contain", "text_excerpt_hint", "content_contains", "fact", "value"):
            val = evidence.get(key)
            if isinstance(val, str) and _contains_phrase(text, val):
                return True
        # if dict has no textual hints, try all string values
        str_vals = [v for v in evidence.values() if isinstance(v, str)]
        return any(_contains_phrase(text, v) for v in str_vals) if str_vals else False
    return False


def _chunk_matches_annotation(chunk: dict[str, Any], ann: dict[str, Any]) -> bool:
    """Match a retrieved chunk to a ground-truth chunk annotation.

    Requires stable evidence_id-aware, sufficiently precise matching (not
    filename/page alone): a precise text hint must match, with page/filename
    used as corroborating signals when provided.
    """
    if not isinstance(chunk, dict) or not isinstance(ann, dict):
        return False
    text = str(chunk.get("text") or "")
    # precise textual match is mandatory for counting a GT chunk as retrieved
    precise_hit = False
    for key in ("text_must_contain", "text_excerpt_hint", "content_contains"):
        hint = ann.get(key)
        if isinstance(hint, str) and hint.strip():
            precise_hit = _contains_phrase(text, hint)
            break
    if not precise_hit:
        # allow chunk_text_hash equality if both provided
        ann_hash = ann.get("chunk_text_hash")
        chunk_hash = chunk.get("chunk_text_hash")
        precise_hit = bool(ann_hash and chunk_hash and ann_hash == chunk_hash)
    if not precise_hit:
        return False

    # corroborate with metadata when annotation specifies it
    ann_page = ann.get("page_number", ann.get("page"))
    if ann_page is not None and str(chunk.get("page_number")) != str(ann_page):
        # allow text-precise hit even if page differs in formatting, but prefer match;
        # treat mismatch as non-match only when both are concrete and different ints/strings
        try:
            if int(chunk.get("page_number")) != int(ann_page):
                return False
        except Exception:
            if str(chunk.get("page_number")).strip() != str(ann_page).strip():
                return False
    ann_file = ann.get("filename")
    if isinstance(ann_file, str) and ann_file.strip():
        if _normalize_text(str(chunk.get("filename") or "")) != _normalize_text(ann_file):
            # filename mismatch with precise text hit: accept only if text hint is strong
            # (already required) but keep strict to avoid filename/page-only claims
            return False
    return True


def _compute_precision_recall(
    retrieved_chunks: list[dict[str, Any]],
    expected_relevant_chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Compute chunk-level precision/recall/F1 when GT annotations are available.

    Returns None when annotations are unavailable/unspecified in a way that
    prevents exact chunk-level metrics.
    """
    if not isinstance(expected_relevant_chunks, list):
        return None
    # treat explicit empty list as a valid annotation (e.g. missing-info => none relevant)
    # caller decides applicability; here we compute from provided annotation set.
    gt = [a for a in expected_relevant_chunks if isinstance(a, dict)]
    retrieved = [c for c in (retrieved_chunks or []) if isinstance(c, dict)]

    matched_gt: set[int] = set()
    relevant_retrieved = 0
    for chunk in retrieved:
        hit_index = None
        for i, ann in enumerate(gt):
            if i in matched_gt:
                continue
            if _chunk_matches_annotation(chunk, ann):
                hit_index = i
                break
        if hit_index is not None:
            matched_gt.add(hit_index)
            relevant_retrieved += 1

    retrieved_count = len(retrieved)
    gt_count = len(gt)

    precision = (relevant_retrieved / retrieved_count) if retrieved_count > 0 else (1.0 if gt_count == 0 else 0.0)
    recall = (relevant_retrieved / gt_count) if gt_count > 0 else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "retrieval_precision": _clamp01(precision),
        "retrieval_recall": _clamp01(recall),
        "retrieval_f1": _clamp01(f1),
        "relevant_retrieved_count": relevant_retrieved,
        "retrieved_count": retrieved_count,
        "ground_truth_relevant_count": gt_count,
    }


def _is_abstention_answer(answer: str, acceptable: list[str] | None = None) -> bool:
    a = _normalize_text(answer)
    if not a:
        return False
    candidates = [DOCUMENT_NOT_FOUND_MESSAGE] + list(acceptable or [])
    for cand in candidates:
        if not isinstance(cand, str) or not cand.strip():
            continue
        c = _normalize_text(cand)
        if c == a or c in a or a in c:
            return True
    # generic grounded abstention cues (support legacy and PDF-grounded phrasings)
    cues = [
        "uploaded documents do not contain",
        "uploaded pdfs do not contain",
        "documents do not contain the requested information",
        "not contain sufficient information",
        "not contain enough information",
        "cannot find relevant information",
        "no relevant information",
    ]
    return any(cue in a for cue in cues)


def _extract_citations(answer: str) -> list[int]:
    return [int(m) for m in re.findall(r"\[Source\s+(\d+)\]", answer or "")]


# ----------------------------
# Retrieval relevance evaluator
# ----------------------------

RETRIEVAL_RELEVANCE_SYSTEM = (
    "You are a strict retrieval relevance evaluator for a RAG system. "
    "Score retrieval relevance independently of final answer correctness and "
    "independently of whether abstention was correct. Return ONLY valid JSON."
)

def _retrieval_relevance_with_judge(
    judge: JudgeFn,
    *,
    question: str,
    retrieved_context: str,
    retrieved_chunks: list[dict[str, Any]],
    expected_supporting_information: list[Any],
    expected_key_facts: list[Any],
    expected_abstention: dict[str, Any],
) -> dict[str, Any]:
    user = (
        "Evaluate how relevant the retrieved chunks are to answering the question "
        "and covering the expected evidence/facts. Do NOT give high relevance just "
        "because the system abstained correctly; irrelevant chunks must score low in "
        "missing-information cases.\n\n"
        f"Question: {question}\n"
        f"Expected abstention: {expected_abstention}\n"
        f"Expected supporting information: {expected_supporting_information}\n"
        f"Expected key facts: {expected_key_facts}\n"
        f"Retrieved chunks: {retrieved_chunks}\n"
        f"Retrieved context: {retrieved_context}\n\n"
        "Return JSON with keys: score (0-1 float), evidence_coverage (0-1 float), "
        "explanation (string), relevant_chunk_indexes (list of ints)."
    )
    return judge(RETRIEVAL_RELEVANCE_SYSTEM, user)


def evaluate_retrieval_relevance(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
    *,
    judge: JudgeFn | None = None,
    threshold: float = 0.7,
) -> EvaluationResult:
    try:
        if not isinstance(inputs, dict) or not isinstance(outputs, dict) or not isinstance(reference_outputs, dict):
            comment = "evaluator_error: invalid_input_types"
            return _eval_result(
                "retrieval_relevance",
                0.0,
                comment=comment,
                passed=False,
                diagnostics={
                    "evidence_coverage": 0.0,
                    "retrieval_precision": None,
                    "retrieval_recall": None,
                    "retrieval_f1": None,
                    "relevant_retrieved_count": None,
                    "retrieved_count": 0,
                    "ground_truth_relevant_count": None,
                },
            )
        question = str((inputs or {}).get("question") or "")
        retrieved_chunks = list((outputs or {}).get("retrieved_chunks") or [])
        retrieved_context = str((outputs or {}).get("retrieved_context") or "")
        expected_supporting = list((reference_outputs or {}).get("expected_supporting_information") or [])
        expected_facts = list((reference_outputs or {}).get("expected_key_facts") or [])
        expected_relevant_chunks = list((reference_outputs or {}).get("expected_relevant_chunks") or [])
        expected_abstention = (reference_outputs or {}).get("expected_abstention") or {"should_abstain": False}

        pr = _compute_precision_recall(retrieved_chunks, expected_relevant_chunks)

        # Evidence/fact coverage (deterministic baseline): proportion of expected
        # evidence/facts supported by any retrieved chunk/context.
        evidence_items = list(expected_supporting) + list(expected_facts)
        should_abstain = bool((expected_abstention or {}).get("should_abstain"))
        if evidence_items:
            supported = 0
            corpus_text = "\n".join([retrieved_context] + [str(c.get("text") or "") for c in retrieved_chunks])
            for item in evidence_items:
                if _evidence_supported_in_text(item, corpus_text):
                    supported += 1
            evidence_coverage = supported / len(evidence_items)
        else:
            # No positive evidence expected (e.g. missing-info): if no relevant
            # chunks are annotated, retrieving irrelevant chunks is poor retrieval.
            if should_abstain or (isinstance(expected_relevant_chunks, list) and len(expected_relevant_chunks) == 0):
                evidence_coverage = 0.0 if retrieved_chunks else 1.0
            else:
                evidence_coverage = 1.0 if not retrieved_chunks else 0.0

        judge_info = None
        if judge is not None:
            try:
                judged = _retrieval_relevance_with_judge(
                    judge,
                    question=question,
                    retrieved_context=retrieved_context,
                    retrieved_chunks=retrieved_chunks,
                    expected_supporting_information=expected_supporting,
                    expected_key_facts=expected_facts,
                    expected_abstention=expected_abstention,
                )
                evidence_coverage = _clamp01(_safe_float(judged.get("evidence_coverage"), evidence_coverage))
                judge_score = _clamp01(_safe_float(judged.get("score"), evidence_coverage))
                judge_info = judged.get("explanation") or "judge-scored"
                relevance_core = judge_score
            except Exception as je:
                judge_info = f"judge_failed_fallback_deterministic: {type(je).__name__}"
                relevance_core = evidence_coverage
        else:
            relevance_core = evidence_coverage

        # Aggregate: emphasize evidence coverage; include P/R when annotations exist.
        # Missing-info with irrelevant retrieved chunks => low precision/relevance even
        # if final answer abstained correctly (abstention not considered here).
        if pr is not None:
            score = 0.5 * relevance_core + 0.25 * pr["retrieval_precision"] + 0.25 * pr["retrieval_recall"]
        else:
            score = relevance_core

        score = _clamp01(score)
        explanation_parts = [
            f"evidence_coverage={evidence_coverage:.3f}",
            f"score={score:.3f}",
        ]
        if pr is not None:
            explanation_parts.append(
                f"precision@k={pr['retrieval_precision']:.3f}, recall@k={pr['retrieval_recall']:.3f}, f1@k={pr['retrieval_f1']:.3f}"
            )
        else:
            explanation_parts.append("precision/recall unavailable (no exact ground-truth chunk annotations)")
        if judge_info:
            explanation_parts.append(str(judge_info))

        comment = "; ".join(explanation_parts)
        return _eval_result(
            "retrieval_relevance",
            score,
            comment=comment,
            passed=score >= threshold,
            diagnostics={
                "evidence_coverage": _clamp01(evidence_coverage),
                "retrieval_precision": pr["retrieval_precision"] if pr else None,
                "retrieval_recall": pr["retrieval_recall"] if pr else None,
                "retrieval_f1": pr["retrieval_f1"] if pr else None,
                "relevant_retrieved_count": pr["relevant_retrieved_count"] if pr else None,
                "retrieved_count": pr["retrieved_count"] if pr else len(retrieved_chunks),
                "ground_truth_relevant_count": pr["ground_truth_relevant_count"] if pr else None,
            },
        )
    except Exception as e:
        comment = f"evaluator_error: {type(e).__name__}"
        return _eval_result(
            "retrieval_relevance",
            0.0,
            comment=comment,
            passed=False,
            diagnostics={
                "evidence_coverage": 0.0,
                "retrieval_precision": None,
                "retrieval_recall": None,
                "retrieval_f1": None,
                "relevant_retrieved_count": None,
                "retrieved_count": 0,
                "ground_truth_relevant_count": None,
            },
        )


# ----------------------------
# Faithfulness / groundedness evaluator
# ----------------------------

FAITHFULNESS_SYSTEM = (
    "You are a strict faithfulness evaluator for a RAG system. "
    "Determine whether every factual claim in the final answer is supported by "
    "the retrieved context ONLY. Do NOT use the reference answer as evidence. "
    "Identify unsupported claims and fabricated source citations. "
    "Return ONLY valid JSON."
)


def _split_claims(answer: str) -> list[str]:
    if not answer:
        return []
    # simple deterministic claim splitter
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    claims = [p.strip(" -•\t") for p in parts if p and p.strip(" -•\t")]
    return claims[:30]


def _deterministic_faithfulness(
    answer: str,
    retrieved_context: str,
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[float, list[str], list[str], str]:
    claims = _split_claims(answer)
    if not claims:
        return 1.0, [], [], "empty answer treated as non-claiming"

    context_text = "\n".join([retrieved_context or ""] + [str(c.get("text") or "") for c in (retrieved_chunks or [])])
    norm_ctx = _normalize_text(context_text)
    ctx_nums = _number_date_tokens(context_text)
    max_source = len(retrieved_chunks or [])

    unsupported: list[str] = []
    for claim in claims:
        # Abstention claims are supported if no evidence required in answer
        if _is_abstention_answer(claim):
            continue
        ctoks = [t for t in _tokenize(claim) if len(t) > 2]
        if not ctoks:
            continue
        overlap = sum(1 for t in ctoks if t in norm_ctx)
        ratio = overlap / max(1, len(ctoks))
        claim_nums = _number_date_tokens(claim)
        unsupported_numbers = [n for n in claim_nums if (n not in norm_ctx and n not in ctx_nums)]
        key_content = _content_tokens(claim)
        missing_key = [t for t in key_content if t not in norm_ctx]
        # Heuristics: claim unsupported if low overlap, or contains numbers/dates
        # absent from context, or introduces key factual terms not grounded.
        if unsupported_numbers and not _contains_phrase(context_text, claim):
            unsupported.append(claim)
            continue
        if ratio < 0.45 and not _contains_phrase(context_text, claim):
            unsupported.append(claim)
            continue
        if key_content and (len(missing_key) / max(1, len(key_content))) >= 0.5 and ratio < 0.7 and not _contains_phrase(context_text, claim):
            unsupported.append(claim)

    citations = _extract_citations(answer)
    fabricated_citations = [f"[Source {n}]" for n in citations if n < 1 or n > max_source]

    # If answer is a correct-style abstention but includes citations when no context, flag
    if _is_abstention_answer(answer) and not context_text.strip() and citations:
        fabricated_citations.extend([f"[Source {n}]" for n in citations if f"[Source {n}]" not in fabricated_citations])

    non_abstain_claims = [c for c in claims if not _is_abstention_answer(c)]
    total = max(1, len(non_abstain_claims) or (1 if claims else 1))
    supported_count = max(0, total - len(unsupported))
    score = supported_count / total
    if fabricated_citations:
        score = min(score, 0.0 if len(fabricated_citations) >= 1 and not context_text.strip() else score * 0.5)
    explanation = f"supported_claims={supported_count}/{total}, unsupported={len(unsupported)}, fabricated_citations={len(fabricated_citations)}"
    return _clamp01(score), unsupported, fabricated_citations, explanation


def evaluate_faithfulness(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
    *,
    judge: JudgeFn | None = None,
    threshold: float = 0.8,
) -> EvaluationResult:
    try:
        if not isinstance(inputs, dict) or not isinstance(outputs, dict) or not isinstance(reference_outputs, dict):
            comment = "evaluator_error: invalid_input_types"
            return _eval_result(
                "faithfulness_groundedness",
                0.0,
                comment=comment,
                passed=False,
                diagnostics={"unsupported_claims": [], "fabricated_citations": []},
            )
        answer = str((outputs or {}).get("final_answer") or "")
        retrieved_context = str((outputs or {}).get("retrieved_context") or "")
        retrieved_chunks = list((outputs or {}).get("retrieved_chunks") or [])
        expected_abstention = (reference_outputs or {}).get("expected_abstention") or {}

        if judge is not None:
            try:
                user = (
                    "Assess faithfulness/groundedness of the final answer using ONLY the retrieved context. "
                    "Do NOT use any reference answer as evidence. Identify unsupported claims and fabricated source citations.\n\n"
                    f"Expected abstention (for behavior context, not evidence): {expected_abstention}\n"
                    f"Retrieved context: {retrieved_context}\n"
                    f"Retrieved chunks: {retrieved_chunks}\n"
                    f"Final answer: {answer}\n\n"
                    "Return JSON with keys: score (0-1 float), unsupported_claims (list of strings), "
                    "fabricated_citations (list of strings), explanation (string)."
                )
                judged = judge(FAITHFULNESS_SYSTEM, user)
                score = _clamp01(_safe_float(judged.get("score"), 0.0))
                unsupported_claims = [str(x) for x in (judged.get("unsupported_claims") or []) if str(x).strip()]
                fabricated_citations = [str(x) for x in (judged.get("fabricated_citations") or []) if str(x).strip()]
                explanation = str(judged.get("explanation") or "judge-scored")
                # harden fabricated citation detection deterministically too
                det_score, det_unsup, det_fab, _ = _deterministic_faithfulness(answer, retrieved_context, retrieved_chunks)
                for c in det_fab:
                    if c not in fabricated_citations:
                        fabricated_citations.append(c)
                if det_fab:
                    score = min(score, 0.5)
                if det_unsup and not unsupported_claims:
                    unsupported_claims = det_unsup
                return _eval_result(
                    "faithfulness_groundedness",
                    score,
                    comment=explanation,
                    passed=bool(score >= threshold and len(fabricated_citations) == 0),
                    diagnostics={
                        "unsupported_claims": unsupported_claims,
                        "fabricated_citations": fabricated_citations,
                    },
                )
            except Exception:
                # fall through to deterministic on judge failure
                pass

        score, unsupported_claims, fabricated_citations, explanation = _deterministic_faithfulness(
            answer, retrieved_context, retrieved_chunks
        )
        return _eval_result(
            "faithfulness_groundedness",
            score,
            comment=explanation,
            passed=bool(score >= threshold and len(fabricated_citations) == 0),
            diagnostics={
                "unsupported_claims": unsupported_claims,
                "fabricated_citations": fabricated_citations,
            },
        )
    except Exception as e:
        comment = f"evaluator_error: {type(e).__name__}"
        return _eval_result(
            "faithfulness_groundedness",
            0.0,
            comment=comment,
            passed=False,
            diagnostics={"unsupported_claims": [], "fabricated_citations": []},
        )


# ----------------------------
# Answer correctness evaluator
# ----------------------------

CORRECTNESS_SYSTEM = (
    "You are a strict answer correctness evaluator for a RAG system. "
    "Compare the actual answer against the reference answer and expected key facts. "
    "If abstention is expected, evaluate abstention behavior separately and do not "
    "reward fabricated facts. Do not require exact string matching. "
    "Return ONLY valid JSON."
)


def _deterministic_correctness(
    actual: str,
    reference: str,
    expected_key_facts: list[Any],
    expected_abstention: dict[str, Any],
) -> tuple[float, str, bool, bool | None, dict[str, Any]]:
    should_abstain = bool((expected_abstention or {}).get("should_abstain"))
    acceptable = list((expected_abstention or {}).get("acceptable_responses") or [])
    diagnostics: dict[str, Any] = {
        "key_fact_coverage": None,
        "covered_key_facts": None,
        "total_key_facts": None,
    }

    if should_abstain:
        abstained = _is_abstention_answer(actual, acceptable)
        # penalize if answer includes clear fabricated factual assertions beyond abstention
        has_citations = bool(_extract_citations(actual))
        extra_facty = len(_tokenize(actual)) > 20 and not abstained
        score = 1.0 if (abstained and not extra_facty) else (0.5 if abstained else 0.0)
        if has_citations and abstained:
            # citing sources in a no-evidence abstention is suspicious
            score = min(score, 0.5)
        diagnostics.update({"key_fact_coverage": None, "covered_key_facts": 0, "total_key_facts": 0})
        return _clamp01(score), "abstention evaluated separately", should_abstain, abstained, diagnostics

    # Non-abstention: key-fact coverage vs reference/facts (not exact match)
    facts = [f for f in expected_key_facts if isinstance(f, str) and f.strip()]
    if not facts:
        # fallback compare against reference
        facts = [reference] if isinstance(reference, str) and reference.strip() else []
    if not facts:
        sim = 1.0 if _normalize_text(actual) == _normalize_text(reference) else 0.0
        diagnostics.update({"key_fact_coverage": sim, "covered_key_facts": 0, "total_key_facts": 0})
        return sim, "no key facts provided; fallback comparison", should_abstain, None, diagnostics

    covered = sum(1 for f in facts if _contains_phrase(actual, f))
    coverage = covered / len(facts)
    # mild semantic leniency via token overlap with reference
    ref_overlap = 0.0
    if reference:
        rt = _tokenize(reference)
        at = _tokenize(actual)
        if rt:
            ref_overlap = len(rt & at) / len(rt)
    score = 0.8 * coverage + 0.2 * ref_overlap
    # if model abstained incorrectly, fail
    if _is_abstention_answer(actual):
        score = min(score, 0.0)
    diagnostics.update(
        {
            "key_fact_coverage": _clamp01(coverage),
            "covered_key_facts": covered,
            "total_key_facts": len(facts),
        }
    )
    return _clamp01(score), f"key_fact_coverage={covered}/{len(facts)}", should_abstain, False, diagnostics


def evaluate_answer_correctness(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
    *,
    judge: JudgeFn | None = None,
    threshold: float = 0.7,
) -> EvaluationResult:
    try:
        if not isinstance(inputs, dict) or not isinstance(outputs, dict) or not isinstance(reference_outputs, dict):
            comment = "evaluator_error: invalid_input_types"
            return _eval_result(
                "answer_correctness",
                0.0,
                comment=comment,
                passed=False,
                diagnostics={"abstention_expected": None, "abstention_correct": None},
            )
        question = str((inputs or {}).get("question") or "")
        actual = str((outputs or {}).get("final_answer") or "")
        reference = str((reference_outputs or {}).get("reference_answer") or "")
        expected_key_facts = list((reference_outputs or {}).get("expected_key_facts") or [])
        expected_abstention = (reference_outputs or {}).get("expected_abstention") or {"should_abstain": False}

        if judge is not None:
            try:
                user = (
                    "Evaluate answer correctness. Compare the actual answer against the reference answer and expected key facts. "
                    "Handle abstention separately when expected. Do not use exact string matching as the only criterion.\n\n"
                    f"Question: {question}\n"
                    f"Reference answer: {reference}\n"
                    f"Expected key facts: {expected_key_facts}\n"
                    f"Expected abstention: {expected_abstention}\n"
                    f"Actual answer: {actual}\n\n"
                    "Return JSON with keys: score (0-1 float), explanation (string), "
                    "abstention_expected (bool), abstention_correct (bool or null)."
                )
                judged = judge(CORRECTNESS_SYSTEM, user)
                score = _clamp01(_safe_float(judged.get("score"), 0.0))
                abstention_expected = bool(judged.get("abstention_expected", bool(expected_abstention.get("should_abstain"))))
                abstention_correct = judged.get("abstention_correct")
                if abstention_correct is not None:
                    abstention_correct = bool(abstention_correct)
                explanation = str(judged.get("explanation") or "judge-scored")
                # sanity: incorrect abstention should not pass
                if abstention_expected is False and _is_abstention_answer(actual):
                    score = min(score, 0.0)
                    explanation += "; overrode: incorrect abstention"
                # Preserve deterministic key-fact coverage diagnostics alongside judge score
                _, _, _, _, det_diag = _deterministic_correctness(actual, reference, expected_key_facts, expected_abstention)
                return _eval_result(
                    "answer_correctness",
                    score,
                    comment=explanation,
                    passed=score >= threshold,
                    diagnostics={
                        "abstention_expected": abstention_expected,
                        "abstention_correct": abstention_correct,
                        **det_diag,
                    },
                )
            except Exception:
                pass

        score, explanation, abstention_expected, abstention_correct, det_diag = _deterministic_correctness(
            actual, reference, expected_key_facts, expected_abstention
        )
        return _eval_result(
            "answer_correctness",
            score,
            comment=explanation,
            passed=score >= threshold,
            diagnostics={
                "abstention_expected": abstention_expected,
                "abstention_correct": abstention_correct,
                **det_diag,
            },
        )
    except Exception as e:
        comment = f"evaluator_error: {type(e).__name__}"
        return _eval_result(
            "answer_correctness",
            0.0,
            comment=comment,
            passed=False,
            diagnostics={
                "abstention_expected": None,
                "abstention_correct": None,
                "key_fact_coverage": None,
                "covered_key_facts": None,
                "total_key_facts": None,
            },
        )
