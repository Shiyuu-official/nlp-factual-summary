"""Evaluation metrics: consistency stats, correction stats, before/after comparison."""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def compute_consistency_stats(samples: List[Dict]) -> Dict:
    """Aggregate consistency statistics across all samples.

    Each sample must have a 'consistency' dict.
    """
    n_samples = 0
    total_sentences = 0
    total_consistent = 0
    total_inconsistent = 0

    for s in samples:
        c = s.get("consistency", {})
        if not c:
            continue
        n_samples += 1
        total_sentences += c.get("n_total", 0)
        total_consistent += c.get("n_consistent", 0)
        total_inconsistent += c.get("n_inconsistent", 0)

    rate = total_consistent / total_sentences if total_sentences > 0 else 0.0

    return {
        "n_samples": n_samples,
        "total_sentences": total_sentences,
        "total_consistent": total_consistent,
        "total_inconsistent": total_inconsistent,
        "overall_consistency_rate": round(rate, 4),
        "overall_error_rate": round(1 - rate, 4),
    }


def compute_correction_stats(samples: List[Dict]) -> Dict:
    """Aggregate correction statistics."""
    total_attempted = 0
    total_succeeded = 0
    failure_reasons = {}

    for s in samples:
        corr = s.get("correction", {})
        total_attempted += corr.get("n_attempted", 0)
        total_succeeded += corr.get("n_succeeded", 0)
        for c in corr.get("corrections", []):
            reason = c.get("failure_reason")
            if reason and not c.get("success", False):
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return {
        "total_attempted": total_attempted,
        "total_succeeded": total_succeeded,
        "success_rate": round(total_succeeded / total_attempted, 4) if total_attempted > 0 else 0.0,
        "failure_reasons": failure_reasons,
    }


def compare_before_after(original_samples: List[Dict],
                         rouge_evaluator) -> Dict:
    """Compare ROUGE before and after correction.

    Joins original and corrected summaries by sample_id.
    """
    # Original ROUGE: generated_summary vs reference_summary
    orig_pairs = []
    for s in original_samples:
        if s.get("generated_summary") and s.get("reference_summary"):
            orig_pairs.append((s["sample_id"], s["reference_summary"], s["generated_summary"]))

    # Corrected ROUGE: corrected_summary vs reference_summary
    corr_pairs = []
    for s in original_samples:
        corr = s.get("correction", {})
        if corr.get("corrected_summary") and s.get("reference_summary"):
            corr_pairs.append((s["sample_id"], s["reference_summary"], corr["corrected_summary"]))

    return {
        "original_rouge": rouge_evaluator.compute_batch(orig_pairs) if orig_pairs else None,
        "corrected_rouge": rouge_evaluator.compute_batch(corr_pairs) if corr_pairs else None,
        "consistency": compute_consistency_stats(original_samples),
        "correction": compute_correction_stats(original_samples),
    }
