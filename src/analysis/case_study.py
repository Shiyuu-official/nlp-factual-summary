"""Analysis Task 2: Correction case study (10+ cases, success + failure)."""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def collect_correction_cases(samples: List[Dict],
                              num_cases: int = 10) -> Dict:
    """Extract correction cases from samples.

    Returns dict with keys: successful, failed, summary.
    Each case: {sample_id, sentence_index, original, corrected, evidence, success, analysis}
    """
    all_cases = []

    for s in samples:
        corr = s.get("correction", {})
        consistency = s.get("consistency", {})
        sentences = consistency.get("sentences", [])

        for c in corr.get("corrections", []):
            idx = c.get("sentence_index", -1)
            # Get the NLI results for this sentence
            sent_info = sentences[idx] if 0 <= idx < len(sentences) else {}
            nli_results = sent_info.get("nli_per_evidence", [])
            verification = c.get("verification", {})

            all_cases.append({
                "sample_id": s.get("sample_id"),
                "sentence_index": idx,
                "original": c.get("original", ""),
                "corrected": c.get("corrected", ""),
                "evidence": c.get("evidence_used", "")[:300],
                "success": c.get("success", False),
                "failure_reason": c.get("failure_reason"),
                "original_nli_label": nli_results[0].get("label", "") if nli_results else "",
                "original_entailment_score": (
                    nli_results[0].get("entailment_score", 0) if nli_results else 0
                ),
                "verified": verification.get("verified", False),
                "improved": verification.get("improved", False),
                "fixed": verification.get("fixed", False),
                "verified_original_entailment_score": verification.get(
                    "original_entailment_score", 0
                ),
                "verified_corrected_entailment_score": verification.get(
                    "corrected_entailment_score", 0
                ),
            })

    successful = [c for c in all_cases if c["success"]]
    failed = [c for c in all_cases if not c["success"]]

    selected = {
        "successful": successful[:num_cases // 2],
        "failed": failed[:num_cases // 2],
        "summary": {
            "total_corrections": len(all_cases),
            "total_successful": len(successful),
            "total_failed": len(failed),
            "success_rate": round(
                len(successful) / len(all_cases), 4
            ) if all_cases else 0.0,
        },
    }

    return selected


def print_case_report(cases: Dict) -> None:
    """Print a formatted case report to console."""
    print("\n" + "=" * 80)
    print("  CORRECTION CASE STUDY REPORT")
    print("=" * 80)
    print(f"Total corrections: {cases['summary']['total_corrections']}")
    print(f"Successful: {cases['summary']['total_successful']} "
          f"({cases['summary']['success_rate']:.1%})")
    print(f"Failed: {cases['summary']['total_failed']}")
    print()

    print("─" * 80)
    print("  SUCCESSFUL CORRECTIONS")
    print("─" * 80)
    for i, case in enumerate(cases["successful"][:5], 1):
        print(f"\nCase S-{i} (sample {case['sample_id']}, sentence {case['sentence_index']})")
        print(f"  Original entailment: {case['original_entailment_score']:.3f} ({case['original_nli_label']})")
        print(f"  Verified: {case['verified']}  improved: {case['improved']}  fixed: {case['fixed']}")
        print(f"  Before: {case['original'][:150]}")
        print(f"  After:  {case['corrected'][:150]}")
        print(f"  Evidence: {case['evidence'][:120]}...")

    print()
    print("─" * 80)
    print("  FAILED CORRECTIONS")
    print("─" * 80)
    for i, case in enumerate(cases["failed"][:5], 1):
        print(f"\nCase F-{i} (sample {case['sample_id']}, sentence {case['sentence_index']})")
        print(f"  Reason: {case['failure_reason']}")
        print(f"  Before: {case['original'][:150]}")
