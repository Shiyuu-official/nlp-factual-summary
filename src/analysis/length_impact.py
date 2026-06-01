"""Analysis Task 1: Summary length vs. factual error rate."""

import json
import logging
import os
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def analyze_length_impact(samples: List[Dict],
                          length_bins: List[List[int]],
                          rouge_evaluator,
                          splitter=None) -> Dict:
    """Analyze how summary length affects factual consistency and ROUGE.

    Args:
        samples: Full pipeline results with consistency and correction fields.
        length_bins: List of [low, high] word-count bins.
        rouge_evaluator: RougeEvaluator instance.
        splitter: SentenceSplitter instance (optional).

    Returns dict keyed by bin label with per-bin statistics.
    """
    # Pre-process: compute word count and ROUGE for each sample
    for s in samples:
        s["_summary_len"] = len(s.get("generated_summary", "").split())
        if s.get("generated_summary") and s.get("reference_summary"):
            s["_rouge"] = rouge_evaluator.compute_single(
                s["reference_summary"], s["generated_summary"],
            )

    results = {}
    for low, high in length_bins:
        if high >= 99999:
            high_display = float("inf")
            label = f"{low}+"
        else:
            high_display = high
            label = f"{low}-{high_display}"

        # Filter samples in this bin
        bucket = [s for s in samples
                  if low <= s.get("_summary_len", 0) < high]

        if not bucket:
            results[label] = {"num_samples": 0}
            continue

        n_samples = len(bucket)
        avg_len = np.mean([s["_summary_len"] for s in bucket])

        # Consistency stats
        rates = [s.get("consistency", {}).get("consistency_rate", 0) for s in bucket]
        avg_consistency = np.mean(rates) if rates else 0.0
        avg_errors = [1 - r for r in rates]
        avg_error_rate = np.mean(avg_errors) if avg_errors else 0.0

        # ROUGE stats
        rouge1_vals = [s.get("_rouge", {}).get("rouge1_f", 0) for s in bucket]
        rouge2_vals = [s.get("_rouge", {}).get("rouge2_f", 0) for s in bucket]
        rougeL_vals = [s.get("_rouge", {}).get("rougeL_f", 0) for s in bucket]

        results[label] = {
            "num_samples": n_samples,
            "avg_summary_words": round(avg_len, 1),
            "avg_consistency_rate": round(avg_consistency, 4),
            "avg_error_rate": round(avg_error_rate, 4),
            "avg_rouge1_f": round(np.mean(rouge1_vals), 4) if rouge1_vals else 0.0,
            "avg_rouge2_f": round(np.mean(rouge2_vals), 4) if rouge2_vals else 0.0,
            "avg_rougeL_f": round(np.mean(rougeL_vals), 4) if rougeL_vals else 0.0,
        }

    return results


def plot_length_impact(analysis: Dict, output_path: str) -> None:
    """Generate two-panel matplotlib figure: error rate + ROUGE vs length bin.

    Saves to output_path as PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    bins = [k for k in analysis.keys() if analysis[k].get("num_samples", 0) > 0]
    if not bins:
        logger.warning("No data to plot")
        return

    error_rates = [analysis[b]["avg_error_rate"] * 100 for b in bins]
    rouge1 = [analysis[b]["avg_rouge1_f"] * 100 for b in bins]
    rouge2 = [analysis[b]["avg_rouge2_f"] * 100 for b in bins]
    rougeL = [analysis[b]["avg_rougeL_f"] * 100 for b in bins]

    x = range(len(bins))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: error rate
    bars = ax1.bar(x, error_rates, color="salmon", alpha=0.8, edgecolor="black")
    ax1.set_xlabel("Summary Length (words)", fontsize=11)
    ax1.set_ylabel("Factual Error Rate (%)", fontsize=11)
    ax1.set_title("Error Rate by Summary Length", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bins, rotation=45)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, error_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    # Right: ROUGE scores
    width = 0.25
    ax2.bar([i - width for i in x], rouge1, width, label="ROUGE-1", color="steelblue", alpha=0.8)
    ax2.bar(x, rouge2, width, label="ROUGE-2", color="darkorange", alpha=0.8)
    ax2.bar([i + width for i in x], rougeL, width, label="ROUGE-L", color="seagreen", alpha=0.8)
    ax2.set_xlabel("Summary Length (words)", fontsize=11)
    ax2.set_ylabel("ROUGE F1 (%)", fontsize=11)
    ax2.set_title("ROUGE by Summary Length", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bins, rotation=45)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Length impact plot saved to {output_path}")
