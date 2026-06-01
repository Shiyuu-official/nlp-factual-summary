"""ROUGE evaluation for summarization."""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


class RougeEvaluator:
    """Computes ROUGE scores between reference and generated summaries."""

    def __init__(self, rouge_types: Optional[List[str]] = None,
                 use_stemmer: bool = True):
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=use_stemmer)

    def compute_single(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute ROUGE for one sample. Returns {rouge1_f, rouge2_f, rougeL_f}."""
        if not reference or not hypothesis:
            return {f"{t}_f": 0.0 for t in self.rouge_types}
        scores = self.scorer.score(reference, hypothesis)
        return {f"{t}_f": scores[t].fmeasure for t in self.rouge_types}

    def compute_batch(self, pairs: List[Tuple[str, str, str]]) -> Dict:
        """Compute ROUGE for a batch.

        Args:
            pairs: list of (sample_id, reference, hypothesis)

        Returns dict with keys: rouge1, rouge2, rougeL, each having
        mean, std, and per_sample list.
        """
        per_sample = []
        all_scores = {t: [] for t in self.rouge_types}

        for sample_id, ref, hyp in pairs:
            scores = self.compute_single(ref, hyp)
            entry = {"sample_id": sample_id}
            for t in self.rouge_types:
                key = f"{t}_f"
                entry[key] = scores[key]
                all_scores[t].append(scores[key])
            per_sample.append(entry)

        result = {}
        for t in self.rouge_types:
            vals = all_scores[t]
            result[t] = {
                "mean": round(np.mean(vals), 4) if vals else 0.0,
                "std": round(np.std(vals), 4) if vals else 0.0,
            }

        result["per_sample"] = per_sample
        return result
