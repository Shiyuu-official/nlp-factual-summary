"""GovReport dataset loader."""

import logging
import os
from typing import List, Dict, Optional

from datasets import load_dataset
import nltk

logger = logging.getLogger(__name__)


class GovReportDataLoader:
    """Loads ccdv/govreport-summarization with optional sample limit and shuffle."""

    def __init__(self, dataset_name: str = "ccdv/govreport-summarization",
                 cache_dir: Optional[str] = None):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        # Ensure nltk sentence tokenizer is available
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    def load(self, split: str = "validation", max_samples: Optional[int] = None,
             seed: int = 42) -> List[Dict]:
        """Load dataset, shuffle deterministically, take max_samples.

        Returns list of dicts: {sample_id, report, reference_summary}
        """
        logger.info(f"Loading {self.dataset_name} ({split} split)...")

        # Environment variable wins on shared GPU servers; config value is fallback.
        cache_dir = os.environ.get("DATASETS_CACHE", self.cache_dir)

        dataset = load_dataset(
            self.dataset_name,
            split=split,
            cache_dir=cache_dir,
        )

        # Deterministic shuffle so test mode sees the same samples every run
        dataset = dataset.shuffle(seed=seed)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        samples = []
        for idx, item in enumerate(dataset):
            samples.append({
                "sample_id": str(idx),
                "report": item["report"],
                "reference_summary": item["summary"],
            })

        logger.info(f"Loaded {len(samples)} samples")
        return samples

    @staticmethod
    def compute_statistics(samples: List[Dict]) -> Dict:
        """Compute dataset statistics."""
        report_lens = [len(s["report"].split()) for s in samples]
        summary_lens = [len(s["reference_summary"].split()) for s in samples]

        return {
            "num_samples": len(samples),
            "avg_report_words": sum(report_lens) / len(report_lens) if report_lens else 0,
            "max_report_words": max(report_lens) if report_lens else 0,
            "min_report_words": min(report_lens) if report_lens else 0,
            "avg_reference_summary_words": sum(summary_lens) / len(summary_lens) if summary_lens else 0,
            "max_reference_summary_words": max(summary_lens) if summary_lens else 0,
            "min_reference_summary_words": min(summary_lens) if summary_lens else 0,
        }
