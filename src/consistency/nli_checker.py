"""Factual consistency checker using NLI (Natural Language Inference).

Checks each summary sentence against the source document:
  1. Split summary into sentences
  2. Retrieve evidence from source for each sentence
  3. Run NLI (entailment/neutral/contradiction) for each sentence-evidence pair
  4. A sentence is "consistent" if max entailment score across evidences >= threshold
"""

import logging
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from .sentence_splitter import SentenceSplitter
from .evidence_retrieval import BaseEvidenceRetriever, Evidence

logger = logging.getLogger(__name__)

NLI_LABELS = ["entailment", "neutral", "contradiction"]


class NLIChecker:
    """Factual consistency checker using an NLI model."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli",
                 entailment_threshold: float = 0.5,
                 device: str = "cpu"):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.device = device

        logger.info(f"Loading NLI model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()
        logger.info("NLI model loaded.")

    def _check_entailment(self, premise: str, hypothesis: str) -> Dict:
        """Run NLI on a single (premise, hypothesis) pair.

        Returns dict with keys: label, entailment_score, scores
        """
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        scores = {label: probs[i].item() for i, label in enumerate(NLI_LABELS)}
        predicted = NLI_LABELS[torch.argmax(probs).item()]

        return {
            "label": predicted,
            "entailment_score": scores["entailment"],
            "scores": scores,
        }

    def _check_sentence(self, sentence: str,
                        retriever: BaseEvidenceRetriever,
                        document_sentences: List[str]) -> Dict:
        """Check factual consistency of one summary sentence.

        Returns dict with: index, text, is_consistent, max_entailment_score,
        evidences, nli_per_evidence
        """
        evidences = retriever.retrieve(
            sentence, document_sentences, top_k=3,
        )

        if not evidences:
            return {
                "text": sentence,
                "is_consistent": False,
                "max_entailment_score": 0.0,
                "evidences": [],
                "nli_per_evidence": [],
                "failure_reason": "no_evidence_found",
            }

        nli_results = []
        max_entailment = 0.0

        for ev in evidences:
            nli = self._check_entailment(ev.text, sentence)
            nli["evidence_score"] = ev.score
            nli_results.append(nli)
            max_entailment = max(max_entailment, nli["entailment_score"])

        is_consistent = max_entailment >= self.entailment_threshold

        return {
            "text": sentence,
            "is_consistent": is_consistent,
            "max_entailment_score": round(max_entailment, 4),
            "evidences": [{"text": ev.text, "score": round(ev.score, 4),
                           "sentence_index": ev.sentence_index} for ev in evidences],
            "nli_per_evidence": nli_results,
        }

    def check_summary(self, summary: str, report: str,
                      splitter: SentenceSplitter,
                      retriever: BaseEvidenceRetriever) -> Dict:
        """Check consistency of an entire summary.

        Returns dict with: sentences (list of per-sentence results),
        n_total, n_consistent, n_inconsistent, consistency_rate
        """
        summary_sentences = splitter.split(summary)
        document_sentences = splitter.split(report)

        if not summary_sentences:
            return {
                "sentences": [],
                "n_total": 0,
                "n_consistent": 0,
                "n_inconsistent": 0,
                "consistency_rate": 0.0,
            }

        sentence_results = []
        for i, sent in enumerate(summary_sentences):
            result = self._check_sentence(sent, retriever, document_sentences)
            result["index"] = i
            sentence_results.append(result)

        n_total = len(sentence_results)
        n_consistent = sum(1 for r in sentence_results if r["is_consistent"])
        n_inconsistent = n_total - n_consistent
        consistency_rate = n_consistent / n_total if n_total > 0 else 0.0

        return {
            "sentences": sentence_results,
            "n_total": n_total,
            "n_consistent": n_consistent,
            "n_inconsistent": n_inconsistent,
            "consistency_rate": round(consistency_rate, 4),
        }

    def check_batch(self, samples: List[Dict],
                    splitter: SentenceSplitter,
                    retriever: BaseEvidenceRetriever) -> List[Dict]:
        """Batch consistency check.

        Each sample must have: sample_id, report, generated_summary.
        Adds 'consistency' dict to each sample.
        """
        results = []
        for item in tqdm(samples, desc="NLI consistency check", unit="sample"):
            record = dict(item)
            try:
                record["consistency"] = self.check_summary(
                    record["generated_summary"],
                    record["report"],
                    splitter,
                    retriever,
                )
            except Exception as e:
                logger.error(f"Sample {record.get('sample_id')}: consistency check failed: {e}")
                record["consistency"] = {
                    "sentences": [],
                    "n_total": 0,
                    "n_consistent": 0,
                    "n_inconsistent": 0,
                    "consistency_rate": 0.0,
                    "error": str(e),
                }
            results.append(record)

        return results
