"""Factual consistency checker using NLI (Natural Language Inference).

Checks each summary sentence against the source document:
  1. Split summary into sentences
  2. Retrieve evidence from source for each sentence
  3. Run NLI (entailment/neutral/contradiction) for each sentence-evidence pair
  4. A sentence is "consistent" if max entailment score across evidences >= threshold
"""

import logging
import re
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from .sentence_splitter import SentenceSplitter
from .evidence_retrieval import BaseEvidenceRetriever

logger = logging.getLogger(__name__)


class NLIChecker:
    """Factual consistency checker using an NLI model."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli",
                 entailment_threshold: float = 0.5,
                 evidence_top_k: int = 3,
                 device: str = "cpu"):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.evidence_top_k = evidence_top_k
        self.device = device

        logger.info(f"Loading NLI model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()
        self.id2label = {
            int(idx): label.lower()
            for idx, label in self.model.config.id2label.items()
        }
        self.entailment_label_id = self._find_label_id("entailment")
        logger.info("NLI model loaded.")

    def _find_label_id(self, target: str) -> int:
        """Find a label id from model metadata instead of assuming fixed order."""
        for idx, label in self.id2label.items():
            if target in label:
                return idx
        raise ValueError(
            f"Cannot find '{target}' label in {self.model_name} labels: {self.id2label}"
        )

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

        scores = {
            self.id2label.get(i, f"label_{i}"): probs[i].item()
            for i in range(len(probs))
        }
        predicted_id = torch.argmax(probs).item()
        predicted = self.id2label.get(predicted_id, f"label_{predicted_id}")

        return {
            "label": predicted,
            "entailment_score": probs[self.entailment_label_id].item(),
            "scores": scores,
        }

    def _check_entailment_batch(self, premises: List[str],
                                hypotheses: List[str]) -> List[Dict]:
        """Run NLI for multiple premise/hypothesis pairs in one model call."""
        if not premises:
            return []

        inputs = self.tokenizer(
            premises, hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for row in probs:
            scores = {
                self.id2label.get(i, f"label_{i}"): row[i].item()
                for i in range(len(row))
            }
            predicted_id = torch.argmax(row).item()
            predicted = self.id2label.get(predicted_id, f"label_{predicted_id}")
            results.append({
                "label": predicted,
                "entailment_score": row[self.entailment_label_id].item(),
                "scores": scores,
            })

        return results

    def check_pair(self, evidence: str, sentence: str) -> Dict:
        """Check whether one evidence passage supports one sentence."""
        result = self._check_entailment(evidence, sentence)
        score = result["entailment_score"]
        result["is_consistent"] = score >= self.entailment_threshold
        result["entailment_score"] = round(score, 4)
        return result

    def _is_checkable_sentence(self, sentence: str) -> bool:
        """Return False for headings, fragments, and instruction artifacts."""
        text = sentence.strip()
        if not text:
            return False
        if re.fullmatch(r"[\W_]+|\d+[.)]?", text):
            return False

        lowered = text.lower()
        blocked_prefixes = (
            "summary:",
            "key point",
            "key points",
            "note:",
            "explanation:",
            "corrected sentence:",
        )
        if lowered.startswith(blocked_prefixes):
            return False

        alpha_tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
        if len(alpha_tokens) < 4:
            return False

        return True

    def _check_sentence(self, sentence: str,
                        retriever: BaseEvidenceRetriever,
                        document_sentences: List[str]) -> Dict:
        """Check factual consistency of one summary sentence.

        Returns dict with: index, text, is_consistent, max_entailment_score,
        evidences, nli_per_evidence
        """
        evidences = retriever.retrieve(
            sentence, document_sentences, top_k=self.evidence_top_k,
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

        premises = [ev.text for ev in evidences]
        hypotheses = [sentence] * len(evidences)
        nli_results = self._check_entailment_batch(premises, hypotheses)
        max_entailment = 0.0

        for ev, nli in zip(evidences, nli_results):
            nli["evidence_score"] = ev.score
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
        raw_summary_sentences = splitter.split(summary)
        summary_sentences = [
            sent for sent in raw_summary_sentences
            if self._is_checkable_sentence(sent)
        ]
        n_skipped = len(raw_summary_sentences) - len(summary_sentences)
        document_sentences = splitter.split(report)

        if not summary_sentences:
            return {
                "sentences": [],
                "n_total": 0,
                "n_consistent": 0,
                "n_inconsistent": 0,
                "n_skipped": n_skipped,
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
            "n_skipped": n_skipped,
            "consistency_rate": round(consistency_rate, 4),
        }

    def check_batch(self, samples: List[Dict],
                    splitter: SentenceSplitter,
                    retriever: BaseEvidenceRetriever,
                    checkpoint_path: Optional[str] = None,
                    existing_results: Optional[List[Dict]] = None) -> List[Dict]:
        """Batch consistency check.

        Each sample must have: sample_id, report, generated_summary.
        Adds 'consistency' dict to each sample.
        """
        results = list(existing_results or [])
        done_ids = {r.get("sample_id") for r in results}
        for item in tqdm(samples, desc="NLI consistency check", unit="sample"):
            if item.get("sample_id") in done_ids:
                continue

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
            done_ids.add(record.get("sample_id"))

            if checkpoint_path:
                from ..utils.io import save_json
                save_json(results, checkpoint_path)

        return results
