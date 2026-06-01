"""Error corrector: locally rewrites inconsistent summary sentences using source evidence.

Key fix vs old code: STRICT prompt that forbids source copying, plus output validation
that rejects corrections which are too long or contain multiple sentences.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Prompt template with strict local-editing constraints
CORRECTION_PROMPT = """You are a precise fact-checking editor. Your task is to correct a SINGLE factual error in ONE sentence using evidence from the source document.

RULES (follow exactly):
1. ONLY change the specific words or phrases that are factually wrong.
2. Keep the original sentence structure, grammar, and style IDENTICAL.
3. Do NOT add information beyond what is needed to fix the error.
4. Do NOT copy or restate entire passages from the evidence.
5. Do NOT write a new sentence from scratch.
6. Your output must be ONE sentence and nothing else.

Source document evidence:
{evidence}

Original sentence (may contain a factual error):
{sentence}

Corrected sentence (ONE sentence only):"""


class LocalEditCorrector:
    """Corrects factually inconsistent summary sentences via local editing.

    Uses Qwen2.5-1.5B-Instruct with beam search and strict output validation.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 max_new_tokens: int = 100, temperature: float = 0.0,
                 num_beams: int = 3, max_length_ratio: float = 2.0,
                 device: str = "cpu"):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_beams = num_beams
        self.max_length_ratio = max_length_ratio
        self.device = device

        logger.info(f"Loading correction model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        logger.info("Correction model loaded.")

    def _validate_correction(self, original: str, corrected: str) -> Tuple[bool, str]:
        """Validate that the corrected sentence meets local-editing constraints.

        Returns (is_valid, failure_reason).
        """
        corrected = corrected.strip()
        if not corrected:
            return False, "empty_output"
        if corrected == original.strip():
            return False, "no_change"

        # Check length: corrected shouldn't be much longer than original
        orig_words = len(original.split())
        corr_words = len(corrected.split())
        if corr_words > orig_words * self.max_length_ratio:
            return False, f"output_too_long ({corr_words} vs {orig_words} words)"

        # Check for multi-sentence output (evidence pasting)
        sentence_seps = re.findall(r'[.!?]+', corrected)
        if len(sentence_seps) > 1 and corr_words > 50:
            return False, "appears_to_be_evidence_passage"

        return True, ""

    def correct_single(self, sentence: str, evidence_text: str) -> Dict:
        """Correct a single inconsistent sentence.

        Args:
            sentence: The inconsistent summary sentence.
            evidence_text: Evidence passage from source document.

        Returns dict with: original, corrected, success, evidence_used, failure_reason
        """
        prompt = CORRECTION_PROMPT.format(evidence=evidence_text, sentence=sentence)

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=2048, padding=True,
        ).to(self.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=5,
                    num_beams=self.num_beams,
                    early_stopping=True,
                    do_sample=(self.temperature > 0),
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the corrected sentence (after "Corrected sentence:")
            if "Corrected sentence" in generated:
                parts = generated.rsplit("Corrected sentence", 1)
                corrected = parts[-1].strip().lstrip(":").strip()
            else:
                # Fallback: strip the prompt part
                prompt_end = "Corrected sentence (ONE sentence only):"
                if prompt_end in generated:
                    corrected = generated.split(prompt_end)[-1].strip()
                else:
                    corrected = generated.strip()

            # Validate
            is_valid, reason = self._validate_correction(sentence, corrected)

            return {
                "original": sentence,
                "corrected": corrected if is_valid else sentence,
                "success": is_valid,
                "evidence_used": evidence_text[:500],
                "failure_reason": reason if not is_valid else None,
            }

        except Exception as e:
            logger.error(f"Correction failed: {e}")
            return {
                "original": sentence,
                "corrected": sentence,
                "success": False,
                "evidence_used": evidence_text[:500],
                "failure_reason": str(e),
            }

    def correct_summary(self, summary: str,
                        consistency: Dict,
                        report: str) -> Dict:
        """Correct all inconsistent sentences in a summary.

        Args:
            summary: The generated summary text.
            consistency: Consistency check result dict (from NLIChecker).
            report: Source document.

        Returns dict with: original_summary, corrected_summary, corrections,
        n_attempted, n_succeeded
        """
        sentences = consistency.get("sentences", [])
        if not sentences:
            return {
                "original_summary": summary,
                "corrected_summary": summary,
                "corrections": [],
                "n_attempted": 0,
                "n_succeeded": 0,
            }

        corrections = []
        # Build corrected summary by patching inconsistent sentences
        sentence_texts = [s["text"] for s in sentences]

        for sent in tqdm(sentences, desc="Correcting", leave=False, unit="sent"):
            if sent["is_consistent"]:
                continue

            # Get best evidence (highest entailment score)
            evidences = sent.get("evidences", [])
            if not evidences:
                corrections.append({
                    "sentence_index": sent["index"],
                    "original": sent["text"],
                    "corrected": sent["text"],
                    "success": False,
                    "evidence_used": "",
                    "failure_reason": "no_evidence",
                })
                continue

            best_ev = max(evidences, key=lambda e: e.get("score", 0))

            result = self.correct_single(sent["text"], best_ev["text"])
            result["sentence_index"] = sent["index"]
            corrections.append(result)

            # Patch the corrected sentence in place
            if result["success"]:
                idx = sent["index"]
                if idx < len(sentence_texts):
                    sentence_texts[idx] = result["corrected"]

        corrected_summary = " ".join(sentence_texts)
        n_attempted = len(corrections)
        n_succeeded = sum(1 for c in corrections if c.get("success", False))

        return {
            "original_summary": summary,
            "corrected_summary": corrected_summary,
            "corrections": corrections,
            "n_attempted": n_attempted,
            "n_succeeded": n_succeeded,
        }

    def correct_batch(self, samples: List[Dict]) -> List[Dict]:
        """Batch correction. Adds 'correction' dict to each sample."""
        results = []
        for item in tqdm(samples, desc="Correcting", unit="sample"):
            record = dict(item)
            try:
                record["correction"] = self.correct_summary(
                    record["generated_summary"],
                    record.get("consistency", {}),
                    record["report"],
                )
            except Exception as e:
                logger.error(f"Sample {record.get('sample_id')}: correction failed: {e}")
                record["correction"] = {
                    "original_summary": record.get("generated_summary", ""),
                    "corrected_summary": record.get("generated_summary", ""),
                    "corrections": [],
                    "n_attempted": 0,
                    "n_succeeded": 0,
                    "error": str(e),
                }
            results.append(record)

        return results
