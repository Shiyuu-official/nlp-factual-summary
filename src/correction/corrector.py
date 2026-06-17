"""Error corrector: locally rewrites inconsistent summary sentences using source evidence.

Key fix vs old code: STRICT prompt that forbids source copying, plus output validation
that rejects corrections which are too long or contain multiple sentences.

Advanced mode: generate multiple local-edit candidates, score them with NLI,
and only patch the summary when the best format-valid candidate is entailed by
the retrieved evidence.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Prompt template with strict local-editing constraints
CORRECTION_PROMPT = """Correct the summary sentence using the evidence.

Rules:
- Return ONLY the corrected sentence.
- Do not explain.
- Do not quote or copy the evidence.
- Keep the corrected sentence close to the original length.
- If only a phrase is wrong, change only that phrase.

Evidence:
{evidence}

Original sentence:
{sentence}

Corrected sentence:"""


class LocalEditCorrector:
    """Corrects factually inconsistent summary sentences via local editing.

    Uses Qwen2.5-1.5B-Instruct with beam search, strict output validation,
    and optional NLI reranking over multiple candidates.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 max_new_tokens: int = 100, temperature: float = 0.0,
                 num_beams: int = 3, num_candidates: int = 1,
                 max_length_ratio: float = 2.0,
                 device: str = "cpu"):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_beams = num_beams
        self.num_candidates = max(1, num_candidates)
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
        if re.fullmatch(r"[\W_]+|\d+[.)]?", original.strip()):
            return False, "invalid_original_sentence"
        if corrected == original.strip():
            return False, "no_change"
        if re.search(r"\b(human|assistant|user)\s*:|\bhuman resources\b",
                     corrected, flags=re.IGNORECASE):
            return False, "dialogue_artifact"

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

    def _clean_generated_text(self, generated: str) -> str:
        """Extract a single corrected sentence from model output."""
        text = generated.strip()
        text = re.sub(r"^(corrected sentence|correction|answer)\s*:\s*", "",
                      text, flags=re.IGNORECASE)
        text = text.splitlines()[0].strip() if text else ""
        text = text.strip(" \"'`")

        text = re.split(
            r"\s+(?:Human|Assistant|User)\s*:|[.!?]?\s*Human resources\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        # If the model still explains itself, keep the part before the explanation.
        for marker in [
            " Evidence:", " Original sentence:", " Explanation:", " Note:",
            " Human:", " Assistant:", " User:", "\nHuman:", "\nAssistant:", "\nUser:",
            " Human resources", " human resources",
        ]:
            if marker in text:
                text = text.split(marker, 1)[0].strip()

        sentence_match = re.match(r"^(.+?[.!?])(?:\s|$)", text)
        if sentence_match:
            text = sentence_match.group(1).strip()

        return text

    def _generate_candidate_texts(self, prompt: str) -> List[str]:
        """Generate one or more raw correction candidates."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=2048, padding=True,
        ).to(self.device)

        generation_count = min(self.num_candidates, max(self.num_beams, 1))
        if self.num_candidates > generation_count:
            logger.warning(
                "num_candidates=%s is larger than num_beams=%s; generating %s candidates",
                self.num_candidates,
                self.num_beams,
                generation_count,
            )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=5,
                num_beams=max(self.num_beams, generation_count),
                num_return_sequences=generation_count,
                early_stopping=True,
                do_sample=(self.temperature > 0),
                temperature=self.temperature if self.temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        return [
            self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            for output in outputs
        ]

    def _build_candidate_records(self, sentence: str, raw_outputs: List[str]) -> List[Dict]:
        """Clean and validate generated candidates, preserving rejected variants."""
        candidates = []
        seen = set()
        for idx, generated in enumerate(raw_outputs):
            corrected = self._clean_generated_text(generated)
            is_valid, reason = self._validate_correction(sentence, corrected)
            dedupe_key = corrected.lower().strip()
            if dedupe_key in seen:
                is_valid = False
                reason = "duplicate_candidate"
            seen.add(dedupe_key)
            candidates.append({
                "rank": idx,
                "raw": generated,
                "corrected": corrected,
                "format_valid": is_valid,
                "validation_reason": reason if not is_valid else None,
            })
        return candidates

    def _rerank_candidates(self, sentence: str, evidence_text: str,
                           candidates: List[Dict], nli_checker) -> Dict:
        """Score format-valid candidates with NLI and return the best candidate."""
        before = nli_checker.check_pair(evidence_text, sentence)
        valid_candidates = [c for c in candidates if c["format_valid"]]
        if not valid_candidates:
            return {
                "before": before,
                "selected": None,
                "accepted": False,
                "decision": "no_format_valid_candidate",
            }

        for candidate in valid_candidates:
            after = nli_checker.check_pair(evidence_text, candidate["corrected"])
            candidate["nli"] = after
            candidate["improved"] = (
                after["entailment_score"] > before["entailment_score"]
            )

        selected = max(
            valid_candidates,
            key=lambda c: c["nli"]["entailment_score"],
        )
        accepted = selected["nli"]["is_consistent"]
        return {
            "before": before,
            "selected": selected,
            "accepted": accepted,
            "decision": "accepted_by_nli" if accepted else "no_candidate_passed_nli",
        }

    def correct_single(self, sentence: str, evidence_text: str,
                       nli_checker=None) -> Dict:
        """Correct a single inconsistent sentence.

        Args:
            sentence: The inconsistent summary sentence.
            evidence_text: Evidence passage from source document.

        Returns dict with: original, corrected, success, evidence_used, failure_reason
        """
        try:
            prompt = CORRECTION_PROMPT.format(evidence=evidence_text, sentence=sentence)
            raw_outputs = self._generate_candidate_texts(prompt)
            candidates = self._build_candidate_records(sentence, raw_outputs)

            if nli_checker is not None:
                rerank = self._rerank_candidates(
                    sentence, evidence_text, candidates, nli_checker,
                )
                selected = rerank["selected"]
                has_format_candidate = selected is not None
                accepted = rerank["accepted"]
                corrected = selected["corrected"] if accepted else sentence
                verification = None
                if selected is not None and "nli" in selected:
                    after = selected["nli"]
                    before = rerank["before"]
                    verification = {
                        "verified": accepted and selected.get("improved", False),
                        "improved": selected.get("improved", False),
                        "fixed": (not before["is_consistent"]) and after["is_consistent"],
                        "original_entailment_score": before["entailment_score"],
                        "corrected_entailment_score": after["entailment_score"],
                        "original_label": before["label"],
                        "corrected_label": after["label"],
                    }

                result = {
                    "original": sentence,
                    "corrected": corrected,
                    "proposed_corrected": selected["corrected"] if selected else sentence,
                    "success": has_format_candidate,
                    "accepted_by_nli": accepted,
                    "rerank_decision": rerank["decision"],
                    "evidence_used": evidence_text[:500],
                    "failure_reason": None if has_format_candidate else rerank["decision"],
                    "candidates": candidates,
                }
                if verification is not None:
                    result["verification"] = verification
                return result

            selected = next((c for c in candidates if c["format_valid"]), candidates[0])
            is_valid = selected["format_valid"]
            corrected = selected["corrected"]
            reason = selected["validation_reason"]

            return {
                "original": sentence,
                "corrected": corrected if is_valid else sentence,
                "success": is_valid,
                "evidence_used": evidence_text[:500],
                "failure_reason": reason if not is_valid else None,
                "candidates": candidates,
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
                        report: str,
                        nli_checker=None) -> Dict:
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

            result = self.correct_single(sent["text"], best_ev["text"], nli_checker)
            result["sentence_index"] = sent["index"]
            corrections.append(result)

            # Patch only when no reranker is used, or when NLI accepts the candidate.
            if result["success"] and result.get("accepted_by_nli", True):
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

    def correct_batch(self, samples: List[Dict],
                      nli_checker=None,
                      checkpoint_path: Optional[str] = None,
                      existing_results: Optional[List[Dict]] = None) -> List[Dict]:
        """Batch correction. Adds 'correction' dict to each sample."""
        results = list(existing_results or [])
        done_ids = {r.get("sample_id") for r in results}
        for item in tqdm(samples, desc="Correcting", unit="sample"):
            if item.get("sample_id") in done_ids:
                continue

            record = dict(item)
            try:
                record["correction"] = self.correct_summary(
                    record["generated_summary"],
                    record.get("consistency", {}),
                    record["report"],
                    nli_checker=nli_checker,
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
            done_ids.add(record.get("sample_id"))

            if checkpoint_path:
                from ..utils.io import save_json
                save_json(results, checkpoint_path)

        return results
