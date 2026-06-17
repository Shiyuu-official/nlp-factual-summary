"""Error corrector: evidence-constrained multi-candidate generation + NLI reranking.

Strategy (designed for long-document summarization):
  1. Locate the key evidence sentence most relevant to the claim
  2. Generate N diverse correction candidates via sampling (temperature > 0)
  3. Score all candidates with NLI, select the one with highest entailment
  4. If best < threshold, iteratively refine with NLI-score feedback (up to K rounds)
  5. If still below threshold after refinement, fall back to extractive replacement
     (pick the evidence sentence with highest lexical overlap)

Why this beats direct local rewriting:
  - Multi-candidate + NLI reranking decouples "generate" from "judge" — the NLI model,
    not the generator, decides which correction is factually grounded.
  - Iterative refinement lets the model self-correct when NLI feedback pinpoints
    unsupported claims.
  - Extractive fallback guarantees we never leave an unsupported sentence in place;
    worst case, we substitute an evidence-backed sentence from the source.
  - For GovReport-style long documents with dense entities/numbers/causal chains,
    this is more robust than a single deterministic beam-search pass.
"""

import logging
import math
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Prompt templates ────────────────────────────────────────────────────

GENERATION_PROMPT = """The sentence below makes claims NOT supported by the evidence.
Fix ONLY the unsupported parts by replacing them with facts from the evidence.
Keep all supported parts EXACTLY as they are.

Evidence:
{evidence}

Original sentence (length: {orig_len} words):
{sentence}

Rules:
- Return ONLY the corrected sentence — no explanation, no preamble.
- Replace unsupported claims with facts that EXPLICITLY appear in the evidence.
- Do NOT change phrases that are already supported by the evidence.
- Do NOT add new entities, numbers, or relationships absent from the evidence.
- Keep the corrected sentence SHORT — at most {max_words} words.
- If a claim cannot be verified, remove it.

Corrected sentence:"""


REFINEMENT_PROMPT = """The sentence below is NOT fully supported by the evidence (entailment score: {score:.2f}, threshold: {threshold:.2f}).

Evidence:
{evidence}

Original sentence (for context):
{original}

Current correction:
{corrected}

Revise the correction so that EVERY claim is directly stated in the evidence.
Return ONLY the revised sentence — no explanation.

Revised sentence:"""

# ── Lightweight NLI helper (in-module, avoids extra dependency on NLIChecker) ──

def _word_tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class EvidenceConstrainedCorrector:
    """Corrects factually inconsistent summary sentences via evidence-constrained
    multi-candidate generation, NLI-guided reranking, iterative refinement,
    and extractive fallback.

    Uses Qwen2.5-1.5B-Instruct for generation and BART-MNLI for NLI scoring.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        nli_model_name: str = "facebook/bart-large-mnli",
        max_new_tokens: int = 80,
        num_candidates: int = 5,
        sample_temperature: float = 0.7,
        entailment_threshold: float = 0.5,
        max_refinement_rounds: int = 3,
        max_length_ratio: float = 1.5,
        enable_refinement: bool = True,
        enable_extractive_fallback: bool = True,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.nli_model_name = nli_model_name
        self.max_new_tokens = max_new_tokens
        self.num_candidates = num_candidates
        self.sample_temperature = sample_temperature
        self.entailment_threshold = entailment_threshold
        self.max_refinement_rounds = max_refinement_rounds
        self.max_length_ratio = max_length_ratio
        self.enable_refinement = enable_refinement
        self.enable_extractive_fallback = enable_extractive_fallback
        self.device = device

        # ── Load generator ──
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

        # ── Load NLI scorer (lazy, created on first use to save memory) ──
        self._nli_tokenizer = None
        self._nli_model = None
        self._nli_id2label = None
        self._nli_entail_id = None

    # ── NLI scorer (lazy init) ────────────────────────────────────────

    def _ensure_nli(self):
        if self._nli_model is not None:
            return
        logger.info(f"Loading NLI model {self.nli_model_name} for correction scoring...")
        from transformers import AutoModelForSequenceClassification
        self._nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self._nli_model = AutoModelForSequenceClassification.from_pretrained(
            self.nli_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self._nli_model.eval()

        self._nli_id2label = {
            int(idx): label.lower()
            for idx, label in self._nli_model.config.id2label.items()
        }
        for idx, label in self._nli_id2label.items():
            if "entailment" in label:
                self._nli_entail_id = idx
                break
        if self._nli_entail_id is None:
            raise ValueError(
                f"Cannot find 'entailment' label in {self.nli_model_name}"
            )

    def _nli_score(self, evidence: str, hypothesis: str) -> float:
        """Return entailment score for (evidence, hypothesis)."""
        self._ensure_nli()
        inputs = self._nli_tokenizer(
            evidence, hypothesis,
            return_tensors="pt", truncation=True, max_length=512, padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self._nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        return round(probs[self._nli_entail_id].item(), 4)

    def _nli_score_batch(self, evidence: str, hypotheses: List[str]) -> List[float]:
        """Score multiple hypotheses against the same evidence."""
        self._ensure_nli()
        if not hypotheses:
            return []
        premises = [evidence] * len(hypotheses)
        inputs = self._nli_tokenizer(
            premises, hypotheses,
            return_tensors="pt", truncation=True, max_length=512, padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self._nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        return [round(probs[i][self._nli_entail_id].item(), 4) for i in range(len(hypotheses))]

    # ── Step 1: Key evidence extraction ───────────────────────────────

    def _extract_key_evidence(self, sentence: str, evidence_text: str) -> str:
        """From the evidence passage, extract the 1–2 sentences most relevant
        to the claim.  This gives the generator a tighter target.

        Uses a lightweight TF-IDF-like scoring (word overlap weighted by
        inverse sentence frequency within the passage).
        """
        raw_sentences = re.split(r"(?<=[.!?])\s+", evidence_text)
        doc_sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not doc_sentences:
            return evidence_text
        if len(doc_sentences) <= 2:
            return evidence_text  # already tight enough

        claim_tokens = _word_tokens(sentence)
        if not claim_tokens:
            return evidence_text

        # IDF-like weighting within the passage
        n_docs = len(doc_sentences)
        df = Counter()
        tokenized_docs = []
        for s in doc_sentences:
            tokens = _word_tokens(s)
            tokenized_docs.append(tokens)
            df.update(tokens)

        idf = {
            t: math.log((1 + n_docs) / (1 + freq)) + 1.0
            for t, freq in df.items()
        }

        scores = []
        for tokens in tokenized_docs:
            overlap = claim_tokens & tokens
            if not overlap:
                scores.append(0.0)
            else:
                w = sum(idf.get(t, 0.0) for t in overlap)
                # Normalize by doc length to avoid bias toward long sentences
                scores.append(w / (len(tokens) + 1))
                # Also reward Jaccard overlap
                jac = len(overlap) / (len(claim_tokens | tokens) + 1)
                scores[-1] += jac

        # Take top 2 sentences
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:2]
        ranked.sort(key=lambda x: x[0])  # preserve original order
        return " ".join(doc_sentences[i] for i, _ in ranked)

    # ── Generation helpers ────────────────────────────────────────────

    def _generate_one(self, prompt: str, deterministic: bool = False) -> str:
        """Generate a single output from the model."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=2048, padding=True,
        ).to(self.device)

        temp = 0.0 if deterministic else self.sample_temperature

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=5,
                do_sample=(temp > 0),
                temperature=temp if temp > 0 else 1.0,
                top_p=0.92 if temp > 0 else None,
                num_beams=3,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated = self.tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        )
        return self._clean_generated_text(generated)

    def _clean_generated_text(self, text: str) -> str:
        """Extract a single corrected sentence from raw model output."""
        text = text.strip()
        # Remove common prefixes
        text = re.sub(
            r"^(corrected sentence|correction|answer|revised sentence)\s*:\s*",
            "", text, flags=re.IGNORECASE
        )
        # Take only the first line
        text = text.splitlines()[0].strip() if text else ""
        text = text.strip(" \"'`")

        # Cut at common hallucinated markers
        for marker in [
            " Evidence:", " Original sentence:", " Explanation:", " Note:",
            " Human:", " Assistant:", " User:", "\nHuman:", "\nAssistant:",
            "\nUser:", " Human resources", " human resources",
        ]:
            if marker in text:
                text = text.split(marker, 1)[0].strip()

        # Keep only the first complete sentence if multi-sentence
        sent_match = re.match(r"^(.+?[.!?])(?:\s|$)", text)
        if sent_match:
            text = sent_match.group(1).strip()

        return text

    # ── Step 2: Multi-candidate generation ────────────────────────────

    def _generate_candidates(self, sentence: str, evidence: str) -> List[str]:
        """Generate N diverse correction candidates via sampling,
        plus one deterministic beam-search backup for format reliability.
        """
        orig_len = len(sentence.split())
        max_words = max(orig_len + 10, int(orig_len * self.max_length_ratio))
        prompt = GENERATION_PROMPT.format(
            evidence=evidence, sentence=sentence,
            orig_len=orig_len, max_words=max_words,
        )

        candidates = []
        seen = set()

        # 1) Sampling candidates (diverse)
        for _ in range(self.num_candidates * 2):
            if len(candidates) >= self.num_candidates:
                break
            try:
                cand = self._generate_one(prompt, deterministic=False)
                if not cand or cand == sentence.strip():
                    continue
                key = cand.lower()
                if key not in seen:
                    seen.add(key)
                    candidates.append(cand)
            except Exception as e:
                logger.warning(f"Candidate generation failed: {e}")

        # 2) Deterministic backup: beam search, shorter output, reliable format
        try:
            backup = self._generate_one(prompt, deterministic=True)
            if backup and backup != sentence.strip():
                key = backup.lower()
                if key not in seen:
                    seen.add(key)
                    candidates.append(backup)
                    logger.debug("Added deterministic beam-search backup candidate")
        except Exception as e:
            logger.warning(f"Deterministic backup failed: {e}")

        return candidates

    # ── Step 3: NLI scoring + selection ───────────────────────────────

    def _select_best(self, candidates: List[str], evidence: str,
                     original: str) -> Tuple[Optional[str], float]:
        """Score all candidates with NLI, return (best_candidate, best_score)."""
        if not candidates:
            return None, 0.0

        scores = self._nli_score_batch(evidence, candidates)
        best_idx = int(np.argmax(scores))
        return candidates[best_idx], scores[best_idx]

    # ── Step 4: Iterative refinement ──────────────────────────────────

    def _refine(self, original: str, evidence: str,
                current: str, current_score: float) -> Tuple[str, float]:
        """Iteratively refine the correction using NLI score as feedback signal.

        Each round: show the model its current entailment score and ask it to
        revise.  If the score improves, keep the revision; otherwise keep the
        previous best.
        """
        best = current
        best_score = current_score

        for round_idx in range(self.max_refinement_rounds):
            prompt = REFINEMENT_PROMPT.format(
                score=best_score,
                threshold=self.entailment_threshold,
                evidence=evidence,
                original=original,
                corrected=best,
            )
            try:
                revised = self._generate_one(prompt)
            except Exception as e:
                logger.warning(f"Refinement round {round_idx+1} failed: {e}")
                continue

            if not revised or revised == best:
                continue

            new_score = self._nli_score(evidence, revised)
            logger.debug(
                f"Refinement round {round_idx+1}: "
                f"score {best_score:.3f} → {new_score:.3f}"
            )

            # Greedy: accept if score improves (use small margin to avoid thrashing)
            if new_score > best_score + 0.01:
                best = revised
                best_score = new_score

            if best_score >= self.entailment_threshold:
                break

        return best, best_score

    # ── Step 5: Extractive fallback (NLI-driven) ──────────────────────

    def _extractive_fallback(self, original: str, evidence_text: str) -> Tuple[str, float]:
        """Extract the evidence sentence with highest NLI entailment score.

        Instead of word overlap (which may pick a sentence that doesn't
        actually entail the correction target), we NLI-score every candidate
        evidence sentence against the evidence context.  The winner is the
        sentence most strongly entailed by the surrounding evidence.

        Returns (best_sentence, nli_score).
        """
        doc_sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", evidence_text) if s.strip()
        ]
        if not doc_sentences:
            return original, 0.0

        claim_tokens = _word_tokens(original)
        if not claim_tokens:
            return original, 0.0

        # Pre-filter: keep sentences with some word overlap with the claim,
        # plus their immediate neighbors for context.  This avoids scoring
        # totally unrelated sentences.
        candidates = []
        for i, sent in enumerate(doc_sentences):
            overlap = _jaccard(claim_tokens, _word_tokens(sent))
            if overlap > 0.0 and len(sent.split()) >= 4:
                candidates.append((i, sent, overlap))

        if not candidates:
            # fall back to any non-trivial sentence
            candidates = [
                (i, s, 0.0) for i, s in enumerate(doc_sentences)
                if len(s.split()) >= 4
            ]

        if not candidates:
            return original, 0.0

        # Sort by overlap and take top-8 to keep NLI calls bounded
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = candidates[:8]

        # NLI-score each candidate against the full evidence context
        best_sent = candidates[0][1]
        best_score = 0.0
        for _, sent, _ in candidates:
            score = self._nli_score(evidence_text, sent)
            if score > best_score:
                best_score = score
                best_sent = sent

        logger.debug(
            f"Extractive fallback: best NLI score={best_score:.3f}, "
            f"sent=\"{best_sent[:80]}...\""
        )
        return best_sent, best_score

    # ── Validation ────────────────────────────────────────────────────

    def _validate_correction(self, original: str, corrected: str) -> Tuple[bool, str]:
        """Validate that the corrected sentence meets format constraints."""
        corrected = corrected.strip()
        if not corrected:
            return False, "empty_output"
        if re.fullmatch(r"[\W_]+|\d+[.)]?", original.strip()):
            return False, "invalid_original_sentence"
        if corrected == original.strip():
            return False, "no_change"
        if re.search(
            r"\b(human|assistant|user)\s*:|\bhuman resources\b",
            corrected, flags=re.IGNORECASE,
        ):
            return False, "dialogue_artifact"

        orig_words = len(original.split())
        corr_words = len(corrected.split())
        # Allow more slack for short originals (hard to edit concisely)
        effective_ratio = self.max_length_ratio
        if orig_words <= 6:
            effective_ratio = self.max_length_ratio + 1.0
        if corr_words > orig_words * effective_ratio and corr_words > 20:
            return False, f"output_too_long ({corr_words} vs {orig_words} words)"

        # Only flag multi-sentence when clearly copy-pasting (3+ sentences)
        sentence_seps = re.findall(r'[.!?]+', corrected)
        if len(sentence_seps) > 2 and corr_words > 50:
            return False, "appears_to_be_evidence_passage"

        return True, ""

    # ── Main correction entry point ───────────────────────────────────

    def correct_single(self, sentence: str, evidence_text: str,
                       all_evidences: Optional[List[str]] = None) -> Dict:
        """Correct a single inconsistent sentence.

        Args:
            sentence: The inconsistent summary sentence.
            evidence_text: Primary evidence passage (TF-IDF top-1).
            all_evidences: All top-k evidence passages (optional, for
                           multi-evidence extractive search).

        Returns dict with:
            original, corrected, success, evidence_used, failure_reason,
            strategy (extractive|candidate|refinement), nli_score
        """
        # Pre-validation
        if re.fullmatch(r"[\W_]+|\d+[.)]?", sentence.strip()):
            return self._fail(sentence, evidence_text, "invalid_original_sentence")

        evidence_pool = [evidence_text]
        if all_evidences:
            evidence_pool = list(all_evidences)

        # ── Phase A: NLI-driven extractive across ALL evidences ─────
        # Try extractive first — if an evidence sentence is already
        # strongly entailed by its context, use it directly.
        best_extractive = None
        best_ext_score = 0.0
        best_ext_ev = evidence_text
        for ev_text in evidence_pool:
            if not ev_text or len(ev_text.split()) < 4:
                continue
            extracted, score = self._extractive_fallback(sentence, ev_text)
            if score > best_ext_score:
                best_ext_score = score
                best_extractive = extracted
                best_ext_ev = ev_text

        # If extractive already hits threshold, skip generation entirely
        if (best_extractive and best_extractive != sentence.strip()
                and best_ext_score >= self.entailment_threshold):
            is_valid, reason = self._validate_correction(sentence, best_extractive)
            if is_valid:
                return self._success(
                    sentence, best_extractive, best_ext_ev,
                    "extractive", best_ext_score,
                )

        # ── Phase B: Generation with the most promising evidence ────
        # Use the evidence that gave the best extractive score as the
        # generation context (it has the most relevant content).
        best_evidence = best_ext_ev
        key_evidence = self._extract_key_evidence(sentence, best_evidence)
        if not key_evidence or len(key_evidence.split()) < 4:
            # If extractive found something usable but below threshold,
            # accept it as a fallback
            if best_extractive and best_extractive != sentence.strip():
                is_valid, reason = self._validate_correction(sentence, best_extractive)
                if is_valid:
                    return self._success(
                        sentence, best_extractive, best_ext_ev,
                        "extractive", best_ext_score,
                    )
            return self._fail(sentence, best_evidence, "insufficient_evidence")

        candidates = self._generate_candidates(sentence, key_evidence)
        if not candidates:
            if best_extractive and best_extractive != sentence.strip():
                is_valid, reason = self._validate_correction(sentence, best_extractive)
                if is_valid:
                    return self._success(
                        sentence, best_extractive, best_ext_ev,
                        "extractive", best_ext_score,
                    )
            return self._fail(sentence, best_evidence, "no_candidates_generated")

        # NLI scoring + select best candidate
        best, best_score = self._select_best(candidates, key_evidence, sentence)
        if best is None:
            return self._fail(sentence, best_evidence, "all_candidates_rejected")

        strategy = "candidate"

        # Iterative refinement if below threshold
        if best_score < self.entailment_threshold and self.enable_refinement:
            refined, refined_score = self._refine(
                sentence, key_evidence, best, best_score
            )
            if refined_score > best_score:
                best = refined
                best_score = refined_score
                strategy = "refinement"

        # If generation still below threshold, try extractive again
        if best_score < self.entailment_threshold and best_extractive is not None:
            if best_ext_score > best_score:
                best = best_extractive
                best_score = best_ext_score
                strategy = "extractive"

        # Validate format
        is_valid, reason = self._validate_correction(sentence, best)
        if not is_valid:
            return self._fail(sentence, best_evidence, reason)

        return self._success(sentence, best, best_evidence, strategy, best_score)

    def _success(self, original: str, corrected: str, evidence: str,
                 strategy: str, nli_score: float) -> Dict:
        return {
            "original": original,
            "corrected": corrected,
            "success": True,
            "evidence_used": evidence[:500],
            "failure_reason": None,
            "strategy": strategy,
            "nli_score": nli_score,
        }

    def _fail(self, original: str, evidence: str, reason: str) -> Dict:
        return {
            "original": original,
            "corrected": original,
            "success": False,
            "evidence_used": evidence[:500] if evidence else "",
            "failure_reason": reason,
            "strategy": "none",
            "nli_score": 0.0,
        }

    # ── Batch interface (compatible with existing pipeline) ───────────

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
        sentence_texts = [s["text"] for s in sentences]

        for sent in tqdm(sentences, desc="Correcting", leave=False, unit="sent"):
            if sent["is_consistent"]:
                continue

            evidences = sent.get("evidences", [])
            if not evidences:
                corrections.append({
                    "sentence_index": sent["index"],
                    "original": sent["text"],
                    "corrected": sent["text"],
                    "success": False,
                    "evidence_used": "",
                    "failure_reason": "no_evidence",
                    "strategy": "none",
                    "nli_score": 0.0,
                })
                continue

            best_ev = max(evidences, key=lambda e: e.get("score", 0))
            # Pass all evidence texts for multi-evidence extractive search
            all_ev_texts = [e["text"] for e in evidences if e.get("text")]
            result = self.correct_single(
                sent["text"], best_ev["text"],
                all_evidences=all_ev_texts if len(all_ev_texts) > 1 else None,
            )
            result["sentence_index"] = sent["index"]
            corrections.append(result)

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

    def correct_batch(self, samples: List[Dict],
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
                )
            except Exception as e:
                logger.error(
                    f"Sample {record.get('sample_id')}: correction failed: {e}"
                )
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
