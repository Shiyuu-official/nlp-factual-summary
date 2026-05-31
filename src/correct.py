"""
Factual error correction module.
For each inconsistent sentence (neutral / contradiction), finds supporting
evidence in the source report, then rewrites the sentence.

Two strategies:
  A (Qwen): LLM instruction-following correction (needs model download)
  B (BART): Evidence-based abstractive rewriting via summarization (offline-ready)
"""

import json
import os
import re
from datetime import datetime

import nltk
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

nltk.download("punkt_tab", quiet=True)

# ── Config ──────────────────────────────────────────────────────
CORRECTION_MODEL = "facebook/bart-large-cnn"  # cached, no download needed
MAX_EVIDENCE_SENTENCES = 3
CONTEXT_WINDOW = 2
MAX_INPUT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_DIR = "outputs"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Evidence extraction ─────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def find_evidence(hypothesis: str, document: str,
                  top_k: int = MAX_EVIDENCE_SENTENCES,
                  window: int = CONTEXT_WINDOW) -> list[str]:
    """Lexical-overlap based evidence retrieval from source document."""
    doc_sents = split_sentences(document)
    if not doc_sents:
        return []

    hypo_words = set(hypothesis.lower().split())
    if not hypo_words:
        return []

    scored = []
    for i, sent in enumerate(doc_sents):
        start = max(0, i - window)
        end = min(len(doc_sents), i + window + 1)
        ctx = " ".join(doc_sents[start:end])
        ctx_words = set(ctx.lower().split())
        overlap = len(hypo_words & ctx_words) / len(hypo_words)
        scored.append((ctx, overlap))

    scored.sort(key=lambda x: x[1], reverse=True)

    seen, unique = set(), []
    for text, score in scored:
        if text not in seen and score > 0:
            seen.add(text)
            unique.append(text)
    return unique[:top_k]


# ── Correction model ────────────────────────────────────────────
def load_correction_model(model_name: str = CORRECTION_MODEL):
    """Load a seq2seq model for evidence-based rewriting (uses local cache)."""
    print(f"Loading correction model {model_name} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()
    print("  Correction model loaded.")
    return tokenizer, model


def correct_sentence(incorrect: str, evidences: list[str],
                     tokenizer, model) -> dict:
    """
    Use evidence text as the source and generate a single factually-correct
    summary sentence via BART.  This leverages BART's training objective:
    produce faithful summaries from source text.
    """
    evidence_text = " ".join(evidences)

    # Prefix to hint the model towards a concise single-sentence output
    input_text = f"summarize: {evidence_text}"

    inputs = tokenizer(
        input_text, max_length=MAX_INPUT_TOKENS, truncation=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=MAX_OUTPUT_TOKENS,
            min_length=10,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    corrected = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    corrected = re.sub(r"\s+", " ", corrected).strip()

    # Determine success
    success = bool(corrected) and corrected.lower() != incorrect.lower()

    return {
        "original": incorrect,
        "corrected": corrected,
        "evidence": evidences,
        "success": success,
    }


# ── Main pipeline ────────────────────────────────────────────────
def correct_consistency_output(input_file: str, tokenizer, model):
    """Load consistency JSON, correct every inconsistent sentence."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_inconsistent = sum(d["nli_inconsistent_count"] for d in data)
    print(f"\nLoaded {len(data)} samples, "
          f"{total_inconsistent} inconsistent sentences to correct.\n")

    corrected_count = 0

    for item in data:
        report = item["report"]
        item["corrections"] = []
        item["corrected_sentences"] = []
        item["corrected_summary"] = ""

        for i, (sent, nli) in enumerate(
            zip(item["generated_sentences"], item["nli_results"])
        ):
            if nli["label"] in ("contradiction", "neutral"):
                evidences = find_evidence(sent, report)
                if evidences:
                    result = correct_sentence(sent, evidences, tokenizer, model)
                else:
                    result = {
                        "original": sent, "corrected": sent,
                        "evidence": [], "success": False,
                    }

                item["corrections"].append(result)
                item["corrected_sentences"].append(result["corrected"])
                if result["success"]:
                    corrected_count += 1

                ok = "OK" if result["success"] else "  "
                label_short = {"contradiction": "CON", "neutral": "NEU"}
                short = result["corrected"][:80]
                print(f"  [{item['id']}][sent {i}] [{ok}] "
                      f"{label_short.get(nli['label'], nli['label'])} -> {short}")
            else:
                item["corrected_sentences"].append(sent)

        item["corrected_summary"] = " ".join(item["corrected_sentences"])

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"corrected_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print(f"Correction Results")
    print(f"{'='*50}")
    print(f"  Total inconsistent      : {total_inconsistent}")
    print(f"  Successfully corrected  : {corrected_count}")
    if total_inconsistent:
        print(f"  Correction rate         : "
              f"{100*corrected_count/total_inconsistent:.1f}%")
    print(f"  Saved to {output_path}")

    return data


def main(input_file=None):
    if not input_file:
        files = sorted(
            [f for f in os.listdir(INPUT_DIR) if f.startswith("consistency_")]
        )
        if not files:
            print("No consistency output found. Run consistency.py first.")
            return
        input_file = os.path.join(INPUT_DIR, files[-1])
        print(f"Using latest consistency: {input_file}")

    tokenizer, model = load_correction_model()
    correct_consistency_output(input_file, tokenizer, model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="Path to consistency output JSON")
    args = parser.parse_args()
    main(args.input)
