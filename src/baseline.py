"""Baseline: long-document summarization with BART on GovReport.

Since BART has a 1024-token limit, long documents are processed in
overlapping chunks: each chunk is summarized, then the concatenated
partial summaries are summarized again to produce the final output.
"""

import json
import os
from datetime import datetime

import nltk
import torch
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

nltk.download("punkt_tab", quiet=True)


# -- Config --
MODEL_NAME = "facebook/bart-large-cnn"
DATASET_NAME = "ccdv/govreport-summarization"
MAX_INPUT_TOKENS = 1024     # BART encoder limit
CHUNK_OVERLAP = 100          # token overlap between chunks
MAX_OUTPUT_TOKENS = 200      # per-chunk summary length
FINAL_MAX_OUTPUT = 512       # final summary max length
NUM_SAMPLES = 10
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(OUTPUT_DIR, f"baseline_{timestamp}.json")


def load_data(split="train", num_samples=NUM_SAMPLES):
    print(f"Loading {DATASET_NAME} ({split}) ...")
    ds = load_dataset(DATASET_NAME, split=split)

    samples = []
    for i, item in enumerate(ds):
        if num_samples and i >= num_samples:
            break
        report = item.get("report", "") or item.get("document", "")
        summary = item.get("summary", "") or item.get("highlights", "")
        if report and summary:
            samples.append({"id": str(i), "report": report, "reference": summary})

    print(f"  Loaded {len(samples)} samples")
    return samples


def load_model():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def chunk_text(text, tokenizer, max_tokens=MAX_INPUT_TOKENS, overlap=CHUNK_OVERLAP):
    """Split text into overlapping token chunks."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]

    chunks = []
    stride = max_tokens - overlap
    for start in range(0, len(token_ids), stride):
        chunk_ids = token_ids[start : start + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if start + max_tokens >= len(token_ids):
            break
    return chunks


def summarize_chunk(text, tokenizer, model, max_output=MAX_OUTPUT_TOKENS):
    """Summarize a single text chunk."""
    inputs = tokenizer(
        text, max_length=MAX_INPUT_TOKENS, truncation=True,
        padding=False, return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_length=max_output, min_length=30,
            num_beams=4, early_stopping=True, no_repeat_ngram_size=3,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def generate_summary(report, tokenizer, model):
    """Chunk → summarize each chunk → combine → final summarize."""
    chunks = chunk_text(report, tokenizer)
    if len(chunks) == 1:
        return summarize_chunk(report, tokenizer, model, FINAL_MAX_OUTPUT)

    chunk_summaries = []
    for chunk in chunks:
        s = summarize_chunk(chunk, tokenizer, model, MAX_OUTPUT_TOKENS)
        if s:
            chunk_summaries.append(s)

    combined = " ".join(chunk_summaries)
    final = summarize_chunk(combined, tokenizer, model, FINAL_MAX_OUTPUT)
    return final


def generate_summaries(samples, tokenizer, model):
    results = []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    for idx, s in enumerate(samples):
        print(f"  [{idx+1}/{len(samples)}] Generating summary for sample {s['id']} ...")

        gen_text = generate_summary(s["report"], tokenizer, model)
        ref_sents = sent_tokenize(s["reference"])
        gen_sents = sent_tokenize(gen_text)
        scores = scorer.score(s["reference"], gen_text)

        results.append({
            "id": s["id"],
            "report": s["report"],
            "reference": s["reference"],
            "reference_sentences": ref_sents,
            "generated": gen_text,
            "generated_sentences": gen_sents,
            "num_chunks": len(chunk_text(s["report"], tokenizer)),
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        })

    return results


def print_summary(results):
    avg_r1 = sum(r["rouge1"] for r in results) / len(results)
    avg_r2 = sum(r["rouge2"] for r in results) / len(results)
    avg_rL = sum(r["rougeL"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"Baseline Results ({len(results)} samples)")
    print(f"{'='*50}")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  ROUGE-1 : {avg_r1:.4f}")
    print(f"  ROUGE-2 : {avg_r2:.4f}")
    print(f"  ROUGE-L : {avg_rL:.4f}")


def main():
    samples = load_data()
    tokenizer, model = load_model()
    results = generate_summaries(samples, tokenizer, model)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
