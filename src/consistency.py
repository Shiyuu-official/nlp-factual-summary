"""
Factual consistency detection via sentence-level NLI.
Loads baseline output, checks each generated sentence against the
source report for factual consistency (entailment / contradiction / neutral).
"""

import json
import os
from datetime import datetime

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# -- Config --
NLI_MODEL = "facebook/bart-large-mnli"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_DIR = "outputs"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    print(f"Loading NLI model {NLI_MODEL} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE)
    model.eval()
    return tokenizer, model


def check_sentence(passage, sentence, tokenizer, model):
    """
    Return {label, score} for a single sentence against the passage.
    label is one of: entailment, contradiction, neutral.
    """
    inputs = tokenizer(
        passage, sentence,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    idx = torch.argmax(probs).item()
    return {
        "label": model.config.id2label[idx],
        "score": round(probs[idx].item(), 4),
        "probs": {
            model.config.id2label[i]: round(p, 4)
            for i, p in enumerate(probs.tolist())
        },
    }


def evaluate_baseline(input_file, tokenizer, model):
    """Load baseline JSON, run NLI on each sentence, save enriched output."""
    with open(input_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    for item in results:
        report = item["report"]
        sent_results = []
        for sent in item["generated_sentences"]:
            sent_results.append(check_sentence(report, sent, tokenizer, model))

        item["nli_results"] = sent_results
        inconsistent = [
            s for s in sent_results
            if s["label"] in ("contradiction", "neutral")
        ]
        item["nli_inconsistent_count"] = len(inconsistent)
        item["nli_total_sentences"] = len(sent_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"consistency_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary stats
    total_sents = sum(r["nli_total_sentences"] for r in results)
    total_incon = sum(r["nli_inconsistent_count"] for r in results)
    print(f"\nNLI Consistency Results ({len(results)} samples)")
    print(f"  Total sentences    : {total_sents}")
    print(f"  Inconsistent       : {total_incon} ({100*total_incon/total_sents:.1f}%)")
    print(f"  Saved to {output_path}")

    return results


def main(input_file=None):
    if not input_file:
        files = sorted([f for f in os.listdir(INPUT_DIR) if f.startswith("baseline_")])
        if not files:
            print("No baseline output found. Run baseline.py first.")
            return
        input_file = os.path.join(INPUT_DIR, files[-1])
        print(f"Using latest baseline: {input_file}")

    tokenizer, model = load_model()
    evaluate_baseline(input_file, tokenizer, model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="Path to baseline output JSON")
    args = parser.parse_args()
    main(args.input)
