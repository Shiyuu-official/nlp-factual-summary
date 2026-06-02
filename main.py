#!/usr/bin/env python
"""main.py — Entry point for the NLP factual summary pipeline.

Usage:
    python main.py --mode test     # 5 samples, fast validation (~10 min)
    python main.py --mode full     # 500+ samples, final experiment
    python main.py --mode test --stage 4   # Run only correction stage
"""

import os
import argparse

# Use HF mirror for model downloads (required in mainland China due to SSL proxy issues)
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Note: HF_HOME / caches use HuggingFace defaults (~/.cache/huggingface).
# To use a custom cache directory, set PROJECT_CACHE_DIR before running.

from src.utils.config import load_config
from src.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Long-document factual summary pipeline: "
                    "summarization → NLI consistency → error correction"
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--mode", choices=["test", "full"], default=None,
                        help="Override config mode (test=5 samples, full=500+)")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5, 6], default=None,
                        help="Run only a specific stage (1=data, 2=summarize, "
                             "3=consistency, 4=correct, 5=evaluate, 6=analyze)")
    args = parser.parse_args()

    config = load_config(args.config, cli_mode=args.mode)
    pipeline = Pipeline(config)

    if args.stage:
        pipeline.run_from_stage(args.stage)
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
