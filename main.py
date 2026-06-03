#!/usr/bin/env python
"""main.py — Entry point for the NLP factual summary pipeline.

Usage:
    python main.py --mode test     # 5 samples, fast validation (~10 min)
    python main.py --mode full     # 500+ samples, final experiment
    python main.py --mode test --stage 4   # Run only correction stage
"""

import os
import argparse

# Optional cache override for GPU servers. Set PROJECT_CACHE_DIR explicitly when needed.
cache_dir = os.environ.get("PROJECT_CACHE_DIR")
if cache_dir:
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "models"))
    os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "huggingface"))
    os.environ.setdefault("DATASETS_CACHE", os.path.join(cache_dir, "datasets"))
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
