"""Configuration loader: YAML -> typed PipelineConfig dataclass."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import yaml


@dataclass
class PipelineConfig:
    """Typed configuration resolved from config.yaml. All fields are flat for simple access."""
    mode: str = "test"
    seed: int = 42
    num_samples: int = 5

    # Dataset
    dataset_name: str = "ccdv/govreport-summarization"
    dataset_split: str = "validation"
    dataset_cache_dir: str = "./data_cache"

    # Summarizer
    summarizer_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    summarizer_chunk_size: int = 2000
    summarizer_chunk_overlap: int = 200
    summarizer_max_summary_length: int = 300
    summarizer_max_chunk_summary_length: int = 150
    summarizer_num_beams: int = 4

    # Consistency
    consistency_nli_model: str = "facebook/bart-large-mnli"
    consistency_entailment_threshold: float = 0.5
    consistency_evidence_top_k: int = 3
    consistency_sentence_window: int = 3
    consistency_evidence_mode: str = "word_overlap"

    # Semantic retrieval
    semantic_retrieval_model: str = "all-MiniLM-L6-v2"
    semantic_retrieval_batch_size: int = 32
    semantic_retrieval_enabled: bool = False

    # Corrector
    corrector_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    corrector_nli_model: str = "facebook/bart-large-mnli"
    corrector_max_new_tokens: int = 100
    corrector_num_candidates: int = 5
    corrector_sample_temperature: float = 0.5
    corrector_max_refinement_rounds: int = 3
    corrector_max_length_ratio: float = 2.0
    corrector_enable_refinement: bool = True
    corrector_enable_extractive_fallback: bool = True

    # Evaluation
    evaluation_rouge_types: List[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    evaluation_use_stemmer: bool = True

    # Analysis
    analysis_length_bins: List[List[int]] = field(default_factory=list)
    analysis_num_case_studies: int = 10

    # Output
    output_root_dir: str = "./results"
    output_save_intermediate: bool = True

    # Logging
    log_level: str = "INFO"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Returns new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _flatten_config(raw: dict) -> PipelineConfig:
    """Convert nested config dict to flat PipelineConfig dataclass."""
    semantic_enabled = raw.get("semantic_retrieval", {}).get("enabled", False)
    evidence_mode = raw["consistency"]["evidence_mode"]
    if semantic_enabled:
        evidence_mode = "semantic"

    return PipelineConfig(
        mode=raw.get("mode", "test"),
        seed=raw.get("seed", 42),
        num_samples=raw.get("num_samples", 5),
        # Dataset
        dataset_name=raw["dataset"]["name"],
        dataset_split=raw["dataset"]["split"],
        dataset_cache_dir=raw["dataset"]["cache_dir"],
        # Summarizer
        summarizer_model=raw["summarizer"]["model"],
        summarizer_chunk_size=raw["summarizer"]["chunk_size"],
        summarizer_chunk_overlap=raw["summarizer"]["chunk_overlap"],
        summarizer_max_summary_length=raw["summarizer"]["max_summary_length"],
        summarizer_max_chunk_summary_length=raw["summarizer"]["max_chunk_summary_length"],
        summarizer_num_beams=raw["summarizer"]["num_beams"],
        # Consistency
        consistency_nli_model=raw["consistency"]["nli_model"],
        consistency_entailment_threshold=raw["consistency"]["entailment_threshold"],
        consistency_evidence_top_k=raw["consistency"]["evidence_top_k"],
        consistency_sentence_window=raw["consistency"]["sentence_window"],
        consistency_evidence_mode=evidence_mode,
        # Semantic retrieval
        semantic_retrieval_model=raw["semantic_retrieval"]["model"],
        semantic_retrieval_batch_size=raw["semantic_retrieval"]["batch_size"],
        semantic_retrieval_enabled=semantic_enabled,
        # Corrector
        corrector_model=raw["corrector"]["model"],
        corrector_nli_model=raw["corrector"].get("nli_model", "facebook/bart-large-mnli"),
        corrector_max_new_tokens=raw["corrector"]["max_new_tokens"],
        corrector_num_candidates=raw["corrector"].get("num_candidates", 5),
        corrector_sample_temperature=raw["corrector"].get("sample_temperature", 0.7),
        corrector_max_refinement_rounds=raw["corrector"].get("max_refinement_rounds", 3),
        corrector_max_length_ratio=raw["corrector"]["max_length_ratio"],
        corrector_enable_refinement=raw["corrector"].get("enable_refinement", True),
        corrector_enable_extractive_fallback=raw["corrector"].get("enable_extractive_fallback", True),
        # Evaluation
        evaluation_rouge_types=raw["evaluation"]["rouge_types"],
        evaluation_use_stemmer=raw["evaluation"]["use_stemmer"],
        # Analysis
        analysis_length_bins=raw["analysis"]["length_bins"],
        analysis_num_case_studies=raw["analysis"]["num_case_studies"],
        # Output
        output_root_dir=raw["output"]["root_dir"],
        output_save_intermediate=raw["output"]["save_intermediate"],
        # Logging
        log_level=raw["logging"]["level"],
    )


def load_config(path: str, cli_mode: Optional[str] = None) -> PipelineConfig:
    """Load YAML config, merge defaults + mode profile, return typed config.

    Args:
        path: Path to config.yaml
        cli_mode: Optional mode override from CLI (e.g. --mode test)

    Returns:
        PipelineConfig dataclass instance
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mode = cli_mode or raw.get("mode", "test")

    # Start with defaults, then merge mode-specific overrides
    merged = _deep_merge(raw["defaults"], raw.get(mode, {}))

    # CLI mode always wins
    merged["mode"] = mode

    return _flatten_config(merged)
