"""Pipeline orchestrator: wires all stages together.

Stages are independent and can be run individually for partial re-runs.
Each stage appends fields to a list of sample dicts that flow through the pipeline.
"""

import logging
import os
import time
import torch
from typing import List, Dict, Optional

from .utils.config import PipelineConfig
from .utils.io import save_json, load_json, timestamped_dir
from .utils.logging import setup_logging
from .data.loader import GovReportDataLoader
from .summarization.summarizer import ChunkedSummarizer
from .consistency.sentence_splitter import SentenceSplitter
from .consistency.evidence_retrieval import create_retriever
from .consistency.nli_checker import NLIChecker
from .correction.corrector import LocalEditCorrector
from .evaluation.rouge import RougeEvaluator
from .evaluation.metrics import compare_before_after
from .analysis.length_impact import analyze_length_impact, plot_length_impact
from .analysis.case_study import collect_correction_cases, print_case_report

logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline orchestrator for the factual summary project.

    Usage:
        config = load_config("config.yaml")
        pipeline = Pipeline(config)
        pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run_dir = None
        self.logger = logging.getLogger(__name__)

    def _setup_run_dir(self) -> str:
        """Create timestamped output directory and save config snapshot."""
        self.run_dir = timestamped_dir(self.config.output_root_dir)
        save_json(
            {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None)))
             else v for k, v in vars(self.config).items()},
            os.path.join(self.run_dir, "config_applied.json"),
        )
        return self.run_dir

    # ── Stage 1: Data Loading ──────────────────────────────────────────

    def stage1_load_data(self) -> List[Dict]:
        """Load GovReport dataset."""
        logger.info("=" * 60)
        logger.info("STAGE 1: Data Loading")
        logger.info("=" * 60)

        loader = GovReportDataLoader(
            dataset_name=self.config.dataset_name,
        )
        samples = loader.load(
            split=self.config.dataset_split,
            max_samples=self.config.num_samples,
            seed=self.config.seed,
        )

        stats = GovReportDataLoader.compute_statistics(samples)
        logger.info(f"Data statistics: {stats}")

        if self.config.output_save_intermediate:
            save_json(stats, os.path.join(self.run_dir, "step1_data_stats.json"))

        return samples

    # ── Stage 2: Summarization ─────────────────────────────────────────

    def stage2_summarize(self, samples: List[Dict]) -> List[Dict]:
        """Generate summaries for all samples."""
        logger.info("=" * 60)
        logger.info("STAGE 2: Summarization")
        logger.info("=" * 60)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        summarizer = ChunkedSummarizer(
            model_name=self.config.summarizer_model,
            chunk_size=self.config.summarizer_chunk_size,
            chunk_overlap=self.config.summarizer_chunk_overlap,
            max_summary_length=self.config.summarizer_max_summary_length,
            max_chunk_summary_length=self.config.summarizer_max_chunk_summary_length,
            num_beams=self.config.summarizer_num_beams,
            device=device,
        )

        results = summarizer.summarize_batch(samples)

        if self.config.output_save_intermediate:
            save_json(results, os.path.join(self.run_dir, "step2_summaries.json"))

        # Free summarizer memory before loading next model
        del summarizer
        return results

    # ── Stage 3: Consistency Checking ──────────────────────────────────

    def stage3_consistency(self, samples: List[Dict]) -> List[Dict]:
        """Run NLI factual consistency detection."""
        logger.info("=" * 60)
        logger.info("STAGE 3: Consistency Checking")
        logger.info("=" * 60)
        logger.info(f"Evidence mode: {self.config.consistency_evidence_mode}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        splitter = SentenceSplitter()

        retriever = create_retriever(
            mode=self.config.consistency_evidence_mode,
            window_size=self.config.consistency_sentence_window,
            model_name=self.config.semantic_retrieval_model,
            batch_size=self.config.semantic_retrieval_batch_size,
            device=device,
        )

        checker = NLIChecker(
            model_name=self.config.consistency_nli_model,
            entailment_threshold=self.config.consistency_entailment_threshold,
            device=device,
        )

        results = checker.check_batch(samples, splitter, retriever)

        # Log summary stats
        total_s = sum(r.get("consistency", {}).get("n_total", 0) for r in results)
        total_c = sum(r.get("consistency", {}).get("n_consistent", 0) for r in results)
        logger.info(f"Overall consistency: {total_c}/{total_s} "
                     f"({total_c/total_s:.2%})" if total_s > 0 else "N/A")

        if self.config.output_save_intermediate:
            save_json(results, os.path.join(self.run_dir, "step3_consistency.json"))

        del checker, retriever, splitter
        return results

    # ── Stage 4: Error Correction ─────────────────────────────────────

    def stage4_correct(self, samples: List[Dict]) -> List[Dict]:
        """Correct inconsistent summary sentences."""
        logger.info("=" * 60)
        logger.info("STAGE 4: Error Correction")
        logger.info("=" * 60)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        corrector = LocalEditCorrector(
            model_name=self.config.corrector_model,
            max_new_tokens=self.config.corrector_max_new_tokens,
            temperature=self.config.corrector_temperature,
            num_beams=self.config.corrector_num_beams,
            max_length_ratio=self.config.corrector_max_length_ratio,
            device=device,
        )

        results = corrector.correct_batch(samples)

        # Log stats
        total_att = sum(r.get("correction", {}).get("n_attempted", 0) for r in results)
        total_succ = sum(r.get("correction", {}).get("n_succeeded", 0) for r in results)
        logger.info(f"Corrections: {total_succ}/{total_att} succeeded")

        if self.config.output_save_intermediate:
            save_json(results, os.path.join(self.run_dir, "step4_corrections.json"))

        del corrector
        return results

    # ── Stage 5: Evaluation ───────────────────────────────────────────

    def stage5_evaluate(self, samples: List[Dict]) -> Dict:
        """Compute ROUGE and before/after comparison."""
        logger.info("=" * 60)
        logger.info("STAGE 5: Evaluation")
        logger.info("=" * 60)

        rouge_eval = RougeEvaluator(
            rouge_types=self.config.evaluation_rouge_types,
            use_stemmer=self.config.evaluation_use_stemmer,
        )

        comparison = compare_before_after(samples, rouge_eval)

        if comparison.get("original_rouge"):
            r1 = comparison["original_rouge"]["rouge1"]["mean"]
            r2 = comparison["original_rouge"]["rouge2"]["mean"]
            rL = comparison["original_rouge"]["rougeL"]["mean"]
            logger.info(f"ROUGE-1: {r1:.4f}  ROUGE-2: {r2:.4f}  ROUGE-L: {rL:.4f}")

        logger.info(f"Consistency rate: {comparison['consistency']['overall_consistency_rate']:.2%}")
        logger.info(f"Correction success: {comparison['correction']['success_rate']:.2%}")

        if self.config.output_save_intermediate:
            save_json(comparison, os.path.join(self.run_dir, "step5_comparison.json"))

        return comparison

    # ── Stage 6: Analysis ─────────────────────────────────────────────

    def stage6_analyze(self, samples: List[Dict]) -> Dict:
        """Run analysis tasks: length impact + case study."""
        logger.info("=" * 60)
        logger.info("STAGE 6: Analysis")
        logger.info("=" * 60)

        rouge_eval = RougeEvaluator(
            rouge_types=self.config.evaluation_rouge_types,
            use_stemmer=self.config.evaluation_use_stemmer,
        )

        # Task 1: Length impact
        logger.info("Analyzing length impact on error rate...")
        bins = [(low, high) for low, high in self.config.analysis_length_bins]
        length_analysis = analyze_length_impact(samples, bins, rouge_eval)

        if self.config.output_save_intermediate:
            save_json(length_analysis, os.path.join(self.run_dir, "step6_length_impact.json"))

        # Plot
        plot_path = os.path.join(self.run_dir, "step6_length_impact.png")
        plot_length_impact(length_analysis, plot_path)

        # Task 2: Case study
        logger.info("Collecting correction cases...")
        cases = collect_correction_cases(samples, num_cases=self.config.analysis_num_case_studies)
        print_case_report(cases)

        if self.config.output_save_intermediate:
            save_json(cases, os.path.join(self.run_dir, "step6_cases.json"))

        return {"length_impact": length_analysis, "cases": cases}

    # ── Full Pipeline ─────────────────────────────────────────────────

    def run(self) -> Dict:
        """Run all stages in sequence. Returns the final combined result."""
        start_time = time.time()

        self._setup_run_dir()
        setup_logging(self.config.log_level, os.path.join(self.run_dir, "pipeline.log"))
        logger = logging.getLogger(__name__)
        logger.info(f"Pipeline starting. Mode: {self.config.mode}, "
                     f"Samples: {self.config.num_samples}")
        logger.info(f"Results directory: {self.run_dir}")

        try:
            samples = self.stage1_load_data()          # 1: Data
            samples = self.stage2_summarize(samples)   # 2: Summarize
            samples = self.stage3_consistency(samples) # 3: Consistency
            samples = self.stage4_correct(samples)     # 4: Correction
            comparison = self.stage5_evaluate(samples) # 5: Evaluate
            analysis = self.stage6_analyze(samples)    # 6: Analyze

            elapsed = time.time() - start_time
            logger.info(f"Pipeline completed in {elapsed:.1f}s "
                         f"({elapsed/60:.1f} min)")

            result = {
                "mode": self.config.mode,
                "num_samples": len(samples),
                "elapsed_seconds": round(elapsed, 1),
                "evaluation": comparison,
                "analysis": analysis,
            }

            save_json(result, os.path.join(self.run_dir, "pipeline_result.json"))
            logger.info(f"Final result saved to {self.run_dir}/pipeline_result.json")

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def run_from_stage(self, start_stage: int) -> Dict:
        """Resume from a specific stage by loading intermediate results.

        Args:
            start_stage: Stage number to resume from (1-6).
        """
        self._setup_run_dir()
        setup_logging(self.config.log_level, os.path.join(self.run_dir, "pipeline.log"))
        logger = logging.getLogger(__name__)
        logger.info(f"Resuming from stage {start_stage}")

        # Load the output of the previous stage
        stage_files = {
            2: "step1_data_stats.json",  # we need to re-load data
            3: "step2_summaries.json",
            4: "step3_consistency.json",
            5: "step4_corrections.json",
            6: "step4_corrections.json",  # analysis needs full samples
        }

        if start_stage == 1:
            return self.run()

        if start_stage == 2:
            samples = self.stage1_load_data()
            samples = self.stage2_summarize(samples)
        elif start_stage >= 3:
            # Load from intermediate JSON
            file_to_load = stage_files.get(start_stage, "step2_summaries.json")
            path = os.path.join(self.run_dir, file_to_load)
            samples = load_json(path)
            if samples is None:
                raise FileNotFoundError(
                    f"Cannot resume: {path} not found. Run earlier stages first."
                )
            logger.info(f"Loaded {len(samples)} samples from {path}")

        if start_stage <= 3:
            samples = self.stage3_consistency(samples)
        if start_stage <= 4:
            samples = self.stage4_correct(samples)
        if start_stage <= 5:
            comparison = self.stage5_evaluate(samples)
        if start_stage <= 6:
            analysis = self.stage6_analyze(samples)

        return {"mode": self.config.mode, "num_samples": len(samples)}
