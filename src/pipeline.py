"""Pipeline orchestrator: wires all stages together.

Stages are independent and can be run individually for partial re-runs.
Each stage appends fields to a list of sample dicts that flow through the pipeline.
"""

import logging
import os
import time
import torch
from typing import List, Dict, Optional
from tqdm import tqdm

from .utils.config import PipelineConfig
from .utils.io import ensure_dir, save_json, load_json, timestamped_dir
from .utils.logging import setup_logging
from .data.loader import GovReportDataLoader
from .summarization.summarizer import ChunkedSummarizer
from .consistency.sentence_splitter import SentenceSplitter
from .consistency.evidence_retrieval import create_retriever
from .consistency.nli_checker import NLIChecker
from .correction.corrector import EvidenceConstrainedCorrector
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

    def __init__(self, config: PipelineConfig, run_dir: Optional[str] = None):
        self.config = config
        self.run_dir = run_dir
        self.logger = logging.getLogger(__name__)

    def _setup_run_dir(self, reuse_existing: bool = False) -> str:
        """Create or reuse an output directory and save config snapshot."""
        if self.run_dir:
            if reuse_existing and not os.path.isdir(self.run_dir):
                raise FileNotFoundError(f"Run directory not found: {self.run_dir}")
            self.run_dir = ensure_dir(self.run_dir)
        else:
            self.run_dir = timestamped_dir(self.config.output_root_dir)

        save_json(
            {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None)))
             else v for k, v in vars(self.config).items()},
            os.path.join(self.run_dir, "config_applied.json"),
        )
        return self.run_dir

    def _stage_path(self, filename: str) -> str:
        return os.path.join(self.run_dir, filename)

    def _load_stage_results(self, final_name: str, partial_name: str) -> List[Dict]:
        """Load final stage output if present, otherwise partial checkpoint."""
        final = load_json(self._stage_path(final_name))
        if final is not None:
            logger.info(f"Loaded completed stage output: {final_name}")
            return final

        partial = load_json(self._stage_path(partial_name))
        if partial is not None:
            logger.info(f"Loaded partial stage checkpoint: {partial_name}")
            return partial

        return []

    def _results_cover_samples(self, results: List[Dict], samples: List[Dict]) -> bool:
        """Return True when checkpoint/final results cover every input sample."""
        result_ids = {item.get("sample_id") for item in results}
        sample_ids = {item.get("sample_id") for item in samples}
        return bool(sample_ids) and sample_ids.issubset(result_ids)

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
        partial_path = self._stage_path("step2_summaries.partial.json")
        existing = self._load_stage_results(
            "step2_summaries.json",
            "step2_summaries.partial.json",
        )
        if self._results_cover_samples(existing, samples):
            logger.info("Stage 2 already complete; skipping summarization model load")
            if self.config.output_save_intermediate:
                save_json(existing, self._stage_path("step2_summaries.json"))
            return existing

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

        results = summarizer.summarize_batch(
            samples,
            checkpoint_path=partial_path,
            existing_results=existing,
        )

        if self.config.output_save_intermediate:
            save_json(results, self._stage_path("step2_summaries.json"))

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
        partial_path = self._stage_path("step3_consistency.partial.json")
        existing = self._load_stage_results(
            "step3_consistency.json",
            "step3_consistency.partial.json",
        )
        if self._results_cover_samples(existing, samples):
            logger.info("Stage 3 already complete; skipping NLI model load")
            if self.config.output_save_intermediate:
                save_json(existing, self._stage_path("step3_consistency.json"))
            return existing

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
            evidence_top_k=self.config.consistency_evidence_top_k,
            device=device,
        )

        results = checker.check_batch(
            samples,
            splitter,
            retriever,
            checkpoint_path=partial_path,
            existing_results=existing,
        )

        # Log summary stats
        total_s = sum(r.get("consistency", {}).get("n_total", 0) for r in results)
        total_c = sum(r.get("consistency", {}).get("n_consistent", 0) for r in results)
        logger.info(f"Overall consistency: {total_c}/{total_s} "
                     f"({total_c/total_s:.2%})" if total_s > 0 else "N/A")

        if self.config.output_save_intermediate:
            save_json(results, self._stage_path("step3_consistency.json"))

        del checker, retriever, splitter
        return results

    # ── Stage 4: Error Correction ─────────────────────────────────────

    def stage4_correct(self, samples: List[Dict]) -> List[Dict]:
        """Correct inconsistent summary sentences."""
        logger.info("=" * 60)
        logger.info("STAGE 4: Error Correction")
        logger.info("=" * 60)
        partial_path = self._stage_path("step4_corrections.partial.json")
        existing = self._load_stage_results(
            "step4_corrections.json",
            "step4_corrections.partial.json",
        )
        if self._results_cover_samples(existing, samples):
            logger.info("Stage 4 already complete; skipping correction model load")
            if self.config.output_save_intermediate:
                save_json(existing, self._stage_path("step4_corrections.json"))
            return existing

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        corrector = EvidenceConstrainedCorrector(
            model_name=self.config.corrector_model,
            nli_model_name=self.config.corrector_nli_model,
            max_new_tokens=self.config.corrector_max_new_tokens,
            num_candidates=self.config.corrector_num_candidates,
            sample_temperature=self.config.corrector_sample_temperature,
            entailment_threshold=self.config.consistency_entailment_threshold,
            max_refinement_rounds=self.config.corrector_max_refinement_rounds,
            max_length_ratio=self.config.corrector_max_length_ratio,
            enable_refinement=self.config.corrector_enable_refinement,
            enable_extractive_fallback=self.config.corrector_enable_extractive_fallback,
            device=device,
        )

        results = corrector.correct_batch(
            samples,
            checkpoint_path=partial_path,
            existing_results=existing,
        )
        del corrector
        if device == "cuda":
            torch.cuda.empty_cache()

        results = self._verify_corrections(
            results,
            device=device,
            checkpoint_path=partial_path,
        )

        # Log stats
        total_att = sum(r.get("correction", {}).get("n_attempted", 0) for r in results)
        total_succ = sum(r.get("correction", {}).get("n_succeeded", 0) for r in results)
        total_verified = sum(r.get("correction", {}).get("n_verified", 0) for r in results)
        # Strategy breakdown
        strategy_counts = {"candidate": 0, "refinement": 0, "extractive": 0, "none": 0}
        for r in results:
            for c in r.get("correction", {}).get("corrections", []):
                s = c.get("strategy", "none")
                strategy_counts[s] = strategy_counts.get(s, 0) + 1
        logger.info(
            f"Corrections: {total_succ}/{total_att} format-valid, "
            f"{total_verified}/{total_succ} NLI-verified"
        )
        logger.info(
            f"Strategy breakdown: candidate={strategy_counts['candidate']}, "
            f"refinement={strategy_counts['refinement']}, "
            f"extractive={strategy_counts['extractive']}, "
            f"none={strategy_counts['none']}"
        )

        if self.config.output_save_intermediate:
            save_json(results, self._stage_path("step4_corrections.json"))

        return results

    def _verify_corrections(self, samples: List[Dict], device: str,
                            checkpoint_path: Optional[str] = None) -> List[Dict]:
        """Re-check accepted corrections with NLI against their selected evidence."""
        total_success = sum(
            1
            for sample in samples
            for corr in sample.get("correction", {}).get("corrections", [])
            if corr.get("success") and "verification" not in corr
        )
        if total_success == 0:
            return samples

        logger.info("Verifying corrected sentences with NLI...")
        checker = NLIChecker(
            model_name=self.config.consistency_nli_model,
            entailment_threshold=self.config.consistency_entailment_threshold,
            evidence_top_k=self.config.consistency_evidence_top_k,
            device=device,
        )

        for sample in tqdm(samples, desc="Verifying corrections", unit="sample"):
            correction = sample.get("correction", {})
            n_verified = 0
            n_improved = 0
            n_fixed = 0

            for corr in correction.get("corrections", []):
                if not corr.get("success"):
                    continue
                if "verification" in corr:
                    n_verified += int(corr["verification"].get("verified", False))
                    n_improved += int(corr["verification"].get("improved", False))
                    n_fixed += int(corr["verification"].get("fixed", False))
                    continue

                evidence = corr.get("evidence_used", "")
                original = corr.get("original", "")
                corrected = corr.get("corrected", "")
                if not evidence or not original or not corrected:
                    corr["verification"] = {
                        "verified": False,
                        "failure_reason": "missing_evidence_or_sentence",
                    }
                    continue

                try:
                    before = checker.check_pair(evidence, original)
                    after = checker.check_pair(evidence, corrected)
                    before_score = before["entailment_score"]
                    after_score = after["entailment_score"]
                    improved = after_score > before_score
                    fixed = (not before["is_consistent"]) and after["is_consistent"]
                    verified = after["is_consistent"] and improved

                    corr["verification"] = {
                        "verified": verified,
                        "improved": improved,
                        "fixed": fixed,
                        "original_entailment_score": before_score,
                        "corrected_entailment_score": after_score,
                        "original_label": before["label"],
                        "corrected_label": after["label"],
                    }

                    n_verified += int(verified)
                    n_improved += int(improved)
                    n_fixed += int(fixed)
                except Exception as e:
                    logger.warning(
                        f"Correction verification failed for sample "
                        f"{sample.get('sample_id')}: {e}"
                    )
                    corr["verification"] = {
                        "verified": False,
                        "failure_reason": str(e),
                    }

            correction["n_verified"] = n_verified
            correction["n_improved"] = n_improved
            correction["n_fixed"] = n_fixed

            if checkpoint_path:
                save_json(samples, checkpoint_path)

        del checker
        if device == "cuda":
            torch.cuda.empty_cache()
        return samples

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
        logger.info(
            f"Correction format success: "
            f"{comparison['correction']['format_success_rate']:.2%}"
        )
        logger.info(
            f"Correction NLI verified: "
            f"{comparison['correction']['nli_verified_rate']:.2%}"
        )

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

        self._setup_run_dir(reuse_existing=False)
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
        if start_stage >= 3 and not self.run_dir:
            raise ValueError(
                "--run-dir is required when resuming from stage 3 or later. "
                "Example: python main.py --stage 4 --run-dir results/2026-06-02_195434"
            )

        self._setup_run_dir(reuse_existing=bool(self.run_dir))
        setup_logging(self.config.log_level, os.path.join(self.run_dir, "pipeline.log"))
        logger = logging.getLogger(__name__)
        logger.info(f"Resuming from stage {start_stage}")

        # Load the output of the previous stage from the selected run directory.
        stage_files = {
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
