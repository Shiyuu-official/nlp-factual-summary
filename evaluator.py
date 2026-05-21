"""
评估模块
计算 ROUGE 分数和其他指标
"""

from rouge_score import rouge_scorer
from typing import List, Dict
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """评估器 - 计算 ROUGE 和事实一致性指标"""

    def __init__(self, rouge_types: List[str] = None):
        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL']

        self.rouge_types = rouge_types
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    def calculate_rouge(
        self,
        reference: str,
        hypothesis: str
    ) -> Dict:
        """
        计算单个样本的 ROUGE 分数

        Args:
            reference: 参考摘要
            hypothesis: 生成摘要

        Returns:
            ROUGE 分数
        """
        if not reference or not hypothesis:
            return {rouge_type: {'fmeasure': 0.0} for rouge_type in self.rouge_types}

        scores = self.scorer.score(reference, hypothesis)
        return scores

    def calculate_batch_rouge(
        self,
        samples: List[Dict]
    ) -> Dict:
        """
        批量计算 ROUGE 分数

        Args:
            samples: 样本列表，包含 'reference_summary' 和 'generated_summary'

        Returns:
            平均 ROUGE 分数
        """
        all_scores = {rouge_type: [] for rouge_type in self.rouge_types}

        for sample in samples:
            ref = sample.get('reference_summary', '')
            hyp = sample.get('generated_summary', '')

            if not hyp:
                continue

            scores = self.calculate_rouge(ref, hyp)

            for rouge_type in self.rouge_types:
                all_scores[rouge_type].append(scores[rouge_type]['fmeasure'])

        avg_scores = {}
        for rouge_type in self.rouge_types:
            if all_scores[rouge_type]:
                avg_scores[rouge_type] = {
                    'mean': np.mean(all_scores[rouge_type]),
                    'std': np.std(all_scores[rouge_type]),
                    'values': all_scores[rouge_type]
                }
            else:
                avg_scores[rouge_type] = {'mean': 0.0, 'std': 0.0, 'values': []}

        return avg_scores

    def analyze_consistency_results(
        self,
        consistency_checks: List[Dict]
    ) -> Dict:
        """
        分析一致性检测结果

        Args:
            consistency_checks: 一致性检测结果列表

        Returns:
            统计分析结果
        """
        total_sentences = 0
        consistent_sentences = 0
        inconsistent_sentences = 0

        error_rates = []

        for check in consistency_checks:
            if 'consistency_result' not in check:
                continue

            result = check['consistency_result']
            total_sentences += result['total_sentences']
            consistent_sentences += result['consistent_sentences']
            inconsistent_sentences += result['inconsistent_sentences']

            if result['total_sentences'] > 0:
                error_rate = 1 - result['consistency_rate']
                error_rates.append(error_rate)

        overall_consistency_rate = (
            consistent_sentences / total_sentences if total_sentences > 0 else 0
        )

        return {
            'total_sentences': total_sentences,
            'consistent_sentences': consistent_sentences,
            'inconsistent_sentences': inconsistent_sentences,
            'overall_consistency_rate': overall_consistency_rate,
            'average_error_rate': np.mean(error_rates) if error_rates else 0,
            'std_error_rate': np.std(error_rates) if error_rates else 0
        }

    def compare_before_after_correction(
        self,
        original_samples: List[Dict],
        corrected_samples: List[Dict]
    ) -> Dict:
        """
        比较纠错前后的变化

        Args:
            original_samples: 纠错前的样本
            corrected_samples: 纠错后的样本

        Returns:
            对比结果
        """
        original_rouge = self.calculate_batch_rouge(original_samples)

        corrected_for_rouge = []
        for orig, corr in zip(original_samples, corrected_samples):
            if 'corrected_summary' in corr:
                corrected_for_rouge.append({
                    'reference_summary': orig.get('reference_summary', ''),
                    'generated_summary': corr['corrected_summary']
                })

        corrected_rouge = self.calculate_batch_rouge(corrected_for_rouge)

        total_corrections = 0
        successful_corrections = 0

        for corr in corrected_samples:
            if 'corrections' in corr:
                total_corrections += corr.get('total_errors', 0)
                successful_corrections += corr.get('num_corrected', 0)

        return {
            'original_rouge': original_rouge,
            'corrected_rouge': corrected_rouge,
            'total_errors_detected': total_corrections,
            'successful_corrections': successful_corrections,
            'correction_success_rate': (
                successful_corrections / total_corrections if total_corrections > 0 else 0
            )
        }
