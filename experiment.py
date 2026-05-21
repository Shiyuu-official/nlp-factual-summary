"""
主实验脚本
执行完整的实验流程：摘要生成 -> 一致性检测 -> 自动纠错
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List

from data_loader import GovReportDataLoader
from summarizer import LongDocumentSummarizer
from consistency_checker import FactConsistencyChecker
from error_corrector import ErrorCorrector
from evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """实验运行器"""

    def __init__(
        self,
        output_dir: str = "./results",
        num_samples: int = 500
    ):
        self.output_dir = output_dir
        self.num_samples = num_samples
        os.makedirs(output_dir, exist_ok=True)

        # 初始化各个组件
        self.data_loader = GovReportDataLoader()
        self.summarizer = LongDocumentSummarizer()
        self.consistency_checker = FactConsistencyChecker()
        self.error_corrector = ErrorCorrector()
        self.evaluator = Evaluator()

    def run_simple_task(self):
        """运行简单任务"""
        logger.info("=" * 80)
        logger.info("开始执行简单任务")
        logger.info("=" * 80)

        # Step 1: 加载数据
        logger.info("\n[Step 1/5] 加载数据...")
        data = self.data_loader.load_data(
            split="validation",
            max_samples=self.num_samples
        )

        stats = self.data_loader.get_statistics(data)
        logger.info(f"数据统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")

        # 保存数据统计
        with open(os.path.join(self.output_dir, 'data_statistics.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # Step 2: 生成摘要
        logger.info("\n[Step 2/5] 生成摘要...")
        summarized_data = self.summarizer.batch_summarize(data)

        # 保存摘要结果
        summaries_file = os.path.join(self.output_dir, 'summaries.json')
        with open(summaries_file, 'w', encoding='utf-8') as f:
            json.dump(summarized_data, f, indent=2, ensure_ascii=False)
        logger.info(f"摘要已保存到 {summaries_file}")

        # Step 3: 计算 ROUGE 分数
        logger.info("\n[Step 3/5] 计算 ROUGE 分数...")
        rouge_scores = self.evaluator.calculate_batch_rouge(summarized_data)
        logger.info(f"ROUGE 分数: {json.dumps(rouge_scores, indent=2, ensure_ascii=False)}")

        # 保存 ROUGE 分数
        with open(os.path.join(self.output_dir, 'rouge_scores.json'), 'w', encoding='utf-8') as f:
            json.dump(rouge_scores, f, indent=2, ensure_ascii=False)

        # Step 4: 事实一致性检测
        logger.info("\n[Step 4/5] 进行事实一致性检测...")
        consistency_checks = self.consistency_checker.batch_check(summarized_data)

        # 保存一致性检测结果
        consistency_file = os.path.join(self.output_dir, 'consistency_checks.json')
        with open(consistency_file, 'w', encoding='utf-8') as f:
            json.dump(consistency_checks, f, indent=2, ensure_ascii=False)
        logger.info(f"一致性检测结果已保存到 {consistency_file}")

        # 分析一致性结果
        consistency_analysis = self.evaluator.analyze_consistency_results(consistency_checks)
        logger.info(f"一致性分析: {json.dumps(consistency_analysis, indent=2, ensure_ascii=False)}")

        with open(os.path.join(self.output_dir, 'consistency_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(consistency_analysis, f, indent=2, ensure_ascii=False)

        # Step 5: 自动纠错
        logger.info("\n[Step 5/5] 执行自动纠错...")
        correction_results = self.error_corrector.batch_correct(consistency_checks)

        # 保存纠错结果
        correction_file = os.path.join(self.output_dir, 'correction_results.json')
        with open(correction_file, 'w', encoding='utf-8') as f:
            json.dump(correction_results, f, indent=2, ensure_ascii=False)
        logger.info(f"纠错结果已保存到 {correction_file}")

        # 比较纠错前后
        comparison = self.evaluator.compare_before_after_correction(
            summarized_data,
            correction_results
        )
        logger.info(f"纠错对比: {json.dumps(comparison, indent=2, ensure_ascii=False)}")

        with open(os.path.join(self.output_dir, 'correction_comparison.json'), 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        logger.info("\n" + "=" * 80)
        logger.info("简单任务完成！")
        logger.info("=" * 80)

        return {
            'summaries': summarized_data,
            'consistency_checks': consistency_checks,
            'correction_results': correction_results,
            'rouge_scores': rouge_scores,
            'consistency_analysis': consistency_analysis,
            'correction_comparison': comparison
        }

    def extract_cases_for_analysis(self, correction_results: List[Dict], num_cases: int = 10):
        """提取用于分析的案例"""
        cases = []

        for result in correction_results:
            if 'corrections' not in result or not result['corrections']:
                continue

            for correction in result['corrections']:
                if len(cases) >= num_cases * 2:
                    break

                cases.append({
                    'id': result['id'],
                    'original': correction['original'],
                    'corrected': correction['corrected'],
                    'success': correction.get('success', False),
                    'evidence': correction.get('evidence_used', [])[:1]
                })

            if len(cases) >= num_cases * 2:
                break

        # 分离成功和失败的案例
        successful = [c for c in cases if c['success']]
        failed = [c for c in cases if not c['success']]

        # 各取一半
        selected_cases = []
        selected_cases.extend(successful[:num_cases//2])
        selected_cases.extend(failed[:num_cases//2])

        # 保存到文件
        cases_file = os.path.join(self.output_dir, 'analysis_cases.json')
        with open(cases_file, 'w', encoding='utf-8') as f:
            json.dump(selected_cases, f, indent=2, ensure_ascii=False)

        logger.info(f"分析案例已保存到 {cases_file}")
        return selected_cases


def main():
    """主函数"""
    runner = ExperimentRunner(
        output_dir="./results/simple_task",
        num_samples=500
    )

    results = runner.run_simple_task()

    cases = runner.extract_cases_for_analysis(results['correction_results'])

    logger.info(f"\n实验完成！结果保存在 ./results/simple_task 目录")
    logger.info(f"提取了 {len(cases)} 个分析案例")


if __name__ == "__main__":
    main()
