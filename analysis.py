"""
分析任务脚本
任务1: 分析摘要长度对事实错误率的影响
任务2: 分析纠错前后的案例
"""

import json
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskAnalyzer:
    """分析任务执行器"""

    def __init__(self, results_dir: str = "./results/simple_task"):
        self.results_dir = results_dir

        # 加载结果
        self.summaries = self._load_json(os.path.join(results_dir, 'summaries.json'))
        self.consistency_checks = self._load_json(os.path.join(results_dir, 'consistency_checks.json'))
        self.correction_results = self._load_json(os.path.join(results_dir, 'correction_results.json'))

    def _load_json(self, filepath: str) -> List[Dict]:
        """加载 JSON 文件"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"文件不存在: {filepath}")
            return []

    def analyze_length_impact(self, length_intervals: List[Tuple] = None):
        """
        分析摘要长度对事实错误率的影响

        Args:
            length_intervals: 长度区间列表，如 [(0, 100), (100, 200), ...]
        """
        if length_intervals is None:
            length_intervals = [
                (0, 100),
                (100, 200),
                (200, 300),
                (300, 400),
                (400, 500),
                (500, float('inf'))
            ]

        logger.info("开始分析摘要长度对事实错误率的影响...")

        # 按长度分组
        length_groups = defaultdict(list)

        for check in self.consistency_checks:
            if 'consistency_result' not in check:
                continue

            summary = check.get('summary', '')
            summary_length = len(summary.split())

            # 找到对应的长度区间
            for low, high in length_intervals:
                if low <= summary_length < high:
                    interval_key = f"{low}-{int(high) if high != float('inf') else 'inf'}"
                    length_groups[interval_key].append(check)
                    break

        # 分析每个区间的结果
        analysis_results = {}

        for interval, checks in length_groups.items():
            if not checks:
                continue

            # 计算一致性统计
            total_sentences = sum(c['consistency_result']['total_sentences'] for c in checks)
            consistent_sentences = sum(c['consistency_result']['consistent_sentences'] for c in checks)
            inconsistent_sentences = sum(c['consistency_result']['inconsistent_sentences'] for c in checks)

            consistency_rate = consistent_sentences / total_sentences if total_sentences > 0 else 0
            error_rate = 1 - consistency_rate

            avg_lengths = [len(c.get('summary', '').split()) for c in checks]

            analysis_results[interval] = {
                'num_samples': len(checks),
                'avg_summary_length': np.mean(avg_lengths),
                'total_sentences': total_sentences,
                'consistent_sentences': consistent_sentences,
                'inconsistent_sentences': inconsistent_sentences,
                'consistency_rate': consistency_rate,
                'error_rate': error_rate
            }

        # 保存结果
        output_file = os.path.join(self.results_dir, 'length_impact_analysis.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        logger.info(f"长度影响分析结果已保存到 {output_file}")

        # 打印结果
        print("\n" + "="*80)
        print("摘要长度对事实错误率的影响分析")
        print("="*80)
        print(f"{'长度区间':<15} {'样本数':<10} {'平均长度':<12} {'一致率':<12} {'错误率':<12}")
        print("-"*80)

        for interval in sorted(analysis_results.keys()):
            data = analysis_results[interval]
            print(f"{interval:<15} {data['num_samples']:<10} {data['avg_summary_length']:<12.1f} "
                  f"{data['consistency_rate']:<12.2%} {data['error_rate']:<12.2%}")

        # 可视化
        self._plot_length_impact(analysis_results)

        return analysis_results

    def _plot_length_impact(self, analysis_results: Dict):
        """可视化长度对错误率的影响"""
        intervals = sorted(analysis_results.keys())
        error_rates = [analysis_results[i]['error_rate'] * 100 for i in intervals]
        consistency_rates = [analysis_results[i]['consistency_rate'] * 100 for i in intervals]

        x = range(len(intervals))

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        bars1 = plt.bar(x, error_rates, color='salmon', alpha=0.7, edgecolor='black')
        plt.xlabel('摘要长度区间', fontsize=12)
        plt.ylabel('错误率 (%)', fontsize=12)
        plt.title('摘要长度对事实错误率的影响', fontsize=14, fontweight='bold')
        plt.xticks(x, intervals, rotation=45)
        plt.grid(axis='y', alpha=0.3)

        # 在柱状图上添加数值标签
        for bar, rate in zip(bars1, error_rates):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.subplot(1, 2, 2)
        bars2 = plt.bar(x, consistency_rates, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('摘要长度区间', fontsize=12)
        plt.ylabel('一致率 (%)', fontsize=12)
        plt.title('摘要长度对事实一致性的影响', fontsize=14, fontweight='bold')
        plt.xticks(x, intervals, rotation=45)
        plt.grid(axis='y', alpha=0.3)

        # 在柱状图上添加数值标签
        for bar, rate in zip(bars2, consistency_rates):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, 'length_impact_plot.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"图表已保存到 {plot_file}")
        plt.close()

    def analyze_correction_cases(self, num_cases: int = 10):
        """
        分析纠错案例

        Args:
            num_cases: 要展示的案例数量
        """
        logger.info("开始分析纠错案例...")

        successful_cases = []
        failed_cases = []

        for result in self.correction_results:
            if 'corrections' not in result or not result['corrections']:
                continue

            for correction in result['corrections']:
                case = {
                    'id': result.get('id'),
                    'original': correction['original'],
                    'corrected': correction['corrected'],
                    'success': correction.get('success', False),
                    'evidence': correction.get('evidence_used', [])[:1]
                }

                if correction.get('success', False):
                    successful_cases.append(case)
                else:
                    failed_cases.append(case)

        # 选择案例
        selected_successful = successful_cases[:num_cases//2]
        selected_failed = failed_cases[:num_cases//2]

        analysis = {
            'total_successful': len(successful_cases),
            'total_failed': len(failed_cases),
            'selected_cases': {
                'successful': selected_successful,
                'failed': selected_failed
            }
        }

        # 保存分析结果
        output_file = os.path.join(self.results_dir, 'correction_case_analysis.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        logger.info(f"案例分析结果已保存到 {output_file}")

        # 打印示例
        self._print_case_examples(selected_successful, selected_failed)

        return analysis

    def _print_case_examples(self, successful: List[Dict], failed: List[Dict]):
        """打印案例示例"""
        print("\n" + "="*80)
        print("成功纠错案例示例:")
        print("="*80)

        for i, case in enumerate(successful[:3], 1):
            print(f"\n案例 {i}:")
            print(f"原句: {case['original']}")
            print(f"修正: {case['corrected']}")
            if case['evidence']:
                print(f"证据: {case['evidence'][0][:200]}...")

        print("\n" + "="*80)
        print("失败纠错案例示例:")
        print("="*80)

        for i, case in enumerate(failed[:3], 1):
            print(f"\n案例 {i}:")
            print(f"原句: {case['original']}")
            print(f"修正: {case['corrected']}")
            print(f"失败原因: 未能有效修正")


def main():
    """主函数"""
    analyzer = TaskAnalyzer(results_dir="./results/simple_task")

    # 任务1: 分析长度影响
    print("\n" + "="*80)
    print("任务1: 分析摘要长度对事实错误率的影响")
    print("="*80)
    length_analysis = analyzer.analyze_length_impact()

    # 任务2: 分析纠错案例
    print("\n" + "="*80)
    print("任务2: 分析纠错案例")
    print("="*80)
    case_analysis = analyzer.analyze_correction_cases(num_cases=10)

    print("\n分析任务完成！")


if __name__ == "__main__":
    main()
