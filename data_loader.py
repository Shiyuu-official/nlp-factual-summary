"""
数据加载和预处理模块
负责加载 GovReport 数据集并进行必要的预处理
"""

from datasets import load_dataset
import nltk
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GovReportDataLoader:
    """GovReport 数据集加载器"""

    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = cache_dir
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    def load_data(self, split: str = "validation", max_samples: int = None) -> List[Dict]:
        """
        加载 GovReport 数据集

        Args:
            split: 数据集分割 ('train', 'validation', 'test')
            max_samples: 最大样本数，None 表示加载全部

        Returns:
            数据列表，每个元素包含 report 和 summary
        """
        logger.info(f"正在加载 GovReport 数据集 ({split} split)...")

        dataset = load_dataset(
            "ccdv/govreport-summarization",
            split=split,
            cache_dir=self.cache_dir
        )

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        data = []
        for idx, item in enumerate(dataset):
            data.append({
                'report': item['report'],
                'summary': item['summary'],
                'id': idx
            })

        logger.info(f"成功加载 {len(data)} 个样本")
        return data

    def get_statistics(self, data: List[Dict]) -> Dict:
        """获取数据统计信息"""
        report_lengths = [len(item['report'].split()) for item in data]
        summary_lengths = [len(item['summary'].split()) for item in data]

        return {
            'num_samples': len(data),
            'avg_report_length': sum(report_lengths) / len(report_lengths),
            'max_report_length': max(report_lengths),
            'min_report_length': min(report_lengths),
            'avg_summary_length': sum(summary_lengths) / len(summary_lengths),
            'max_summary_length': max(summary_lengths),
            'min_summary_length': min(summary_lengths),
        }
