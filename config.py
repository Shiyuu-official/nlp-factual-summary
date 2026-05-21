"""
实验配置文件
集中管理所有实验参数
"""

# 数据配置
DATA_CONFIG = {
    'dataset_name': 'ccdv/govreport-summarization',
    'cache_dir': './data_cache',
    'split': 'validation',
    'num_samples': 500,  # 建议范围: 500-973
}

# 模型配置
MODEL_CONFIG = {
    # 摘要模型
    'summarizer': {
        'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
        'chunk_size': 2000,
        'chunk_overlap': 200,
        'max_summary_length': 300,
    },

    # 一致性检测模型
    'consistency_checker': {
        'model_name': 'facebook/bart-large-mnli',
        'sentence_window': 3,
        'entailment_threshold': 0.5,
    },

    # 纠错模型
    'error_corrector': {
        'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
    }
}

# 实验配置
EXPERIMENT_CONFIG = {
    'output_dir': './results',
    'log_level': 'INFO',
    'save_intermediate_results': True,
}

# 分析任务配置
ANALYSIS_CONFIG = {
    'length_intervals': [
        (0, 100),
        (100, 200),
        (200, 300),
        (300, 400),
        (400, 500),
        (500, float('inf'))
    ],
    'num_analysis_cases': 10,
}
