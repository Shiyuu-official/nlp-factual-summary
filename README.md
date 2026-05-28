# 长文摘要事实一致性评测与纠错

NJU NLP 课程项目 2026 · 第4组

## 环境配置

```bash
conda create -n nlp-summary python=3.13
conda activate nlp-summary
pip install -r requirements.txt
```

## 运行

### 第一步：生成基线摘要

```bash
python src/baseline.py
```

- 数据集：`ccdv/govreport-summarization`（GovReport）
- 模型：`facebook/bart-large-cnn`（分块摘要策略，1024 token/块，100 token重叠）
- 输出：`outputs/baseline_YYYYMMDD_HHMMSS.json`

### 第二步：事实一致性检测

```bash
python src/consistency.py --input outputs/baseline_*.json
```

- 模型：`facebook/bart-large-mnli`（句子级 NLI）
- 对每个生成句判断 entailment / neutral / contradiction
- 输出：`outputs/consistency_YYYYMMDD_HHMMSS.json`

## 输出格式

### baseline JSON

```json
[
  {
    "id": "0",
    "report": "原始报告全文...",
    "reference": "参考摘要...",
    "reference_sentences": ["句1", "句2"],
    "generated": "生成摘要...",
    "generated_sentences": ["句1", "句2"],
    "num_chunks": 12,
    "rouge1": 0.1087,
    "rouge2": 0.0280,
    "rougeL": 0.0714
  }
]
```

### consistency JSON

在上方基础上增加：

```json
{
  "nli_results": [
    {"label": "entailment", "score": 0.98, "probs": {...}},
    {"label": "neutral", "score": 0.72, "probs": {...}}
  ],
  "nli_inconsistent_count": 1,
  "nli_total_sentences": 3
}
```

## 项目结构

```
├── src/
│   ├── baseline.py       # 摘要生成基线
│   └── consistency.py    # NLI 事实一致性检测
├── outputs/              # 实验结果（gitignored）
├── docs/proposal/        # 项目提案
├── requirements.txt
└── README.md
```
