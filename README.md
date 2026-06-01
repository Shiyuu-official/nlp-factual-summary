# NLP Factual Summary — 长文摘要事实一致性评测与纠错

NJU NLP Course 2026 项目。对 GovReport 长文摘要进行句子级事实一致性检测与自动纠错。

## 成员

- 刘可宜 (组长) 241880087
- 康乔其 241880088
- 曹家雄 241880474

## 环境

```bash
conda activate nlp-summary   # Python 3.13
pip install -r requirements.txt
```

## 项目结构

```
├── main.py                     # 入口
├── config.yaml                 # test/full 双模式配置
├── src/
│   ├── pipeline.py             # 中央调度器
│   ├── data/loader.py          # GovReport 数据加载
│   ├── summarization/summarizer.py    # Qwen2.5 分块摘要
│   ├── consistency/
│   │   ├── sentence_splitter.py       # nltk 句子切分
│   │   ├── evidence_retrieval.py      # 词重叠 / 语义检索
│   │   └── nli_checker.py             # BART-MNLI 蕴含判断
│   ├── correction/corrector.py        # 局部改写纠错
│   ├── evaluation/rouge.py + metrics.py
│   ├── analysis/length_impact.py + case_study.py
│   └── utils/config.py, io.py, logging.py
├── results/                    # 时间戳子目录 (gitignored)
└── docs/                       # 提案 + 作业要求
```

## 快速开始

```bash
# 开发验证 (2 样本，每样本约 5-10 分钟 CPU)
python main.py --mode test

# 完整实验 (500 样本)
python main.py --mode full

# 进阶任务 — 语义证据检索
# 编辑 config.yaml full 部分:
#   consistency.evidence_mode: semantic
#   semantic_retrieval.enabled: true
python main.py --mode full
```

## Pipeline 流程

```
Data Load → Summarize → NLI Consistency Check → Error Correction → Evaluate → Analyze
  Stage 1     Stage 2          Stage 3                Stage 4        Stage 5    Stage 6
```

每阶段结果保存到 `results/YYYY-MM-DD_HHMMSS/stepN_*.json`。

## 关键设计决策

- **CPU only**: 无 GPU，使用 Qwen2.5-1.5B-Instruct + 分块策略
- **HF 镜像**: 默认使用 hf-mirror.com 下载模型（国内 SSL 问题）
- **懒加载模型**: 每个 Stage 独立加载/释放，避免内存溢出
- **证据检索策略模式**: `word_overlap`（基础）/ `semantic`（进阶，sentence-transformers）
- **纠错严格校验**: 拒绝输出为空、无变化、长度超过 2 倍原文、多句输出的"纠错"
- **sample_id 对齐**: 所有阶段通过 sample_id 关联，不用 zip
