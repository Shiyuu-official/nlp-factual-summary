# 实验日志

## 2026-05-28 — 基线测试

**目的**：搭建摘要生成与事实一致性检测 pipeline，小样本验证可行性。

**环境**：CPU only (Intel), conda nlp-summary, Python 3.13, transformers 4.48.0

**参数**：
- 摘要模型：`facebook/bart-large-cnn`
- 分块策略：1024 tokens/块，100 token 重叠
- NLI 模型：`facebook/bart-large-mnli`
- 样本数：2 (train split)

**结果**：

| 样本 | ROUGE-1 | ROUGE-2 | ROUGE-L | 分块数 | 句子数 | 不一致句 |
|------|---------|---------|---------|--------|--------|----------|
| 0 | 0.1087 | 0.0280 | 0.0714 | 12 | 3 | 1 (NEUTRAL) |
| 1 | 0.1608 | 0.0828 | 0.1196 | 11 | 3 | 1 (NEUTRAL) |

**生成质量**：摘要内容与原文相关，语法通顺。
- 样本0：讨论 DOD 文职人员海外部署的数据管理问题
- 样本1：讨论美国商业所得税的税收合规差距

**NLI 检测结果**：6句中共2句被标为 NEUTRAL（原文中缺乏直接证据），4句标为 entailment。无 contradiction 标签。

**结论**：Pipeline 端到端跑通。下一步扩大样本量并开发纠错模块。

---

## 技术决策记录

1. **LongT5 → BART**：`google/long-t5-local-base` 在 transformers 4.48+ 存在 embedding 权重映射缺陷（shared.weight 未正确映射至 encoder/decoder/lm_head），导致生成乱码。改用 `facebook/bart-large-cnn` + 分块策略，输出质量正常。
2. **transformers 版本**：5.x 对 T5 家族模型的 safetensors 加载存在兼容性问题，回退至 4.48.0。
3. **CPU 运行**：单条样本生成约 2-3 分钟（取决于文档长度和分块数），小样本迭代可接受。
