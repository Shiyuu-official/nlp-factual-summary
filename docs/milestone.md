# 长文摘要事实一致性评测与纠错中期报告

## 一、项目背景与目标

长文摘要任务要求模型从较长的原始文档中提取关键信息并生成简洁摘要。然而，当前大语言模型在生成摘要时仍容易产生事实错误，例如引入原文中不存在的实体、错误归纳因果关系、生成无关主题内容，或对数字和机构名称进行错误改写。传统 ROUGE 指标主要衡量生成摘要与参考摘要之间的词面重叠，难以充分反映摘要是否忠实于原文。

本项目围绕 GovReport Dataset 上的长文摘要事实一致性问题，构建“摘要生成 + 事实一致性检测 + 自动纠错 + 实验分析”的完整流程。项目目标包括：

1. 使用长文摘要模型在 GovReport 数据集上生成摘要。
2. 对生成摘要进行句子级事实一致性检测。
3. 对检测出的风险句进行基于证据的局部纠错。
4. 分析摘要长度、事实一致性和纠错效果之间的关系。
5. 为后续 500 篇以上正式实验打好基础。

## 二、当前完成情况

截至中期阶段，项目已经完成端到端实验流程的搭建，并在小规模样本上完成了多轮测试和优化。

目前已经实现的模块包括：

- GovReport 数据加载模块。
- 基于 Qwen/Qwen2.5-1.5B-Instruct 的分块摘要生成模块。
- 句子切分与证据检索模块。
- 基于 facebook/bart-large-mnli 的 NLI 事实一致性检测模块。
- 基于 Qwen 的局部纠错模块。
- ROUGE 评估与纠错前后对比模块。
- 摘要长度影响分析模块。
- 成功与失败纠错案例收集模块。

同时，项目已经完成服务器部署，能够在 Tesla T4 GPU 上运行。此前本地环境只能使用 CPU，模型推理速度较慢；迁移到服务器后，测试流程运行时间明显降低。

## 三、方法设计

### 3.1 长文摘要生成

由于 GovReport 原文较长，直接输入模型容易超过上下文长度限制。因此当前系统采用分块摘要策略：

1. 将原始报告按固定长度切分为多个重叠片段。
2. 对每个片段分别生成局部摘要。
3. 将多个局部摘要合并。
4. 如果合并后的摘要过长，再进行二次摘要融合。

当前使用的摘要模型为：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

在中期实验中，我们发现 Qwen Instruct 在普通 prompt 下容易产生泛化文本，例如生成与原文无关的主题、Markdown 标题、编号列表、对话模板残留等。因此我们对摘要 prompt 和后处理进行了多轮优化，包括：

- 使用 Qwen chat template 构造输入。
- 明确要求只生成原文支持的事实。
- 禁止 Markdown、编号、列表和建议式输出。
- 禁止引入原文不存在的国家、组织、人物、事件或主题。
- 对生成摘要进行清洗，去除明显跑题内容和对话模板残留。

### 3.2 事实一致性检测

事实一致性检测采用句子级流程：

1. 将生成摘要切分为句子。
2. 对每个摘要句，从原文中检索相关证据片段。
3. 使用 NLI 模型判断该摘要句是否能被证据支持。
4. 若摘要句的最大 entailment 分数低于阈值，则判定为事实风险句。

当前 NLI 模型为：

```text
facebook/bart-large-mnli
```

中期阶段修复了一个关键问题：原代码错误假设 NLI 标签顺序为 `entailment, neutral, contradiction`，但 BART-MNLI 实际标签顺序并不应硬编码。现在系统会从模型配置中自动读取标签映射，从而避免将 contradiction 分数误当作 entailment 分数。

### 3.3 自动纠错

自动纠错模块针对 NLI 检测出的风险句进行局部改写。纠错时输入包括：

- 原始风险句。
- 从原文中检索到的证据片段。

纠错模型同样使用：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

在早期测试中，纠错模型经常输出过长内容，甚至复制证据片段，导致 59 次纠错全部失败。随后我们做了以下优化：

- 将 `max_new_tokens` 从 100 降低到 40。
- 将最大长度比例从 2.0 收紧到 1.5。
- 只解码模型新生成部分，避免 prompt 混入输出。
- 清理 `Human:`、`Assistant:`、`User:` 等对话残留。
- 清理 `Human resources` 等无关模板残留。
- 对原句为单独标点或编号的无效句进行过滤。

优化后，纠错模块已经能够产生有效的成功案例。

## 四、中期实验结果

目前已在 GovReport validation split 上选取 2 个样本完成端到端测试。该实验属于小规模流程验证，不代表最终正式结果。

### 4.1 初始纠错实验结果

在早期版本中，系统共检测出 59 个事实风险句，但纠错成功数为 0：

```text
total_attempted: 59
total_succeeded: 0
success_rate: 0.0
```

主要失败原因是模型输出过长：

```text
output_too_long
appears_to_be_evidence_passage
```

这说明原始纠错 prompt 对模型约束不足，模型没有执行局部改写，而是倾向于生成长段解释或复制证据。

### 4.2 优化后纠错结果

在限制生成长度并改进输出抽取后，纠错效果明显改善。某次 2 样本实验结果如下：

```text
total_attempted: 59
total_succeeded: 42
success_rate: 0.7119
```

ROUGE 指标也有小幅提升：

| 指标    | 纠错前 | 纠错后 |
| ------- | -----: | -----: |
| ROUGE-1 | 0.3492 | 0.3548 |
| ROUGE-2 | 0.1132 | 0.1205 |
| ROUGE-L | 0.1376 | 0.1399 |

该结果说明纠错模块经过优化后能够对部分风险句进行有效改写。但需要注意，当前“成功”主要基于格式和长度校验，并不完全等价于事实真正修复，仍需结合人工案例分析和纠错后 NLI 复检。

### 4.3 摘要生成优化结果

后续实验发现，摘要生成阶段本身是系统误差的重要来源。早期摘要中出现了无关内容，例如模型生成了 Human Rights Watch、Myanmar、military junta 等与原文无关的主题。经过 prompt 和摘要清洗优化后，严重主题漂移有所缓解。

最新一轮 2 样本测试中，系统运行结果如下：

```text
total_sentences: 13
total_consistent: 3
total_inconsistent: 10
overall_consistency_rate: 23.08%
correction_success: 30.00%
ROUGE-1: 0.2650
ROUGE-2: 0.0694
ROUGE-L: 0.1383
```

该轮实验中，摘要主题漂移问题有所减少，ROUGE 相比前一轮有明显改善。但纠错成功率下降，原因主要包括：

1. 后处理规则更加严格。
2. 样本数量太少，比例波动明显。
3. 摘要中仍存在部分建议式或不完整句。
4. 纠错模型仍偶尔生成过长内容。

因此，中期阶段可以认为：摘要生成质量已有改善，但仍是后续优化重点。

## 五、案例分析

### 5.1 成功纠错案例

示例：

```text
Before:
The collaboration included sharing information and best practices learned during previous deployments.

After:
The collaboration included sharing information and best practices learned from previous deployments.
```

该案例中，纠错模型将 “during previous deployments” 改为 “from previous deployments”，语义更自然，且没有引入明显额外信息。

另一个案例：

```text
Before:
This disparity reflects a pattern of overrepresentation among Black students in disciplinary measures, highlighting the need for targeted intervention.

After:
This disparity highlights the need for targeted interventions to address the overrepresentation of Black students in disciplinary measures.
```

该案例保留了原句核心含义，并对表达进行了局部改写。

### 5.2 失败或存疑案例

部分纠错结果仍存在问题。例如：

```text
After:
The collaboration included sharing information and best practices learned from previous deployments.Human resources
```

该输出出现了无关模板残留。针对这一问题，后续已经在 corrector 中加入了额外清洗规则，用于截断和拒绝 `Human resources` 等异常片段。

另一个问题是，摘要中可能出现单独标点或不完整短句，例如：

```text
.
```

这类句子不应进入 NLI 或纠错流程。目前已经加入无效原句过滤规则。

## 六、当前问题

中期阶段主要发现以下问题：

1. **摘要生成仍不够稳定**  
   Qwen Instruct 在长文摘要任务中偶尔会生成报告外内容、建议式表述或模板化文本。

2. **事实一致性检测依赖证据检索质量**  
   当前主要使用词重叠检索，可能无法找到最合适的原文证据，导致 NLI 判断偏低。

3. **纠错成功率不等于事实修复率**  
   当前成功率主要表示输出通过格式和长度校验，仍需进行纠错后 NLI 复检和人工分析。

4. **小样本结果波动较大**  
   当前只在 2 个样本上多轮测试，指标不能代表最终效果。

5. **正式实验成本较高**  
   2 个样本运行约 10-20 分钟，若直接扩展到 500 篇，预计耗时较长，需要进一步优化或后台长时间运行。

## 七、下一步计划

后续工作计划如下：

1. **继续优化摘要生成**
   - 减少建议式输出。
   - 过滤无效短句。
   - 保证摘要为自然段文本。
   - 避免生成原文外主题。

2. **增加纠错后事实一致性复检**
   - 对 corrected_summary 再运行 NLI。
   - 比较纠错前后一致性率。
   - 该指标比单纯 ROUGE 更符合题目要求。

3. **扩大实验规模**
   - 先从 2 篇扩展到 5 篇、10 篇。
   - 稳定后再运行 500 篇 validation 样本。
   - 正式实验命令为：

```bash
python main.py --mode full
```

4. **改进证据检索方法**
   - 当前基础方法为 word_overlap。
   - 后续可尝试 semantic retrieval，用句向量检索更相关证据。
   - 对比改进前后的 NLI 一致性结果，作为进阶任务方向。

5. **完善案例分析**
   - 收集至少 10 个案例。
   - 包含成功纠错和失败纠错。
   - 对失败原因进行分类，如证据错误、模型过度生成、摘要本身跑题、NLI 判断误差等。

## 八、中期总结

目前项目已经完成端到端系统搭建，并成功在服务器 GPU 环境下运行。系统能够完成 GovReport 数据加载、长文分块摘要生成、句子级事实一致性检测、风险句局部纠错、ROUGE 评估、长度影响分析和案例收集。

中期实验表明，纠错模块经过优化后已经具备初步可用性，纠错成功率曾由 0% 提升到 71.19%。同时，摘要生成模块仍是当前系统的主要瓶颈，存在主题漂移、建议式输出和无关模板残留等问题。经过 prompt 和后处理优化后，严重跑题现象有所缓解，但仍需进一步改进。

下一阶段将重点扩大实验规模、加入纠错后 NLI 复检、优化证据检索方法，并整理不少于 10 个成功和失败案例，为最终报告和 500 篇正式实验做准备。
