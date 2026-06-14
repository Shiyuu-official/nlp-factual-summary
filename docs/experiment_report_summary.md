# GovReport Factual Summary Experiment Report

## 1. Experiment Setup

This run evaluates the factual summarization pipeline on the GovReport validation split. Although the result directory is named `server_full_100`, the applied configuration and log show that the run processed **500 samples**.

| Item | Setting |
| --- | --- |
| Dataset | `ccdv/govreport-summarization`, validation split |
| Number of samples | 500 |
| Summarization model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Consistency model | `facebook/bart-large-mnli` |
| Evidence retrieval | TF-IDF retrieval |
| Sentence splitting | Rule-based splitter |
| Correction model | Local Qwen correction |
| Hardware | Tesla T4 server GPU |
| Total runtime | 91,042.8 seconds, about 25.3 hours |

Dataset statistics:

| Metric | Value |
| --- | ---: |
| Average report length | 8,036.878 words |
| Maximum report length | 54,091 words |
| Minimum report length | 173 words |
| Average reference summary length | 566.446 words |
| Maximum reference summary length | 1,398 words |
| Minimum reference summary length | 63 words |

## 2. Overall Results

### 2.1 ROUGE Before and After Correction

| Metric | Original summary | Corrected summary | Change |
| --- | ---: | ---: | ---: |
| ROUGE-1 F1 | 0.2709 | 0.2626 | -0.0083 |
| ROUGE-2 F1 | 0.0880 | 0.0856 | -0.0024 |
| ROUGE-L F1 | 0.1496 | 0.1452 | -0.0044 |

The corrected summaries have slightly lower ROUGE scores. This is expected because ROUGE measures lexical overlap with the reference summary, while factual correction rewrites unsupported claims according to retrieved source evidence. A correction can improve factual support while reducing surface similarity.

### 2.2 Consistency Detection

| Metric | Value |
| --- | ---: |
| Samples | 500 |
| Checked summary sentences | 3,242 |
| Consistent sentences | 2,260 |
| Inconsistent sentences | 982 |
| Skipped fragments | 426 |
| Overall consistency rate | 69.71% |
| Overall error rate | 30.29% |

The generated summaries contain a meaningful number of factual issues: 982 inconsistent sentences were detected among 3,242 checked sentences.

### 2.3 Correction Results

| Metric | Value |
| --- | ---: |
| Correction attempts | 982 |
| Format-successful corrections | 807 |
| Format success rate | 82.18% |
| NLI-verified corrections | 80 |
| NLI verified rate among format-successful corrections | 9.91% |
| Improved corrections | 420 |
| Improved rate among format-successful corrections | 52.04% |
| Fully fixed corrections | 71 |
| Fixed rate among all attempts | 7.23% |

The correction model usually returns a usable sentence format, but strict NLI verification is much harder. The main verification failure reason is `not_improved_or_not_entailed`, meaning many rewritten sentences are still not clearly entailed by the retrieved evidence.

## 3. Length Impact

| Generated summary length | Samples | Avg. consistency rate | Avg. error rate | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-100 words | 113 | 60.34% | 39.66% | 0.1456 | 0.0531 | 0.0950 |
| 100-200 words | 187 | 63.44% | 36.56% | 0.2627 | 0.0842 | 0.1537 |
| 200-300 words | 200 | 76.27% | 23.73% | 0.3493 | 0.1112 | 0.1766 |
| 300-400 words | 0 | N/A | N/A | N/A | N/A | N/A |
| 400-500 words | 0 | N/A | N/A | N/A | N/A | N/A |
| 500+ words | 0 | N/A | N/A | N/A | N/A | N/A |

In this run, summaries in the 200-300 word range achieved the best consistency and ROUGE scores. The result does not support a simple claim that longer summaries always introduce more errors; very short summaries appear to lose important context.

## 4. Representative Correction Cases

### 4.1 Successful Cases

| Case | Sample | Sentence | Original claim | Corrected claim | Evidence summary | Verification |
| --- | ---: | ---: | --- | --- | --- | --- |
| 1 | 0 | 1 | The Assistant Secretary for CWMD will coordinate with DHS policy leadership to develop chemical defense strategy and implementation plans. | The Assistant Secretary for Countering Weapons of Mass Destruction will coordinate with DHS policy leadership to develop chemical defense strategy and implementation plans. | Evidence supports the expanded title and the strategy/implementation-plan responsibility. | Improved but not fully NLI-verified. |
| 2 | 3 | 2 | The risk register is updated regularly for changes in project scope, schedule, resources, and new risks. | Thresholds were not used within the risk management program to define when a risk becomes unacceptable. | Evidence directly states that thresholds were not used to trigger mitigation or contingency plans. | Verified and fixed; entailment improved from 0.0149 to 0.9834. |
| 3 | 3 | 4 | The risk register is also used to facilitate communication and collaboration among TRIO stakeholders. | The risk register is used to facilitate communication and collaboration among TRIO stakeholders. | Evidence discusses DHS modernization and SSP options, but does not strongly support the whole claim. | Improved but not fully NLI-verified. |
| 4 | 3 | 6 | The risk register is updated regularly for changes in project scope, schedule, resources, and new risks. | Thresholds were not used within the risk management program to define when a risk becomes unacceptable. | Same strong evidence as Case 2. | Verified and fixed; entailment improved from 0.0149 to 0.9834. |
| 5 | 3 | 8 | The risk register is also used to facilitate communication and collaboration among TRIO stakeholders. | The risk register is used to facilitate communication and collaboration among TRIO stakeholders. | Evidence is related to DHS modernization but does not fully entail the corrected sentence. | Improved but not fully NLI-verified. |

The strongest successful cases are Cases 2 and 4. In both examples, the correction replaces a generic unsupported risk-register claim with a sentence directly grounded in the retrieved report evidence.

### 4.2 Failed Cases

| Case | Sample | Sentence | Original claim | Evidence issue | Failure reason |
| --- | ---: | ---: | --- | --- | --- |
| 1 | 5 | 2 | `S. train-and-equip program continues.` | Retrieved evidence discusses the broader Syria conflict and does not cleanly support the fragment. | `output_too_long (18 vs 4 words)` |
| 2 | 5 | 5 | `S. train-and-equip program continues.` | Same fragmented sentence and broad Syria evidence. | `output_too_long (18 vs 4 words)` |
| 3 | 7 | 1 | `Mine Safety and Health Administration (MSHA).` | Retrieved evidence discusses coal tax assumptions and the Trust Fund, not a complete factual claim about MSHA. | `output_too_long (23 vs 6 words)` |
| 4 | 16 | 0 | The sequester would reduce discretionary spending by 3% across all federal departments and agencies. | Evidence discusses the Budget Control Act sequester trigger but does not directly support the 3% claim. | `output_too_long (27 vs 14 words)` |
| 5 | 17 | 0 | Transition to the Cerner system requires stakeholder involvement in acquisition to improve success. | Evidence discusses VA, VistA, and Cerner acquisition context, but the generated correction contained dialogue-like artifacts. | `dialogue_artifact` |

The failed cases show two main limitations. First, short fragments are difficult to correct because they are not complete factual claims. Second, the correction model can produce outputs that are too long or contain dialogue-like artifacts, so the post-processing filter rejects them.

## 5. Main Findings

1. The system completed a full 500-sample experiment and produced all intermediate and final result files.
2. The summaries contain measurable factual inconsistency: 30.29% of checked sentences were classified as inconsistent.
3. TF-IDF evidence retrieval plus BART-MNLI can identify unsupported claims and provide useful evidence for correction.
4. The correction stage is useful but conservative after verification: 82.18% format success, 52.04% improved among format-successful outputs, and 7.23% fully fixed among all attempts.
5. ROUGE decreased slightly after correction, so the final report should separate factual consistency from lexical similarity.
6. For the advanced comparison requirement, this TF-IDF run can be used as the improved system. A baseline run with word-overlap retrieval would provide the cleanest before/after comparison.

