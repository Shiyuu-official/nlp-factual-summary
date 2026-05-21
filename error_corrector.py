"""
自动纠错模块
对检测出的不一致句子进行局部改写
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorCorrector:
    """错误纠正器 - 基于原文证据进行局部改写"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"正在加载纠错模型 {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("纠错模型加载完成")

    def correct_sentence(
        self,
        incorrect_sentence: str,
        evidence_texts: List[str],
        original_context: str = ""
    ) -> Dict:
        """
        纠正单个错误句子

        Args:
            incorrect_sentence: 错误的摘要句
            evidence_texts: 证据文本列表
            original_context: 原文上下文（可选）

        Returns:
            纠错结果
        """
        evidence_str = "\n".join([f"证据{i+1}: {e}" for i, e in enumerate(evidence_texts[:3])])

        prompt = f"""请根据以下证据文本，修正给定的句子中的事实错误。保持句子的整体结构和风格，只修改与证据不符的部分。

证据文本：
{evidence_str}

需要修正的句子：
{incorrect_sentence}

修正后的句子："""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        ).to(self.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    min_new_tokens=10,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "修正后的句子：" in generated_text:
                corrected = generated_text.split("修正后的句子：")[-1].strip()
            else:
                corrected = generated_text.strip()

            if not corrected or corrected == incorrect_sentence:
                success = False
            else:
                success = True

            return {
                "original": incorrect_sentence,
                "corrected": corrected,
                "success": success,
                "evidence_used": evidence_texts[:3]
            }

        except Exception as e:
            logger.error(f"纠错过程中出错: {e}")
            return {
                "original": incorrect_sentence,
                "corrected": incorrect_sentence,
                "success": False,
                "error": str(e)
            }

    def correct_summary(
        self,
        summary: str,
        consistency_result: Dict,
        document: str
    ) -> Dict:
        """
        纠正整个摘要中的错误句子

        Args:
            summary: 原始摘要
            consistency_result: 一致性检测结果
            document: 原文

        Returns:
            纠错结果
        """
        sentence_results = consistency_result.get('sentence_results', [])

        if not sentence_results:
            return {
                "original_summary": summary,
                "corrected_summary": summary,
                "corrections": [],
                "num_corrected": 0
            }

        corrections = []
        corrected_sentences = []

        for sent_result in tqdm(sentence_results, desc="纠正错误句子", leave=False):
            if not sent_result['is_consistent']:
                evidences = [e[0] for e in sent_result.get('evidences', [])]

                if evidences:
                    correction_result = self.correct_sentence(
                        sent_result['sentence'],
                        evidences,
                        document
                    )
                    corrections.append(correction_result)
                    corrected_sentences.append(correction_result['corrected'])
                else:
                    corrections.append({
                        "original": sent_result['sentence'],
                        "corrected": sent_result['sentence'],
                        "success": False,
                        "reason": "no_evidence"
                    })
                    corrected_sentences.append(sent_result['sentence'])
            else:
                corrected_sentences.append(sent_result['sentence'])

        corrected_summary = " ".join(corrected_sentences)

        num_corrected = sum(1 for c in corrections if c.get('success', False))

        return {
            "original_summary": summary,
            "corrected_summary": corrected_summary,
            "corrections": corrections,
            "num_corrected": num_corrected,
            "total_errors": len([s for s in sentence_results if not s['is_consistent']])
        }

    def batch_correct(
        self,
        consistency_checks: List[Dict]
    ) -> List[Dict]:
        """
        批量纠正

        Args:
            consistency_checks: 一致性检测结果列表

        Returns:
            纠错结果列表
        """
        results = []

        for item in tqdm(consistency_checks, desc="批量纠错"):
            if 'error' in item or 'consistency_result' not in item:
                results.append({
                    'id': item.get('id'),
                    'error': 'invalid_input'
                })
                continue

            try:
                correction_result = self.correct_summary(
                    item['summary'],
                    item['consistency_result'],
                    item['report']
                )

                results.append({
                    'id': item.get('id'),
                    'report': item['report'],
                    **correction_result
                })
            except Exception as e:
                logger.error(f"纠正样本 {item.get('id')} 时出错: {e}")
                results.append({
                    'id': item.get('id'),
                    'error': str(e)
                })

        return results
