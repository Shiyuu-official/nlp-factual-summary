"""
事实一致性检测模块
使用 NLI 模型检测摘要句是否能被原文支持
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from typing import List, Dict, Tuple
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactConsistencyChecker:
    """事实一致性检测器"""

    def __init__(
        self,
        nli_model_name: str = "facebook/bart-large-mnli",
        device: str = None,
        sentence_window: int = 3
    ):
        self.sentence_window = sentence_window

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"正在加载 NLI 模型 {nli_model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.nli_model.eval()
        logger.info("NLI 模型加载完成")

    def split_sentences(self, text: str) -> List[str]:
        """将文本切分为句子"""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def find_evidence(
        self,
        hypothesis_sentence: str,
        document: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        从原文中查找支持假设句子的证据片段

        Args:
            hypothesis_sentence: 需要验证的摘要句
            document: 原文
            top_k: 返回最相关的 k 个证据片段

        Returns:
            证据片段列表，每个元素为 (证据文本, 相似度分数)
        """
        doc_sentences = self.split_sentences(document)

        if not doc_sentences:
            return []

        hypothesis_words = set(hypothesis_sentence.lower().split())

        similarities = []
        for i, sent in enumerate(doc_sentences):
            start_idx = max(0, i - self.sentence_window)
            end_idx = min(len(doc_sentences), i + self.sentence_window + 1)

            evidence_context = " ".join(doc_sentences[start_idx:end_idx])

            evidence_words = set(evidence_context.lower().split())
            if len(hypothesis_words) > 0:
                overlap = len(hypothesis_words & evidence_words) / len(hypothesis_words)
            else:
                overlap = 0

            similarities.append((evidence_context, overlap))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def check_entailment(self, premise: str, hypothesis: str) -> Dict:
        """
        使用 NLI 模型检查蕴含关系

        Args:
            premise: 前提（证据片段）
            hypothesis: 假设（摘要句）

        Returns:
            包含标签和置信度的字典
        """
        max_length = 512
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        labels = ["entailment", "neutral", "contradiction"]
        scores = {label: prob.item() for label, prob in zip(labels, probabilities)}

        predicted_label = labels[torch.argmax(probabilities).item()]

        return {
            "label": predicted_label,
            "scores": scores,
            "entailment_score": scores["entailment"]
        }

    def check_sentence_consistency(
        self,
        summary_sentence: str,
        document: str
    ) -> Dict:
        """
        检查单个摘要句与原文的一致性

        Args:
            summary_sentence: 摘要句
            document: 原文

        Returns:
            检测结果
        """
        evidences = self.find_evidence(summary_sentence, document)

        if not evidences:
            return {
                "sentence": summary_sentence,
                "is_consistent": False,
                "reason": "no_evidence_found",
                "evidences": [],
                "nli_results": []
            }

        nli_results = []
        max_entailment = 0

        for evidence_text, similarity in evidences:
            nli_result = self.check_entailment(evidence_text, summary_sentence)
            nli_result["evidence_similarity"] = similarity
            nli_results.append(nli_result)

            max_entailment = max(max_entailment, nli_result["entailment_score"])

        threshold = 0.5
        is_consistent = max_entailment >= threshold

        return {
            "sentence": summary_sentence,
            "is_consistent": is_consistent,
            "max_entailment_score": max_entailment,
            "evidences": [(e[0], e[1]) for e in evidences],
            "nli_results": nli_results
        }

    def check_summary_consistency(
        self,
        summary: str,
        document: str
    ) -> Dict:
        """
        检查整个摘要的事实一致性

        Args:
            summary: 摘要文本
            document: 原文

        Returns:
            一致性检测结果
        """
        sentences = self.split_sentences(summary)

        if not sentences:
            return {
                "total_sentences": 0,
                "consistent_sentences": 0,
                "inconsistent_sentences": 0,
                "consistency_rate": 0.0,
                "sentence_results": []
            }

        sentence_results = []
        for sent in tqdm(sentences, desc="检查句子一致性", leave=False):
            result = self.check_sentence_consistency(sent, document)
            sentence_results.append(result)

        consistent_count = sum(1 for r in sentence_results if r["is_consistent"])
        inconsistent_count = len(sentence_results) - consistent_count
        consistency_rate = consistent_count / len(sentence_results) if sentence_results else 0.0

        return {
            "total_sentences": len(sentence_results),
            "consistent_sentences": consistent_count,
            "inconsistent_sentences": inconsistent_count,
            "consistency_rate": consistency_rate,
            "sentence_results": sentence_results
        }

    def batch_check(
        self,
        samples: List[Dict]
    ) -> List[Dict]:
        """
        批量检查一致性

        Args:
            samples: 样本列表，每个元素包含 'report' 和 'generated_summary'

        Returns:
            检测结果列表
        """
        results = []

        for item in tqdm(samples, desc="批量检查一致性"):
            if not item.get('generated_summary'):
                results.append({
                    'id': item.get('id'),
                    'error': 'empty_summary'
                })
                continue

            try:
                consistency_result = self.check_summary_consistency(
                    item['generated_summary'],
                    item['report']
                )

                results.append({
                    'id': item.get('id'),
                    'report': item['report'],
                    'summary': item['generated_summary'],
                    'consistency_result': consistency_result
                })
            except Exception as e:
                logger.error(f"检查样本 {item.get('id')} 时出错: {e}")
                results.append({
                    'id': item.get('id'),
                    'error': str(e)
                })

        return results
