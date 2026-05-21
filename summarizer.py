"""
长文摘要生成模块
使用 Qwen2.5-1.5B-Instruct 模型生成分块摘要并融合
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LongDocumentSummarizer:
    """长文档摘要生成器 - 使用分块摘要策略"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"正在加载模型 {model_name}...")
        logger.info(f"使用设备: {self.device}")

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
        logger.info("模型加载完成")

    def split_document(self, document: str) -> List[str]:
        """将长文档切分为多个块"""
        chunks = []
        start = 0
        doc_len = len(document)

        while start < doc_len:
            end = min(start + self.chunk_size, doc_len)

            if end < doc_len:
                for sep in ['. ', '\n', '。', '\n\n']:
                    last_sep = document[start:end].rfind(sep)
                    if last_sep != -1:
                        end = start + last_sep + len(sep)
                        break

            chunk = document[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap if end < doc_len else end

        return chunks

    def generate_chunk_summary(self, chunk: str, max_length: int = 150) -> str:
        """为单个文本块生成摘要"""
        prompt = f"请为以下文本生成一个简洁的摘要（不超过{max_length}字）：\n\n{chunk}\n\n摘要："

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=20,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "摘要：" in generated_text:
            summary = generated_text.split("摘要：")[-1].strip()
        else:
            summary = generated_text.strip()

        return summary

    def merge_summaries(self, chunk_summaries: List[str]) -> str:
        """将多个块的摘要融合为最终摘要"""
        if not chunk_summaries:
            return ""

        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        combined_text = "\n\n".join(chunk_summaries)

        if len(combined_text.split()) < 500:
            return combined_text

        final_summary = self.generate_chunk_summary(
            combined_text,
            max_length=300
        )

        return final_summary

    def summarize(self, document: str, max_length: int = 300) -> str:
        """
        为长文档生成摘要

        Args:
            document: 输入文档
            max_length: 期望的最大摘要长度

        Returns:
            生成的摘要
        """
        chunks = self.split_document(document)
        logger.info(f"文档被切分为 {len(chunks)} 个块")

        chunk_summaries = []
        for i, chunk in enumerate(tqdm(chunks, desc="生成块摘要")):
            try:
                summary = self.generate_chunk_summary(chunk, max_length=max_length // len(chunks) + 50)
                chunk_summaries.append(summary)
            except Exception as e:
                logger.error(f"处理第 {i+1} 块时出错: {e}")
                continue

        final_summary = self.merge_summaries(chunk_summaries)

        return final_summary

    def batch_summarize(self, documents: List[Dict]) -> List[Dict]:
        """
        批量生成摘要

        Args:
            documents: 文档列表，每个元素包含 'report' 和 'id'

        Returns:
            结果列表，每个元素包含原始数据和生成的摘要
        """
        results = []

        for item in tqdm(documents, desc="批量生成摘要"):
            try:
                summary = self.summarize(item['report'])
                results.append({
                    'id': item['id'],
                    'report': item['report'],
                    'reference_summary': item.get('summary', ''),
                    'generated_summary': summary
                })
            except Exception as e:
                logger.error(f"处理样本 {item['id']} 时出错: {e}")
                results.append({
                    'id': item['id'],
                    'report': item['report'],
                    'reference_summary': item.get('summary', ''),
                    'generated_summary': "",
                    'error': str(e)
                })

        return results
