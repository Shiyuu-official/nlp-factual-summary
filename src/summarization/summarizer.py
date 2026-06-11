"""Long-document summarizer using chunked Qwen2.5-1.5B-Instruct with beam search."""

import logging
import re
from typing import Optional
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

logger = logging.getLogger(__name__)


SUMMARY_SYSTEM_PROMPT = """You are a government-report summarization system.
You must write only facts that are directly supported by the provided report excerpt."""


SUMMARY_USER_PROMPT = """Write a factual summary of the government report excerpt below.

Rules:
- Return only the summary.
- Write one plain paragraph, not a list.
- Do not explain your process.
- Do not mention that you are an AI assistant.
- Do not evaluate the summary itself.
- Do not use Markdown, headings, bullets, numbering, or labels.
- Do not write recommendations unless the report explicitly states them.
- Do not introduce countries, organizations, people, events, or topics absent from the excerpt.
- Every sentence must be grounded in the excerpt.
- Preserve concrete entities, agencies, numbers, dates, and actions.
- Use complete sentences.

Report excerpt:
{text}

Summary:"""


class ChunkedSummarizer:
    """Summarizes long documents by chunking, summarizing each chunk, then merging."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 chunk_size: int = 2000, chunk_overlap: int = 200,
                 max_summary_length: int = 300, max_chunk_summary_length: int = 150,
                 num_beams: int = 4, device: str = "cpu"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_summary_length = max_summary_length
        self.max_chunk_summary_length = max_chunk_summary_length
        self.num_beams = num_beams
        self.device = device

        logger.info(f"Loading summarization model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        logger.info("Summarization model loaded.")

    def _split_document(self, text: str) -> List[str]:
        """Split document into overlapping chunks, respecting sentence boundaries."""
        chunks = []
        start = 0
        doc_len = len(text)

        while start < doc_len:
            end = min(start + self.chunk_size, doc_len)
            # Try to break at a sentence boundary near chunk_size
            if end < doc_len:
                for sep in [". ", "\n", ".\n", "\n\n"]:
                    pos = text.rfind(sep, start, end)
                    if pos > start + self.chunk_size // 2:
                        end = pos + len(sep)
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap if end < doc_len else end

        return chunks

    def _summarize_chunk(self, chunk: str, max_length: int) -> str:
        """Summarize a single text chunk."""
        prompt = self._build_prompt(chunk)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=2048, padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=20,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_len:]
        generated = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return self._clean_summary_text(generated)

    def _build_prompt(self, text: str) -> str:
        """Build a prompt using the model's chat template when available."""
        user_prompt = SUMMARY_USER_PROMPT.format(text=text)
        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{SUMMARY_SYSTEM_PROMPT}\n\n{user_prompt}"

    def _clean_summary_text(self, text: str) -> str:
        """Remove common instruction-following artifacts and normalize spacing."""
        if not text:
            return ""

        text = text.replace("\r", "\n")
        text = re.sub(r"^(summary|final summary)\s*:\s*", "", text.strip(),
                      flags=re.IGNORECASE)
        text = re.sub(r"\*\*(.*?)\*\*\s*:\s*", r"\1: ", text)
        text = re.sub(r"(^|\s)([-*]|\d+[.)])\s+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", text)
        for marker in [" Human:", " Assistant:", " User:", "\nHuman:", "\nAssistant:", "\nUser:"]:
            if marker in text:
                text = text.split(marker, 1)[0].strip()

        blocked_patterns = [
            r"\byou are an ai assistant\b",
            r"\bas an ai\b",
            r"\bi cannot\b",
            r"\bthe summary does not\b",
            r"\bthis summary\b",
            r"\bhuman resources management system\b",
            r"\bhuman resources department\b",
            r"\bhuman rights watch\b",
            r"\bmyanmar\b",
            r"\bmilitary junta\b",
            r"\bpeaceful protesters\b",
        ]

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        cleaned = []
        for sent in sentences:
            sent = sent.strip(" \"'`")
            if not sent:
                continue
            if re.fullmatch(r"\d+[.)]?", sent):
                continue
            lowered = sent.lower()
            if any(re.search(pattern, lowered) for pattern in blocked_patterns):
                continue
            cleaned.append(sent)

        return " ".join(cleaned).strip()

    def _merge_summaries(self, chunk_summaries: List[str]) -> str:
        """Merge multiple chunk summaries into a final summary."""
        if not chunk_summaries:
            return ""
        if len(chunk_summaries) == 1:
            return self._clean_summary_text(chunk_summaries[0])

        combined = "\n\n".join(chunk_summaries)
        # If already short enough, return as-is
        if len(combined.split()) <= self.max_summary_length:
            return self._clean_summary_text(combined)

        # Otherwise re-summarize the combined text
        return self._clean_summary_text(
            self._summarize_chunk(combined, self.max_summary_length)
        )

    def summarize_single(self, document: str) -> str:
        """Summarize a single document. Returns generated summary string."""
        chunks = self._split_document(document)
        logger.debug(f"Document split into {len(chunks)} chunks")

        chunk_summaries = []
        for chunk in chunks:
            try:
                s = self._summarize_chunk(chunk, self.max_chunk_summary_length)
                chunk_summaries.append(s)
            except Exception as e:
                logger.warning(f"Chunk summarization failed: {e}")

        return self._merge_summaries(chunk_summaries)

    def summarize_batch(self, samples: List[Dict],
                        checkpoint_path: Optional[str] = None,
                        existing_results: Optional[List[Dict]] = None) -> List[Dict]:
        """Summarize a batch of samples.

        Each sample (dict) must have keys: sample_id, report, reference_summary.
        Returns the same list with 'generated_summary' and 'num_chunks' added.
        """
        results = list(existing_results or [])
        done_ids = {r.get("sample_id") for r in results}
        for item in tqdm(samples, desc="Summarizing", unit="sample"):
            if item.get("sample_id") in done_ids:
                continue

            record = dict(item)
            try:
                chunks = self._split_document(record["report"])
                chunk_summaries = []
                for chunk in chunks:
                    s = self._summarize_chunk(chunk, self.max_chunk_summary_length)
                    chunk_summaries.append(s)
                record["generated_summary"] = self._merge_summaries(chunk_summaries)
                record["num_chunks"] = len(chunks)
            except Exception as e:
                logger.error(f"Sample {record['sample_id']}: summarization failed: {e}")
                record["generated_summary"] = ""
                record["num_chunks"] = 0
            results.append(record)
            done_ids.add(record.get("sample_id"))

            if checkpoint_path:
                from ..utils.io import save_json
                save_json(results, checkpoint_path)

        return results
