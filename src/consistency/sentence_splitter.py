"""Sentence splitting using regex — no NLTK download required.

Uses punctuation-boundary heuristics that work well for formal English text
(government reports, news, academic papers). Avoids the network dependency
of nltk punkt downloads, which is blocked behind some firewalls.
"""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Sentence-ending punctuation followed by space and capital letter / number / quote
_SENT_END = re.compile(
    r"(?<=[.!?])(?:\s+)(?=[A-Z0-9\"'‘“(])"
)


class SentenceSplitter:
    """Regex-based sentence tokenizer. No network calls, no downloads."""

    def __init__(self):
        pass  # nothing to download

    def split(self, text: str) -> List[str]:
        """Split text into sentences. Returns non-empty sentence strings."""
        if not text or not text.strip():
            return []

        # Split on sentence boundaries, keeping the punctuation
        parts = _SENT_END.split(text)
        sentences = [s.strip() for s in parts if s.strip()]

        # Merge very short fragments with the previous sentence
        merged = []
        for s in sentences:
            if merged and len(s.split()) <= 2 and len(merged[-1].split()) > 5:
                merged[-1] = merged[-1] + " " + s
            else:
                merged.append(s)

        return merged

    def split_with_indices(self, text: str) -> List[Tuple[int, str]]:
        """Split text and return (start_char_index, sentence_text) pairs."""
        sentences = self.split(text)
        result = []
        pos = 0
        for sent in sentences:
            idx = text.find(sent, pos)
            if idx == -1:
                idx = pos  # fallback
            result.append((idx, sent))
            pos = idx + len(sent)
        return result
