"""Sentence splitting with nltk."""

import logging
from typing import List, Tuple

import nltk

logger = logging.getLogger(__name__)


class SentenceSplitter:
    """Wraps nltk sentence tokenizer with error handling.

    Lazily downloads punkt data on first use.
    """

    def __init__(self):
        self._ready = False

    def _ensure_ready(self):
        if self._ready:
            return
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        self._ready = True

    def split(self, text: str) -> List[str]:
        """Split text into sentences. Returns non-empty sentence strings."""
        self._ensure_ready()
        if not text or not text.strip():
            return []
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def split_with_indices(self, text: str) -> List[Tuple[int, str]]:
        """Split text and return (start_char_index, sentence_text) pairs.

        Uses a simple scan to find each sentence's position in the original text.
        """
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
