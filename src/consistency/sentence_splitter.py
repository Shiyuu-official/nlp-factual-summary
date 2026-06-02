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
        # Try punkt_tab first (newer nltk), then punkt (older)
        for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
            try:
                nltk.data.find(resource)
            except LookupError:
                logger.info(f"NLTK {resource} not found, downloading...")
                try:
                    nltk.download(resource.split("/")[-1], quiet=True, raise_on_error=True)
                except Exception as e:
                    logger.warning(f"NLTK download failed ({e}). "
                                   "If you are behind a firewall, set NLTK_DATA manually "
                                   "or run: python -m nltk.downloader punkt")
                    raise RuntimeError(
                        f"NLTK '{resource}' is missing and auto-download failed. "
                        f"Download it manually: python -m nltk.downloader punkt"
                    ) from e
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
