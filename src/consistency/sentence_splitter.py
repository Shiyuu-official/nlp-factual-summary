"""Lightweight regex sentence splitting.

This avoids runtime NLTK downloads on restricted servers.
"""

import re
from typing import List, Tuple


class SentenceSplitter:
    """Rule-based sentence splitter for English report text."""

    _boundary = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
    _abbreviations = {
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sen.", "Rep.", "Gov.",
        "U.S.", "U.K.", "D.C.", "No.", "Fig.", "e.g.", "i.e.", "vs.",
    }

    def _protect_abbreviations(self, text: str) -> str:
        protected = text
        for abbr in self._abbreviations:
            protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
        return protected

    def _restore_abbreviations(self, text: str) -> str:
        return text.replace("<DOT>", ".")

    def split(self, text: str) -> List[str]:
        """Split text into non-empty sentence-like strings."""
        if not text or not text.strip():
            return []

        normalized = re.sub(r"\s+", " ", text.strip())
        protected = self._protect_abbreviations(normalized)
        pieces = self._boundary.split(protected)

        sentences = []
        for piece in pieces:
            sent = self._restore_abbreviations(piece).strip()
            if sent:
                sentences.append(sent)

        return sentences

    def split_with_indices(self, text: str) -> List[Tuple[int, str]]:
        """Split text and return (start_char_index, sentence_text) pairs."""
        sentences = self.split(text)
        result = []
        pos = 0
        for sent in sentences:
            idx = text.find(sent, pos)
            if idx == -1:
                idx = pos
            result.append((idx, sent))
            pos = idx + len(sent)
        return result
