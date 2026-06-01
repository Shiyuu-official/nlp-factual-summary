"""Evidence retrieval strategies for factual consistency checking.

Implements two strategies:
  - WordOverlapRetriever: fast, based on word overlap (Jaccard similarity)
  - SemanticRetriever: uses sentence-transformers for semantic similarity

Both implement the BaseEvidenceRetriever interface so they can be swapped
via config with a single line change.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """A retrieved evidence snippet from the source document."""
    text: str
    score: float
    sentence_index: int


class BaseEvidenceRetriever(ABC):
    """Abstract base for evidence retrievers."""

    @abstractmethod
    def retrieve(self, hypothesis: str, document_sentences: List[str],
                 top_k: int = 3) -> List[Evidence]:
        """Retrieve top-k evidence sentences for a given hypothesis sentence."""


class WordOverlapRetriever(BaseEvidenceRetriever):
    """Retrieves evidence by word overlap (Jaccard similarity) with a window."""

    def __init__(self, window_size: int = 3):
        self.window_size = window_size

    def _jaccard(self, words_a: set, words_b: set) -> float:
        """Jaccard similarity between two sets of words."""
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def retrieve(self, hypothesis: str, document_sentences: List[str],
                 top_k: int = 3) -> List[Evidence]:
        """Compute word overlap for each document sentence, return top-k."""
        hyp_words = set(hypothesis.lower().split())
        if not hyp_words or not document_sentences:
            return []

        # Score each document sentence
        scored = []
        for i, sent in enumerate(document_sentences):
            sent_words = set(sent.lower().split())
            score = self._jaccard(hyp_words, sent_words)
            scored.append((i, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build evidence with context window
        evidences = []
        for idx, score in scored[:top_k]:
            # Expand to context window
            start = max(0, idx - self.window_size)
            end = min(len(document_sentences), idx + self.window_size + 1)
            context_text = " ".join(document_sentences[start:end])
            evidences.append(Evidence(
                text=context_text,
                score=score,
                sentence_index=idx,
            ))

        return evidences


class SemanticRetriever(BaseEvidenceRetriever):
    """Retrieves evidence by semantic similarity using sentence-transformers.

    Uses all-MiniLM-L6-v2 (80MB) by default — fast on CPU.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 window_size: int = 3, batch_size: int = 32,
                 device: str = "cpu"):
        self.model_name = model_name
        self.window_size = window_size
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._encoded_sentences = None
        self._sentence_list = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return
        logger.info(f"Loading semantic retriever model: {self.model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic retrieval. "
                "Install with: pip install sentence-transformers"
            )

    def _encode_document(self, document_sentences: List[str]):
        """Pre-encode all document sentences."""
        self._load_model()
        self._sentence_list = list(document_sentences)
        self._encoded_sentences = self._model.encode(
            document_sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def retrieve(self, hypothesis: str, document_sentences: List[str],
                 top_k: int = 3) -> List[Evidence]:
        """Encode hypothesis, compute cosine similarity, return top-k evidence."""
        if not document_sentences:
            return []

        self._load_model()

        # Encode hypothesis
        hyp_embedding = self._model.encode(
            [hypothesis],
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

        # Encode all document sentences
        doc_embeddings = self._model.encode(
            document_sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Cosine similarity
        hyp_norm = hyp_embedding / (np.linalg.norm(hyp_embedding) + 1e-8)
        doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(doc_norms, hyp_norm)

        # Top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        evidences = []
        for idx in top_indices:
            score = float(similarities[idx])
            # Expand to context window
            start = max(0, idx - self.window_size)
            end = min(len(document_sentences), idx + self.window_size + 1)
            context_text = " ".join(document_sentences[start:end])
            evidences.append(Evidence(
                text=context_text,
                score=score,
                sentence_index=int(idx),
            ))

        return evidences


def create_retriever(mode: str, **kwargs) -> BaseEvidenceRetriever:
    """Factory function to create the appropriate retriever.

    Args:
        mode: "word_overlap" or "semantic"
        **kwargs: passed to the retriever constructor

    Returns:
        BaseEvidenceRetriever instance
    """
    if mode == "semantic":
        return SemanticRetriever(
            model_name=kwargs.get("model_name", "all-MiniLM-L6-v2"),
            window_size=kwargs.get("window_size", 3),
            batch_size=kwargs.get("batch_size", 32),
            device=kwargs.get("device", "cpu"),
        )
    else:
        return WordOverlapRetriever(
            window_size=kwargs.get("window_size", 3),
        )
