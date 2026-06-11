"""Evidence retrieval strategies for factual consistency checking.

Implements three strategies:
  - WordOverlapRetriever: fast, based on word overlap (Jaccard similarity)
  - TfidfRetriever: stronger lexical baseline with TF-IDF n-grams
  - SemanticRetriever: uses sentence-transformers for semantic similarity

Both implement the BaseEvidenceRetriever interface so they can be swapped
via config with a single line change.
"""

import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def _word_tokens(text: str) -> set:
    """Tokenize text into lowercase word/number tokens for lexical matching."""
    return set(re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower()))


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
        hyp_words = _word_tokens(hypothesis)
        if not hyp_words or not document_sentences:
            return []

        # Score each document sentence
        scored = []
        for i, sent in enumerate(document_sentences):
            sent_words = _word_tokens(sent)
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


class TfidfRetriever(BaseEvidenceRetriever):
    """Retrieves evidence by TF-IDF cosine similarity with context windows."""

    def __init__(self, window_size: int = 3,
                 ngram_range: tuple = (1, 2),
                 max_features: int = 20000):
        self.window_size = window_size
        self.ngram_range = ngram_range
        self.max_features = max_features
        self._sentence_list = None
        self._vectorizer = None
        self._doc_matrix = None
        self._fallback_idf = None
        self._fallback_doc_vectors = None

    def _fit_document(self, document_sentences: List[str]) -> None:
        sentence_list = list(document_sentences)
        if (
            self._sentence_list == sentence_list
            and (self._doc_matrix is not None or self._fallback_doc_vectors is not None)
        ):
            return

        self._sentence_list = sentence_list
        self._vectorizer = None
        self._doc_matrix = None
        self._fallback_idf = None
        self._fallback_doc_vectors = None

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9'-]*\b",
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                sublinear_tf=True,
                norm="l2",
            )
            self._doc_matrix = self._vectorizer.fit_transform(sentence_list)
        except ImportError:
            logger.warning(
                "scikit-learn is not installed; using lightweight TF-IDF fallback"
            )
            self._fit_document_fallback(sentence_list)

    def _tokenize_ngrams(self, text: str) -> List[str]:
        tokens = list(_word_tokens(text))
        if self.ngram_range[1] < 2:
            return tokens
        ordered = re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower())
        bigrams = [
            f"{ordered[i]} {ordered[i + 1]}"
            for i in range(len(ordered) - 1)
        ]
        return tokens + bigrams

    def _fit_document_fallback(self, sentence_list: List[str]) -> None:
        tokenized = [self._tokenize_ngrams(sent) for sent in sentence_list]
        doc_freq = Counter()
        for tokens in tokenized:
            doc_freq.update(set(tokens))

        n_docs = max(len(sentence_list), 1)
        self._fallback_idf = {
            term: math.log((1 + n_docs) / (1 + freq)) + 1
            for term, freq in doc_freq.items()
        }
        self._fallback_doc_vectors = [
            self._fallback_vectorize_tokens(tokens)
            for tokens in tokenized
        ]

    def _fallback_vectorize_tokens(self, tokens: List[str]) -> dict:
        counts = Counter(tokens)
        weighted = {
            term: (1 + math.log(count)) * self._fallback_idf.get(term, 0.0)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        return {term: value / norm for term, value in weighted.items()}

    def _fallback_similarity(self, hypothesis: str) -> np.ndarray:
        query = self._fallback_vectorize_tokens(self._tokenize_ngrams(hypothesis))
        similarities = []
        for doc_vec in self._fallback_doc_vectors:
            score = sum(value * doc_vec.get(term, 0.0) for term, value in query.items())
            similarities.append(score)
        return np.array(similarities)

    def retrieve(self, hypothesis: str, document_sentences: List[str],
                 top_k: int = 3) -> List[Evidence]:
        """Retrieve evidence sentences by TF-IDF similarity."""
        if not hypothesis.strip() or not document_sentences:
            return []

        self._fit_document(document_sentences)
        if self._vectorizer is not None:
            query = self._vectorizer.transform([hypothesis])
            similarities = (self._doc_matrix @ query.T).toarray().ravel()
        else:
            similarities = self._fallback_similarity(hypothesis)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        evidences = []
        for idx in top_indices:
            score = float(similarities[idx])
            start = max(0, idx - self.window_size)
            end = min(len(document_sentences), idx + self.window_size + 1)
            context_text = " ".join(document_sentences[start:end])
            evidences.append(Evidence(
                text=context_text,
                score=score,
                sentence_index=int(idx),
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
        sentence_list = list(document_sentences)
        if self._sentence_list == sentence_list and self._encoded_sentences is not None:
            return

        self._sentence_list = sentence_list
        self._encoded_sentences = self._model.encode(
            sentence_list,
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

        # Encode all document sentences once per source document.
        self._encode_document(document_sentences)
        doc_embeddings = self._encoded_sentences

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
        mode: "word_overlap", "tfidf", or "semantic"
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
    if mode == "tfidf":
        return TfidfRetriever(
            window_size=kwargs.get("window_size", 3),
        )
    else:
        return WordOverlapRetriever(
            window_size=kwargs.get("window_size", 3),
        )
