"""Recherche des passages pertinents.

Deux méthodes, même interface :
- "tfidf"      : correspondance de mots (rapide, aucun téléchargement, très lisible)
- "embeddings" : correspondance de sens via sentence-transformers (comprend les paraphrases)
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from loader import Chunk
from textutils import FRENCH_STOPWORDS

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]: ...


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words=FRENCH_STOPWORDS,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_df=0.85,
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        best = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in best]


class EmbeddingRetriever:
    name = "embeddings"

    def __init__(self, chunks: list[Chunk], model_name: str = DEFAULT_EMBEDDING_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers n'est pas installé : pip install -r requirements-embeddings.txt"
            ) from exc
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
        )

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ vec
        best = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in best]


def build_retriever(name: str, chunks: list[Chunk]) -> Retriever:
    if name == "tfidf":
        return TfidfRetriever(chunks)
    if name == "embeddings":
        return EmbeddingRetriever(chunks)
    raise ValueError(f"Retriever inconnu : {name}")
