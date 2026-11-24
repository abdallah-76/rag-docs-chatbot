"""Pipeline RAG : question -> passages pertinents -> réponse + sources."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from generator import generate
from loader import ROOT, Chunk, build_chunks
from retriever import build_retriever

DEFAULT_DOCS = ROOT / "data" / "sample_docs"


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)


class RAG:
    def __init__(self, docs_dir: str | Path = DEFAULT_DOCS, retriever: str = "tfidf",
                 chunk_size: int = 600, overlap: int = 100):
        self.chunks: list[Chunk] = build_chunks(docs_dir, chunk_size, overlap)
        if not self.chunks:
            raise ValueError(f"Aucun document (.txt, .md, .pdf) trouvé dans {docs_dir}")
        self.retriever = build_retriever(retriever, self.chunks)

    @property
    def documents(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def retrieve(self, question: str, k: int = 3):
        return self.retriever.search(question, k)

    def ask(self, question: str, k: int = 3, generator: str = "extractive") -> Answer:
        passages = self.retrieve(question, k)
        text = generate(generator, question, passages)
        sources = [
            {"source": c.source, "page": c.page, "score": round(score, 3), "text": c.text}
            for c, score in passages
        ]
        return Answer(text=text, sources=sources)
