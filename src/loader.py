"""Chargement des documents (.txt, .md, .pdf) et découpage en passages (chunks)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUPPORTED = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    id: int
    source: str  # nom du fichier
    page: int  # page du PDF (1 pour les fichiers texte)
    text: str


def load_documents(folder: str | Path) -> list[tuple[str, int, str]]:
    """Retourne une liste (fichier, page, texte)."""
    docs = []
    for path in sorted(Path(folder).rglob("*")):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            docs.append((path.name, 1, path.read_text(encoding="utf-8")))
        elif suffix == ".pdf":
            from pypdf import PdfReader

            for i, page in enumerate(PdfReader(str(path)).pages, start=1):
                docs.append((path.name, i, page.extract_text() or ""))
    return docs


def _pieces(paragraph: str, size: int) -> list[str]:
    """Coupe un paragraphe trop long en morceaux <= size, en respectant les phrases."""
    if len(paragraph) <= size:
        return [paragraph]
    out, cur = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        if cur and len(cur) + len(sentence) + 1 > size:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    """Regroupe les paragraphes en passages d'environ `size` caractères.

    Un petit chevauchement (`overlap`) est gardé entre deux passages pour ne pas
    couper une idée en deux.
    """
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    units = [piece for p in paragraphs if p for piece in _pieces(p, size)]

    chunks, cur = [], ""
    for unit in units:
        if cur and len(cur) + len(unit) + 2 > size:
            chunks.append(cur)
            tail = cur[-overlap:].split(" ", 1)[-1] if overlap else ""
            cur = f"{tail}\n\n{unit}" if tail else unit
        else:
            cur = f"{cur}\n\n{unit}" if cur else unit
    if cur.strip():
        chunks.append(cur)
    return chunks


def build_chunks(folder: str | Path, size: int = 600, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, page, text in load_documents(folder):
        for piece in chunk_text(text, size, overlap):
            chunks.append(Chunk(id=len(chunks), source=source, page=page, text=piece))
    return chunks
