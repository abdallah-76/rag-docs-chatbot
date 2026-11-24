"""Génération de la réponse à partir des passages trouvés.

Trois options :
- "extractive" : pas de LLM. Renvoie les phrases les plus pertinentes des passages (gratuit, instantané).
- "ollama"     : LLM open source en local (https://ollama.com), ex. OLLAMA_MODEL=llama3.2
- "anthropic"  : API Claude (variable d'environnement ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from loader import Chunk
from textutils import FRENCH_STOPWORDS

NOT_FOUND = "Je n'ai pas trouvé la réponse dans les documents."

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond aux questions en t'appuyant UNIQUEMENT sur les extraits fournis. "
    "Réponds en français, de façon concise. Cite les extraits utilisés avec [1], [2]... "
    f"Si les extraits ne contiennent pas la réponse, réponds exactement : « {NOT_FOUND} »"
)


def build_user_prompt(question: str, passages: list[tuple[Chunk, float]]) -> str:
    blocks = [f"[{i}] (source : {c.source})\n{c.text}" for i, (c, _) in enumerate(passages, 1)]
    return "Extraits :\n\n" + "\n\n".join(blocks) + f"\n\nQuestion : {question}"


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    parts = [p for p in parts if not p.lstrip().startswith("#")]  # on ignore les titres
    cleaned = [re.sub(r"^[\-\*\s]+", "", p).strip() for p in parts]
    return [p for p in cleaned if len(p) > 3]


def extractive_answer(question: str, passages: list[tuple[Chunk, float]], n_sentences: int = 2) -> str:
    """Choisit les phrases des passages les plus proches de la question."""
    candidates: list[tuple[int, int, str]] = []  # (rang du passage, position, phrase)
    for rank, (chunk, _) in enumerate(passages):
        for pos, sentence in enumerate(_sentences(chunk.text)):
            candidates.append((rank, pos, sentence))
    # Le chevauchement entre passages crée des doublons et des fragments : on les retire.
    texts = [s for _, _, s in candidates]
    candidates = [
        c for i, c in enumerate(candidates)
        if not any(j != i and c[2] in texts[j] and (len(texts[j]) > len(c[2]) or j < i) for j in range(len(texts)))
    ]
    if not candidates:
        return NOT_FOUND

    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True,
                          stop_words=FRENCH_STOPWORDS)
    try:
        matrix = vec.fit_transform([s for _, _, s in candidates] + [question])
    except ValueError:  # question composée uniquement de mots vides
        return NOT_FOUND
    scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
    if scores.max() <= 0:
        return NOT_FOUND

    # On garde les meilleures phrases, mais seulement si elles sont presque aussi pertinentes que la 1re.
    best = [i for i in np.argsort(scores)[::-1][:n_sentences] if scores[i] >= 0.6 * scores.max()]
    best.sort(key=lambda i: (candidates[i][0], candidates[i][1]))
    return " ".join(candidates[i][2] for i in best)


def ollama_answer(question: str, passages: list[tuple[Chunk, float]]) -> str:
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, passages)},
        ],
    }
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())["message"]["content"].strip()


def anthropic_answer(question: str, passages: list[tuple[Chunk, float]]) -> str:
    import anthropic

    client = anthropic.Anthropic()  # lit ANTHROPIC_API_KEY
    message = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(question, passages)}],
    )
    return message.content[0].text.strip()


def generate(name: str, question: str, passages: list[tuple[Chunk, float]]) -> str:
    if not passages:
        return NOT_FOUND
    if name == "extractive":
        return extractive_answer(question, passages)
    if name == "ollama":
        return ollama_answer(question, passages)
    if name == "anthropic":
        return anthropic_answer(question, passages)
    raise ValueError(f"Générateur inconnu : {name}")
