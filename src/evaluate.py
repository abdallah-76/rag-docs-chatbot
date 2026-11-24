"""Évalue la qualité de la RECHERCHE (la partie la plus importante d'un RAG).

Pour chaque question de eval/questions.json, on vérifie si le bon document fait partie des
k premiers passages trouvés.
  - Hit@1 : le bon document est en première position
  - Hit@3 : le bon document est dans les 3 premiers
  - MRR   : moyenne de 1/rang du bon document (1.0 = toujours premier)

Usage :
    python src/evaluate.py                       # TF-IDF seulement
    python src/evaluate.py --retrievers tfidf embeddings
"""
from __future__ import annotations

import argparse
import json

from loader import ROOT
from rag import DEFAULT_DOCS, RAG

QUESTIONS = ROOT / "eval" / "questions.json"
RESULTS = ROOT / "results"


def evaluate(name: str, questions: list[dict], docs_dir) -> tuple[dict, list[dict]]:
    rag = RAG(docs_dir, retriever=name)
    hit1 = hit3 = 0
    reciprocal = 0.0
    misses = []
    for item in questions:
        found = [c.source for c, _ in rag.retrieve(item["question"], k=len(rag.chunks))]
        rank = found.index(item["source"]) + 1 if item["source"] in found else None
        hit1 += rank == 1
        hit3 += rank is not None and rank <= 3
        reciprocal += 1 / rank if rank else 0
        if rank != 1:
            misses.append({"question": item["question"], "attendu": item["source"], "trouvé": found[0]})
    n = len(questions)
    return {"retriever": name, "n": n, "hit1": hit1 / n, "hit3": hit3 / n, "mrr": reciprocal / n}, misses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--retrievers", nargs="+", default=["tfidf"], choices=["tfidf", "embeddings"])
    parser.add_argument("--docs", default=str(DEFAULT_DOCS))
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows, all_misses = [], {}
    for name in args.retrievers:
        row, misses = evaluate(name, questions, args.docs)
        rows.append(row)
        all_misses[name] = misses
        print(f"{name:11s} Hit@1 {row['hit1']:.0%} | Hit@3 {row['hit3']:.0%} | MRR {row['mrr']:.3f}  ({row['n']} questions)")

    lines = ["| Retriever | Questions | Hit@1 | Hit@3 | MRR |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['retriever']} | {r['n']} | {r['hit1']:.0%} | {r['hit3']:.0%} | {r['mrr']:.3f} |")
    for name, misses in all_misses.items():
        if misses:
            lines += ["", f"**Questions où {name} ne place pas le bon document en premier :**", ""]
            lines += [f"- « {m['question']} » : attendu `{m['attendu']}`, trouvé `{m['trouvé']}`" for m in misses]

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "retrieval_eval.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nÉcrit : results/retrieval_eval.md")


if __name__ == "__main__":
    main()
