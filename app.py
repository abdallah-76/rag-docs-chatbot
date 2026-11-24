"""Interface de chat : pose une question, le chatbot répond à partir des documents.

Lancer :  streamlit run app.py
"""
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from rag import RAG  # noqa: E402

st.set_page_config(page_title="Chatbot sur mes documents", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Indexation des documents...")
def load_rag(docs_dir: str, retriever: str) -> RAG:
    return RAG(docs_dir, retriever=retriever)


with st.sidebar:
    st.header("Réglages")
    docs_input = st.text_input("Dossier de documents", "data/sample_docs")
    docs_dir = str(Path(docs_input) if Path(docs_input).is_absolute() else ROOT / docs_input)
    retriever = st.selectbox("Recherche", ["tfidf", "embeddings"],
                             help="tfidf : mots-clés, rapide. embeddings : comprend le sens (sentence-transformers requis).")
    generator = st.selectbox("Réponse", ["extractive", "ollama", "anthropic"],
                             help="extractive : sans LLM. ollama : LLM local. anthropic : API Claude (clé requise).")
    k = st.slider("Passages utilisés (k)", 1, 6, 3)

try:
    rag = load_rag(docs_dir, retriever)
except Exception as error:  # noqa: BLE001
    st.error(f"Impossible de charger les documents : {error}")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader(f"{len(rag.documents)} documents indexés")
    for name in rag.documents:
        st.write(f"📄 {name}")
    st.caption(f"{len(rag.chunks)} passages")

st.title("📚 Chatbot sur mes documents")
st.caption("Les réponses viennent uniquement des documents indexés. Les sources sont affichées sous chaque réponse.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources: list[dict]) -> None:
    useful = [s for s in sources if s["score"] > 0]  # on masque les passages sans aucun rapport
    with st.expander(f"Sources ({len(useful)})"):
        for i, s in enumerate(useful, 1):
            st.markdown(f"**[{i}] {s['source']}** (page {s['page']}, score {s['score']})")
            text = re.sub(r"^#+\s*", "", s["text"], flags=re.M).replace("\n\n", "\n")
            st.text(text if len(text) <= 400 else text[:400] + "...")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

if question := st.chat_input("Pose ta question sur les documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Recherche..."):
                answer = rag.ask(question, k=k, generator=generator)
            st.write(answer.text)
            show_sources(answer.sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer.text, "sources": answer.sources}
            )
        except Exception as error:  # noqa: BLE001
            st.error(f"Erreur : {error}")
