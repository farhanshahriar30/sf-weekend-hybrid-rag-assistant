"""
streamlit_app.py

A tiny demo UI for the SF Weekend Hybrid RAG Assistant.

User flow:
1) Choose retrieval mode (bm25 / vector / hybrid)
2) Ask a question
3) See a grounded answer + citations + optional debug retrieval info
"""

import streamlit as st

from app.retrieval import build_retrievers
from app.rag import answer_question


# Cache heavy objects so we don't reload models on every interaction
@st.cache_resource
def load_system():
    # Loads chunks, builds BM25, connects Qdrant, loads embedder
    chunks, id_map, bm25, qdrant_client, embedder = build_retrievers()
    return chunks, id_map, bm25, qdrant_client, embedder


def main():
    st.set_page_config(
        page_title="SF Weekend Hybrid RAG Assistant", page_icon="🌉", layout="wide"
    )

    st.title("🌉 SF Weekend Hybrid RAG Assistant")
    st.caption(
        "Ask questions and get answers grounded in your SF PDF corpus, with citations."
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        mode = st.selectbox("Retrieval mode", ["hybrid", "bm25", "vector"], index=0)
        top_k = st.slider("Top-K chunks", min_value=5, max_value=25, value=15, step=1)
        rrf_k = st.slider(
            "RRF k (only for hybrid)", min_value=20, max_value=100, value=60, step=5
        )

        show_debug = st.checkbox("Show debug retrieval panel", value=False)

    # Load system
    with st.spinner("Loading retrievers (BM25 + embedder + Qdrant)..."):
        chunks, id_map, bm25, qdrant_client, embedder = load_system()

    # Input box
    question = st.text_input(
        "Ask something (e.g., “Plan a 2-day first-timer weekend in SF with food + transit tips”)",
        value="Plan a 2-day first-timer weekend in SF with food + transit tips",
    )

    ask = st.button("Ask")

    if ask and question.strip():
        with st.spinner("Retrieving + generating answer..."):
            out = answer_question(
                question=question.strip(),
                mode=mode,
                chunks=chunks,
                bm25=bm25,
                client_qdrant=qdrant_client,
                embedder=embedder,
                id_map=id_map,
                top_k=top_k,
                rrf_k=rrf_k,
            )

        st.subheader("Answer")
        st.markdown(out["answer"])

        st.subheader("Citations")
        for c in out["citations"]:
            st.write(f"[{c['n']}] {c['source']} (chunk {c['chunk_index']})")

        if show_debug:
            st.subheader("Debug: Retrieval Results")
            for r in out["retrieval"]:
                st.write(
                    f"**{r['method']}** score={r['score']:.4f} | "
                    f"{r['source']} (chunk {r['chunk_index']})"
                )
                st.caption(r["text"][:400] + ("..." if len(r["text"]) > 400 else ""))


if __name__ == "__main__":
    main()
