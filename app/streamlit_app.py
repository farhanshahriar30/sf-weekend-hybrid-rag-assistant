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
from app.rag import answer_question, stream_answer

from pathlib import Path

# Streamlit best practice: set config before any other st.* calls
st.set_page_config(
    page_title="SF Weekend Hybrid RAG Assistant",
    page_icon="🌉",
    layout="wide",
)


def load_css(path: str) -> None:
    css = Path(path).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css("app/styles.css")


# Cache heavy objects so we don't reload models on every interaction
@st.cache_resource
def load_system():
    # Loads chunks, builds BM25, connects Qdrant, loads embedder
    chunks, id_map, bm25, qdrant_client, embedder = build_retrievers()
    return chunks, id_map, bm25, qdrant_client, embedder


def main():
    st.title("🌉 SF Weekend Hybrid RAG Assistant")
    st.caption(
        "Ask questions and get answers grounded in your SF PDF corpus, with citations."
    )

    # Conversation memory (chat history)
    if "history" not in st.session_state:
        st.session_state.history = []  # [{"role": "user"|"assistant", "content": str}, ...]

    if "last_citations" not in st.session_state:
        st.session_state.last_citations = []
    if "last_retrieval" not in st.session_state:
        st.session_state.last_retrieval = []

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

    left, right = st.columns([3, 1], gap="large")

    question = st.chat_input(
        "Ask something (e.g., “Plan a 2-day first-timer weekend in SF with food + transit tips”)",
        # value="Plan a 2-day first-timer weekend in SF with food + transit tips",
        # height=80,
    )

    with left:
        for m in st.session_state.history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if question and question.strip():
            # show the user's message
            with st.chat_message("user"):
                st.markdown(question.strip())

            # stream the assistant reply inside the assistant bubble
            with st.chat_message("assistant"):
                answer_box = st.empty()
                running = ""
                final_out = None

                with st.spinner("Retrieving + generating answer (streaming)..."):
                    for evt in stream_answer(
                        question=question.strip(),
                        mode=mode,
                        chunks=chunks,
                        bm25=bm25,
                        client_qdrant=qdrant_client,
                        embedder=embedder,
                        id_map=id_map,
                        top_k=top_k,
                        rrf_k=rrf_k,
                        history=st.session_state.history,
                    ):
                        if evt.get("type") == "delta":
                            running += evt.get("text", "")
                            answer_box.markdown(running)

                        elif evt.get("type") == "final":
                            final_out = evt
                            answer_box.markdown(final_out.get("answer", running))
                            st.session_state.last_citations = final_out.get(
                                "citations", []
                            )
                            st.session_state.last_retrieval = final_out.get(
                                "retrieval", []
                            )
                            break

            # only after assistant finishes: update history
            st.session_state.history.append(
                {"role": "user", "content": question.strip()}
            )
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": (final_out or {}).get("answer", running),
                }
            )

    with right:
        # Citations
        st.subheader("Citations")
        citations = st.session_state.last_citations
        if not citations:
            st.info("No citations were used in the final answer.")
        else:
            for c in citations:
                title = f"[{c['n']}] {c['source']} (chunk {c['chunk_index']})"
                with st.expander(title):
                    st.write(c.get("text", ""))

        # Debug panel
        if show_debug:
            st.subheader("Debug: Retrieval Results")
            for r in st.session_state.last_retrieval:
                st.write(
                    f"**{r['method']}** score={r['score']:.4f} | "
                    f"{r['source']} (chunk {r['chunk_index']})"
                )
                txt = r.get("text", "")
                st.caption(txt[:400] + ("..." if len(txt) > 400 else ""))


if __name__ == "__main__":
    main()
