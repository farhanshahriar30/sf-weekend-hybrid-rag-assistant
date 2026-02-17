"""
rag.py

Goal:
- Take a user question
- Retrieve relevant chunks (bm25 / vector / hybrid)
- Send question + chunks to OpenAI
- Get a grounded answer with citations like [1], [2] that map to chunks
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from app.ingest import _BAD_RE, _PUA_RE
from app.retrieval import RetrievalResult, retrieve
import re
from app.prompts import build_messages


_CITE_RE = re.compile(r"\[(\d+)\]")


def filter_citations_used(answer_text: str, citations: List[Dict]) -> List[Dict]:
    """
    Keep only citations whose [n] appears in the model's answer text.
    """
    used = set(int(n) for n in _CITE_RE.findall(answer_text))
    return [c for c in citations if c["n"] in used]


# PHASE A: Turn retrieved chunks into a "context pack"
def format_context(
    results: List[RetrievalResult],
    max_chars: int = 24000,
    per_chunk_chars: int = 1200,
) -> Tuple[str, List[Dict]]:
    """
    Convert retrieved chunks into:
    1) A single context string the LLM can read
    2) A citations list Streamlit can display nicely

    Context format:
      [1] source=... chunk=...
      chunk text...

      ---
      [2] source=... chunk=...
      chunk text...
    """
    blocks: List[str] = []
    citations: List[Dict] = []
    total_chars = 0

    for i, r in enumerate(results, start=1):
        # Cap each chunk so one PDF chunk doesn't eat the whole context budget
        snippet = r.text or ""
        snippet = snippet.replace("\x00", " ")  # nulls -> space
        snippet = _PUA_RE.sub(" ", snippet)  # kill weird    glyphs
        snippet = _BAD_RE.sub(" ", snippet)  # kill invisible junk chars
        snippet = re.sub(r"\s+", " ", snippet).strip()  # collapse whitespace/newlines
        snippet = snippet[:per_chunk_chars]

        # Use snippet (NOT full r.text) inside the block
        block = f"[{i}] source={r.source} chunk={r.chunk_index}\n{snippet}\n"

        # Stop once we hit the total context budget
        if total_chars + len(block) > max_chars:
            break

        blocks.append(block)
        citations.append(
            {
                "n": i,
                "source": r.source,
                "chunk_index": r.chunk_index,
                "text": snippet,
            }
        )
        total_chars += len(block)

    context = "\n---\n".join(blocks)
    return context, citations


# PHASE B: Load environment variables + init OpenAI client
def get_openai_client() -> Tuple[OpenAI, str]:
    """
    Read OPENAI_API_KEY and OPENAI_MODEL from .env and return:
    - an OpenAI client
    - the model name

    Why:
    - Keeps secrets out of code (good security + good resume practice)
    - Lets you switch models by editing .env (no code changes)
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Add it to your .env file.")

    client = OpenAI(api_key=api_key)
    return client, model


# PHASE D: Full RAG pipeline (retrieve -> context -> generate)
def answer_question(
    question: str,
    mode: str,
    chunks: List[Dict],
    bm25,
    client_qdrant,
    embedder,
    id_map,
    top_k: int = 8,
    rrf_k: int = 60,
) -> Dict:
    """
    End-to-end RAG function used by Streamlit.

    Steps:
    - Retrieve relevant chunks (bm25/vector/hybrid)
    - Build context pack + citations
    - Call OpenAI to generate an answer grounded in that context

    Returns:
      {
        "answer": str,
        "citations": [{"n":1,"source":"...","chunk_index":...}, ...],
        "retrieval": [{"id":..,"method":..,"score":..,"source":..,"chunk_index":..}, ...]
      }
    """

    # D1) Retrieve evidence chunks
    # Why:
    # - Retrieval narrows the doc space so the model answers from your PDFs
    # - Hybrid mode (RRF) is usually best overall
    results = retrieve(
        query=question,
        mode=mode,
        chunks=chunks,
        bm25=bm25,
        client=client_qdrant,
        embedder=embedder,
        id_map=id_map,
        top_k=top_k,
        rrf_k=rrf_k,
    )

    # D2) Format chunks into an LLM-readable context pack
    context, citations = format_context(results)
    # D3) Create OpenAI client + choose model
    client, model = get_openai_client()
    # D4) Build messages (rules + question + context)
    messages = build_messages(question, context)

    # D5) Call the model and extract the answer text
    resp = client.responses.create(
        model=model,
        input=messages,
        temperature=0.3,  # lower temp = more grounded + consistent
    )

    answer_text = (resp.output_text or "").strip()
    # Only show citations the model actually referenced in the answer
    citations_used = filter_citations_used(answer_text, citations)
    if not citations_used:
        citations_used = citations[:3]
    # D6) Return answer + citations + debug retrieval info
    # Why keep retrieval debug?
    # - Very useful during development
    # - In Streamlit, you can show it under an "Advanced" expander
    return {
        "answer": answer_text,
        "citations": citations_used,
        "retrieval": [
            {
                "id": r.id,
                "method": r.method,
                "score": float(r.score),
                "source": r.source,
                "chunk_index": r.chunk_index,
            }
            for r in results
        ],
    }
