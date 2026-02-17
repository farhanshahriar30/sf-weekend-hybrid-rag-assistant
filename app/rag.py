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
from typing import Dict, List, Tuple, Iterator
from dotenv import load_dotenv
from openai import OpenAI

from app.retrieval import RetrievalResult, retrieve
from app.prompts import build_messages
from app.utils import clean_snippet, filter_citations_used


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
        # Clean + normalize + truncate each chunk
        snippet = clean_snippet(r.text, per_chunk_chars)

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


# PHASE C: Full RAG pipeline with streaming response support
def stream_answer(
    question: str,
    mode: str,
    chunks: List[Dict],
    bm25,
    client_qdrant,
    embedder,
    id_map,
    top_k: int = 8,
    rrf_k: int = 60,
    history: List[Dict] | None = None,
) -> Iterator[Dict]:
    """
    Stream a grounded answer back to the UI.

    Yields dict events:
      {"type": "delta", "text": "..."}         # partial text chunks
      {"type": "final", "answer": str, "citations": [...], "retrieval": [...]}
    """

    # 1) Retrieve evidence (same as non-streaming)
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

    # 2) Build context pack
    context, citations = format_context(results)

    # 3) OpenAI client + messages
    client, model = get_openai_client()
    # messages = build_messages(question, context, history=history)
    # Build ONE prompt string (your SDK's responses.stream expects input=str)
    system_rules = (
        "You are a San Francisco weekend planning assistant for first-timers.\n"
        "Use ONLY the provided CONTEXT.\n"
        "If something is not in the context, say you don't know and ask a follow-up question.\n"
        "Cite supporting chunks using bracket citations like [1], [2].\n"
        "Be practical: itinerary bullets, neighborhoods, transit tips, and food suggestions.\n"
    )

    parts: List[str] = [system_rules]

    # Include chat history (optional)
    if history:
        parts.append("CHAT HISTORY:")
        for h in history:
            role = (h.get("role") or "").upper()
            content = h.get("content") or ""
            parts.append(f"{role}: {content}")
        parts.append("")  # spacer

    # Latest question + retrieved context
    parts.append(f"QUESTION:\n{question}\n")
    parts.append(f"CONTEXT:\n{context}\n")
    parts.append(
        "Write an answer grounded in the context. Include citations like [1], [2]."
    )

    prompt = "\n".join(parts)

    # 4) Stream response (Responses API streaming)
    acc: List[str] = []

    with client.responses.stream(
        model=model,
        input=prompt,
        temperature=0.3,
    ) as stream:
        for event in stream:
            etype = getattr(event, "type", None)

            # Text deltas arrive as "response.output_text.delta"
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    acc.append(delta)
                    yield {"type": "delta", "text": delta}

        # Once the stream ends, get the final assembled response object
        final = stream.get_final_response()

    answer_text = (final.output_text or "").strip()

    # 5) Keep only citations actually referenced in the final answer
    citations_used = filter_citations_used(answer_text, citations)

    # 6) Emit final payload (UI uses this to render citations + debug panel)
    yield {
        "type": "final",
        "answer": answer_text,
        "citations": citations_used,
        "retrieval": [
            {
                "id": r.id,
                "method": r.method,
                "score": float(r.score),
                "source": r.source,
                "chunk_index": r.chunk_index,
                "text": (r.text or "")[:400],  # 400 is nicer for debug than 200
            }
            for r in results
        ],
    }


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
    history: List[Dict] | None = None,
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
    messages = build_messages(question, context, history=history)

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
