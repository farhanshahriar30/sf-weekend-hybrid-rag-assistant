"""
evals.py

Lightweight evaluation harness for the SF Weekend Hybrid RAG Assistant.

What it checks:
- Does the model produce an answer?
- Does it use citations (grounding)?
- How many citations on average?
- How long does each query take?

Outputs:
- Writes JSON results to data/processed/evals.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from app.retrieval import build_retrievers
from app.rag import answer_question


OUT_PATH = Path("data/processed/evals.json")

# Small starter eval set
# Each item has:
# - id: stable name for comparing across runs
# - question: what we ask the app
# - mode: retrieval strategy to use (bm25/vector/hybrid)
EVAL_QUESTIONS: List[Dict] = [
    {
        "id": "itinerary_first_timer",
        "question": "Plan a 2-day first-timer weekend in SF with food + transit tips",
        "mode": "hybrid",
    },
    {
        "id": "airport_to_city",
        "question": "How do I get to downtown SF from SFO using transit?",
        "mode": "hybrid",
    },
    {
        "id": "no_car_get_around",
        "question": "What are good areas to stay for a first-time visitor and why?",
        "mode": "hybrid",
    },
]


def run_eval(top_k: int = 15, rrf_k: int = 60) -> Dict:
    """
    Run the eval suite once and return a dict with:
    - summary metrics
    - per-question rows (latency, citations, preview, etc.)

    top_k:
      how many chunks we retrieve per query
    rrf_k:
      only used for hybrid mode, controls RRF fusion behavior
    """
    # Load heavy ovjects ONCE
    # This avoids reloading the embedder/BM25/Qdrant client for every question
    chunks, id_map, bm25, qdrant_client, embedder = build_retrievers()

    rows = []  # we store one results dict per question here
    t0 = time.perf_counter()  # total runtime timer

    # Evaluate each question in our suite
    for item in EVAL_QUESTIONS:
        q = item["question"]
        mode = item.get("mode", "hybrid")

        # measure latency per question (including retrieval + OpenAI generation)
        start = time.perf_counter()

        # Call real RAG pipeline (same one used by streamlit)
        out = answer_question(
            question=q,
            mode=mode,
            chunks=chunks,
            bm25=bm25,
            client_qdrant=qdrant_client,
            embedder=embedder,
            id_map=id_map,
            top_k=top_k,
            rrf_k=rrf_k,
        )

        elapsed = time.perf_counter() - start

        # Pull out key signls we care about
        # Did we generate text?
        # Did the model include citations like [1], [2]?
        answer = (out.get("answer") or "").strip()
        citations = out.get("citations") or []

        # 5) Record a single “row” for this question
        #    Keeping citations here makes it easy to inspect why an answer is grounded.
        rows.append(
            {
                "id": item["id"],
                "mode": mode,
                "question": q,
                "latency_s": round(elapsed, 3),
                "answer_len": len(answer),
                "citations_used": len(citations),
                "has_citation": len(citations) > 0,
                "answer_preview": answer[:300],
                "citations": citations,
            }
        )
    total_s = time.perf_counter() - t0
    # Aggregate lightweight metrics for a quick “health check”
    n = len(rows)
    citation_coverage = sum(r["has_citation"] for r in rows) / n if n else 0.0
    avg_citations = sum(r["citations_used"] for r in rows) / n if n else 0.0
    empty_answer_rate = sum(r["answer_len"] == 0 for r in rows) / n if n else 0.0
    avg_latency = sum(r["latency_s"] for r in rows) / n if n else 0.0

    summary = {
        "n_questions": n,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "total_time_s": round(total_s, 3),
        "citation_coverage": round(citation_coverage, 3),
        "avg_citations_used": round(avg_citations, 3),
        "empty_answer_rate": round(empty_answer_rate, 3),
        "avg_latency_s": round(avg_latency, 3),
    }
    return {"summary": summary, "rows": rows}
