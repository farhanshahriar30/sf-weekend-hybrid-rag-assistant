"""
Lightweight evaluation harness for the SF Weekend Hybrid RAG Assistant.

What it checks:
- Does the model produce an answer?
- Does it use citations (grounding)?
- How many citations on average?
- How many UNIQUE sources (PDFs) are cited?
- How diverse retrieval is (unique sources retrieved)?
- Does the model "refuse" (not enough context / I don't know)?
- How long each query takes?

Outputs:
- Writes JSON results to data/processed/evals.json
"""

from __future__ import annotations

import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any

from app.retrieval import build_retrievers
from app.rag import answer_question

OUT_PATH = Path("data/processed/evals.json")

# Small starter eval set
EVAL_QUESTIONS: List[Dict[str, str]] = [
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
        "id": "areas_to_stay",
        "question": "What are good areas to stay for a first-time visitor and why?",
        "mode": "hybrid",
    },
    {
        "id": "muni_line_numbers_sfo_to_union_square",
        "question": "Give me the exact Muni/BART line numbers and step-by-step route from SFO to Union Square.",
        "mode": "hybrid",
    },
    {
        "id": "ferry_building_hours",
        "question": "What time does the Ferry Building open and close today?",
        "mode": "hybrid",
    },
    {
        "id": "quiet_stay_easy_transit",
        "question": "Where should I stay if I want quiet nights but still easy public transit access, and why?",
        "mode": "hybrid",
    },
]

# Heuristic: catches common "I can't answer from context" responses
_REFUSAL_RE = re.compile(
    r"\b(i\s+don['’]t\s+know|not\s+enough\s+context|not\s+enough\s+detail|"
    r"provided\s+context\s+does\s+not|can['’]t\s+responsibly|"
    r"i\s+don['’]t\s+have\s+enough)\b",
    re.IGNORECASE,
)


def is_refusal_like(answer: str) -> bool:
    """True if the answer looks like a refusal / missing-context response."""
    if not answer:
        return True
    return bool(_REFUSAL_RE.search(answer))


def unique_sources(items: List[Dict[str, Any]], key: str = "source") -> int:
    """Count distinct sources across a list of dicts."""
    return len({(it.get(key) or "") for it in items if it.get(key)})


def run_eval(top_k: int = 15, rrf_k: int = 60) -> Dict:
    """
    Run the eval suite once and return:
    - summary metrics
    - per-question rows (latency, citations, preview, etc.)
    """
    # Load heavy objects once (embedder/BM25/Qdrant)
    chunks, id_map, bm25, qdrant_client, embedder = build_retrievers()

    rows: List[Dict] = []
    t0 = time.perf_counter()

    for item in EVAL_QUESTIONS:
        q = item["question"]
        mode = item.get("mode", "hybrid")

        start = time.perf_counter()

        # Same RAG pipeline used by Streamlit
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

        # Extract what we evaluate
        answer = (out.get("answer") or "").strip()
        citations = (
            out.get("citations") or []
        )  # already filtered to used citations (your rag.py does this)
        retrieval = out.get("retrieval") or []  # debug info (all retrieved chunks)

        # Metrics per question
        citations_used = len(citations)
        has_citation = citations_used > 0
        unique_sources_used = unique_sources(citations, key="source")
        retrieval_unique_sources = unique_sources(retrieval, key="source")
        refusal_like = is_refusal_like(answer)

        rows.append(
            {
                "id": item["id"],
                "mode": mode,
                "question": q,
                "latency_s": round(elapsed, 3),
                # Answer health
                "answer_len": len(answer),
                "answer_preview": answer[:300],
                # Grounding / evidence
                "citations_used": citations_used,
                "has_citation": has_citation,
                "unique_sources_used": unique_sources_used,
                # Retrieval diversity (debug)
                "retrieval_count": len(retrieval),
                "retrieval_unique_sources": retrieval_unique_sources,
                # Refusal / missing context detector
                "refusal_like": refusal_like,
                # Keep raw evidence for inspection
                "citations": citations,
            }
        )

    total_s = time.perf_counter() - t0

    # Summary metrics
    n = len(rows)
    citation_coverage = sum(r["has_citation"] for r in rows) / n if n else 0.0
    avg_citations = sum(r["citations_used"] for r in rows) / n if n else 0.0
    empty_answer_rate = sum(r["answer_len"] == 0 for r in rows) / n if n else 0.0
    avg_latency = sum(r["latency_s"] for r in rows) / n if n else 0.0

    avg_unique_sources_used = (
        sum(r["unique_sources_used"] for r in rows) / n if n else 0.0
    )
    avg_retrieval_unique_sources = (
        sum(r["retrieval_unique_sources"] for r in rows) / n if n else 0.0
    )
    refusal_rate = sum(r["refusal_like"] for r in rows) / n if n else 0.0

    summary = {
        "n_questions": n,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "total_time_s": round(total_s, 3),
        # Existing checks
        "citation_coverage": round(citation_coverage, 3),
        "avg_citations_used": round(avg_citations, 3),
        "empty_answer_rate": round(empty_answer_rate, 3),
        "avg_latency_s": round(avg_latency, 3),
        # New checks
        "avg_unique_sources_used": round(avg_unique_sources_used, 3),
        "avg_retrieval_unique_sources": round(avg_retrieval_unique_sources, 3),
        "refusal_rate": round(refusal_rate, 3),
    }

    return {"summary": summary, "rows": rows}


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = run_eval(top_k=15, rrf_k=60)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=== EVAL SUMMARY ===")
    for k, v in results["summary"].items():
        print(f"{k}: {v}")

    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
