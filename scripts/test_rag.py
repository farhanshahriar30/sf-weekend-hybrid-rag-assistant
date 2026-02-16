# scripts/test_rag.py
from app.retrieval import build_retrievers
from app.rag import answer_question


def main() -> None:
    # Build retrieval components once
    chunks, id_map, bm25, qdrant_client, embedder = build_retrievers()

    # Simple smoke-test question
    question = "Plan a first-timer 2-day weekend in San Francisco with transit tips and food neighborhoods."

    # Run full RAG (hybrid = BM25 + vector, fused with RRF)
    out = answer_question(
        question=question,
        mode="hybrid",
        chunks=chunks,
        bm25=bm25,
        client_qdrant=qdrant_client,
        embedder=embedder,
        id_map=id_map,
        top_k=15,
        rrf_k=60,
    )
    print("\n=== DEBUG COUNTS ===")
    print("retrieval_count:", len(out["retrieval"]))
    print("citations_count:", len(out["citations"]))

    # show the first 10 retrieved sources so we can see what's coming back
    print("\nFirst 10 retrieved sources:")
    for r in out["retrieval"][:10]:
        print("-", r["method"], r["source"], "chunk", r["chunk_index"])

    print("\n==================== ANSWER ====================\n")
    print(out["answer"])

    print("\n==================== CITATIONS ====================\n")
    for c in out["citations"][:8]:
        print(f"[{c['n']}] {c['source']} (chunk {c['chunk_index']})")


if __name__ == "__main__":
    main()
