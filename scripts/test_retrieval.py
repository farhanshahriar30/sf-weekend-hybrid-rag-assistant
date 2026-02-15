from app.retrieval import build_retrievers, retrieve


def preview(results, n=5):
    for i, r in enumerate(results[:n], start=1):
        snippet = r.text.replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        print(f"\n{i}. [{r.method}] score={r.score:.4f}")
        print(f"   source={r.source}  chunk={r.chunk_index}  id={r.id}")
        print(f"   {snippet}")


def main():
    chunks, id_map, bm25, client, embedder = build_retrievers()

    queries = [
        "Weekend itinerary for first time in San Francisco",
        "How do I get around SF without a car? Muni, Clipper, BART",
        "Best neighborhoods for food and walking",
        "Things to do near Fisherman's Wharf and Pier 39",
    ]

    for q in queries:
        print("\n" + "=" * 90)
        print("QUERY:", q)

        bm = retrieve(q, "bm25", chunks, bm25, client, embedder, id_map, top_k=8)
        ve = retrieve(q, "vector", chunks, bm25, client, embedder, id_map, top_k=8)
        hy = retrieve(q, "hybrid", chunks, bm25, client, embedder, id_map, top_k=8)

        print("\n--- BM25 (keyword) ---")
        preview(bm, n=5)

        print("\n--- VECTOR (semantic) ---")
        preview(ve, n=5)

        print("\n--- HYBRID (RRF) ---")
        preview(hy, n=5)


if __name__ == "__main__":
    main()
