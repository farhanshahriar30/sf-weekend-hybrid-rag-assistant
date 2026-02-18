"""
retrieval.py

Purpose:
- Given a user query, retrieve the most relevant chunks from your SF corpus.
- Supports:
  1) BM25 keyword search (exact-term matching, classic search engine style)
  2) Qdrant vector search (semantic similarity via embeddings)
  3) Hybrid search using RRF (Reciprocal Rank Fusion) to combine both ranked lists

Why hybrid?
- BM25 is great when the user uses the same keywords as the docs (e.g., "Clipper", "Muni")
- Vector search is great when wording differs but meaning is similar (e.g., "getting around without a car")
- RRF combines them robustly without worrying about score scales
"""

import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# from app.retrieval import build_id_map
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Config (match ingest.py)
CHUNKS_PATH = Path("data/processed/chunks.jsonl")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "sf_weekend_chunks"

# Must match the embedding model used during ingestion so query vectors live in same space
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Data structures for results
@dataclass
class RetrievalResult:
    """
    A single retrieved chunk plus metadata for citations and UI display.
    """

    id: int
    text: str
    source: str
    chunk_index: int
    method: str  # "bm25", "vector", or "hybrid"
    score: float


# PHASE A: Load chunk store (chunks.jsonl)
def load_chunks(chunks_path: Path = CHUNKS_PATH) -> List[Dict]:
    """
    Load the local chunk database written by ingest.py.

    Each line is a JSON object:
    { "id": ..., "source": ..., "chunk_index": ..., "text": ... }

    Why:
    - Needed for BM25 indexing (needs text)
    - Needed to map IDs -> chunk text + metadata for citations
    """
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing {chunks_path}. Run ingestion first to generate chunks.jsonl."
        )
    chunks: List[Dict] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    if not chunks:
        raise RuntimeError(f"{chunks_path} exists but contains no chunks")
    return chunks


def build_id_map(chunks: List[Dict]) -> Dict[int, Dict]:
    """
    Build a fast lookup: chunk_id -> chunk_record
    Why:
    - Retrieval returns IDs, and we need to quickly fetch text/source/chunk_index
    """
    return {int(c["id"]): c for c in chunks}


# PHASE B: BM25 keyword indexing + search
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer:
    - Lowercase
    - Extract alphanumeric/"'"

    Why simple:
    - Robust across messy PDF text
    - Good enough for BM25 baseline
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def build_bm25(chunks: List[Dict]) -> Tuple[BM25Okapi, List[List[str]]]:
    """
    Build a BM25 index over all chunks.

    Returns:
    - bm25 model
    - tokenized_corpus (stored so we don't retokenize everything repeatedly)
    """
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def bm25_search(
    query: str,
    chunks: List[Dict],
    bm25: BM25Okapi,
    top_k: int = 8,
) -> List[RetrievalResult]:
    """
    Run BM25 search:
    1) Tokenize query
    2) Score each chunk
    3) Return top_k results by score

    Output is a ranked list of RetrievalResult(method="bm25").
    """
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)

    # Get indices of top scores (descending)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :top_k
    ]

    results: List[RetrievalResult] = []
    for idx in ranked_idx:
        c = chunks[idx]
        results.append(
            RetrievalResult(
                id=int(c["id"]),
                source=c["source"],
                chunk_index=int(c["chunk_index"]),
                text=c["text"],
                score=float(scores[idx]),
                method="bm25",
            )
        )
    return results


# PHASE C: Vector search (Qdrant) + query embedding
def get_qdrant_client(url: str = QDRANT_URL) -> QdrantClient:
    """
    Create a Qdrant client for querying.
    """
    return QdrantClient(url=url, api_key=QDRANT_API_KEY)


def get_embedder(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    """
    Load the embedding model to embed user queries.
    Must match ingest.py model to ensure compatibility with stored vectors.
    """
    return SentenceTransformer(model_name)


def qdrant_search(
    query: str,
    client: QdrantClient,
    embedder: SentenceTransformer,
    id_map: Dict[int, Dict],
    top_k: int = 8,
) -> List[RetrievalResult]:
    """
    Run semantic search using Qdrant:
    1) Embed the query to a vector
    2) Ask Qdrant for top_k nearest chunks
    3) Convert returned points into RetrievalResult(method="vector")

    We use id_map to reconstruct metadata (source/chunk_index/text).
    """
    q_vec = embedder.encode([query])[0].tolist()

    res = client.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        limit=top_k,
        with_payload=True,
    )
    hits = res.points

    results: List[RetrievalResult] = []
    for h in hits:
        # h.id can be int or str depending on client version; normalize to int
        cid = int(h.id)

        # Prefer payload if present; fallback to id_map
        payload = h.payload or {}
        rec = id_map.get(cid, {})

        source = payload.get("source") or rec.get("source", "unknown")
        chunk_index = int(payload.get("chunk_index") or rec.get("chunk_index", -1))
        text = payload.get("text") or rec.get("text", "")

        results.append(
            RetrievalResult(
                id=cid,
                source=source,
                chunk_index=chunk_index,
                text=text,
                score=float(h.score),  # cosine similarity-ish score
                method="vector",
            )
        )
    return results


# PHASE D: Hybrid fusion via RRF
def rrf_fuse(
    ranked_lists: List[List[RetrievalResult]],
    k: int = 60,
    top_k: int = 8,
) -> List[RetrievalResult]:
    """
    Reciprocal Rank Fusion (RRF)

    Input:
    - ranked_lists: multiple ranked result lists (e.g., [bm25_results, vector_results])
      Each list is assumed to be ordered best -> worst.

    RRF score for a document d:
        score(d) = sum_over_lists 1 / (k + rank(d))
    where rank starts at 1.

    Why RRF:
    - Doesn't require calibrating score scales between BM25 and vector similarity
    - Very robust and common in hybrid retrieval systems

    Output:
    - top_k results with method="hybrid" and score=RRF score
    """
    fused_scores: Dict[int, float] = {}
    best_payload: Dict[int, RetrievalResult] = {}

    for results in ranked_lists:
        for rank, r in enumerate(results, start=1):
            fused_scores[r.id] = fused_scores.get(r.id, 0.0) + (1.0 / (k + rank))

            # Keep one repreentative copy of metadata to return later
            if r.id not in best_payload:
                best_payload[r.id] = r
    # Sort by fused score descending
    ranked_ids = sorted(
        fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True
    )[:top_k]

    fused_results: List[RetrievalResult] = []
    for cid in ranked_ids:
        base = best_payload[cid]
        fused_results.append(
            RetrievalResult(
                id=base.id,
                source=base.source,
                chunk_index=base.chunk_index,
                text=base.text,
                score=float(fused_scores[cid]),
                method="hybrid",
            )
        )
    return fused_results


# PHASE E: Unified retrieval entrypoint
def retrieve(
    query: str,
    mode: str,
    chunks: List[Dict],
    bm25: BM25Okapi,
    client: QdrantClient,
    embedder: SentenceTransformer,
    id_map: Dict[int, Dict],
    top_k: int = 8,
    rrf_k: int = 60,
) -> List[RetrievalResult]:
    """
    One function your app can call.

    mode:
    - "bm25"   -> keyword search only
    - "vector" -> semantic search only
    - "hybrid" -> BM25 + vector combined via RRF
    """
    mode = mode.lower().strip()
    if mode not in {"bm25", "vector", "hybrid"}:
        raise ValueError("mode must be one of: bm25, vector, hybrid")

    if mode == "bm25":
        return bm25_search(query, chunks, bm25, top_k=top_k)

    if mode == "vector":
        return qdrant_search(query, client, embedder, id_map, top_k=top_k)

    # Hybrid: get each list, then fuse with RRF
    bm25_results = bm25_search(query, chunks, bm25, top_k=top_k)
    vec_results = qdrant_search(query, client, embedder, id_map, top_k=top_k)

    return rrf_fuse([bm25_results, vec_results], k=rrf_k, top_k=top_k)


# Convenience loader (build everything once)
def build_retrievers() -> Tuple[
    List[Dict], Dict[int, Dict], BM25Okapi, QdrantClient, SentenceTransformer
]:
    """
    Build all retrieval components:
    - Load chunks
    - Build id_map
    - Build BM25 model
    - Create Qdrant client
    - Load embedder model

    Why:
    - Your Streamlit app can call this once and reuse objects for fast queries.
    """
    chunks = load_chunks()
    id_map = build_id_map(chunks)
    bm25, _ = build_bm25(chunks)
    client = get_qdrant_client()
    embedder = get_embedder()
    return chunks, id_map, bm25, client, embedder
