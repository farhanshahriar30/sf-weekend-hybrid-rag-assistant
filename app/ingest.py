import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

from pypdf import PdfReader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


# ---- Config (keep simple for v1) ----
RAW_DIR = Path("data/raw/sf")
# Where we write processed artifacts (chunk store, later maybe BM25 artifacts, etc.)
OUT_DIR = Path("data/processed")
# A local, human-inspectable "chunk database" (JSON Lines: one JSON object per line)
CHUNKS_PATH = OUT_DIR / "chunks.jsonl"
# Qdrant connection details
QDRANT_URL = "http://localhost:6333"
COLLECTION = "sf_weekend_chunks"
# Embedding model: converts chunk text -> numeric vector for semantic search
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Chunking strategy (simple v1):
# - CHUNK_SIZE: how many characters per chunk
# - CHUNK_OVERLAP: repeated characters between chunks so we don't cut ideas mid-sentence
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# PHASE 1: Find PDFs
def iter_pdfs(root: Path) -> Iterable[Path]:
    """
    Recursively yield every .pdf file under the given root folder.
    Why:
    - No hardcoding file names
    - You can add/remove PDFs and ingestion will adapt automatically
    """
    for p in root.rglob("*.pdf"):
        yield p


# PHASE 2: PDF -> text extraction
def pdf_to_text(pdf_path: Path) -> str:
    """
    Read a PDF and extract text from each page.
    Why:
    - Retrieval + embeddings work on text, not raw PDFs
    Notes:
    - extract_text() can return None on some pages, so we fall back to ""
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append(txt)
    return "\n".join(pages)


# PHASE C: Light text clean up
def normalize_text(text: str) -> str:
    """
    Clean up common PDF text issues.
    What we do:
    - Normalize line endings
    - Strip extra whitespace from each line
    Why:
    - Cleaner chunk boundaries
    - Better retrieval signal (less junk spacing)
    """
    text = text.replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    return text


# PHASE D: Chunking
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split a long document into overlapping chunks (character-based).
    How it works:
    - Take text[start : start + chunk_size] as a chunk
    - Move forward, but step back by `overlap` characters so content overlaps
    Why overlap matters:
    - Prevents cutting an important sentence/idea exactly at a boundary
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be > overlap")
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be > overlap")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        # only keep non-empty chunks
        if chunk:
            chunks.append(chunk)
        # Move start forward but keep some overlap
        start = end - overlap
        if start < 0:
            start = 0

        # If we have reached the end, stop
        if end == n:
            break
    return chunks


# PHASE E: Stable chunks IDs
def stable_id(s: str) -> int:
    """
    Create a stable integer ID from a string.
    We use it for Qdrant point IDs.
    Why stable IDs:
    - If you re-run ingestion, the same chunk gets the same ID
    - That allows "upsert" to update instead of duplicating
    Implementation:
    - sha256 hash -> take first 16 hex chars -> convert to int
    """
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


# PHASE F: Create Qdrant collections (if missing)
def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    """
    Make sure the Qdrant collection exists.
    If it doesn't exist, create it with:
    - vectors of length `vector_size` (depends on embedding model)
    - cosine similarity (standard for sentence embeddings)
    """
    collections = client.get_collections().collections
    if any(c.name == COLLECTION for c in collections):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


# Main Pipeline (orchestration)
def main() -> None:
    """
    Runs the full ingestion pipeline:
    1) PDFs -> cleaned text -> chunks
    2) Save chunks locally to JSONL
    3) Embed chunks into vectors
    4) Upsert vectors + metadata into Qdrant
    """
    # Make sure processed folder exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Read PDFs -> chunk -> build records
    records: List[Dict] = []
    pdf_paths = list(iter_pdfs(RAW_DIR))

    # If we didn't find any PDFs, stop early with helpful error
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found under {RAW_DIR.resolve()}")
    print(f"Found {len(pdf_paths)} PDFs under {RAW_DIR}")

    # Loop through each PDF and create chunk records
    for pdf in tqdm(pdf_paths, desc="Reading PDFs"):
        # Extract raw text from PDF pages
        raw = pdf_to_text(pdf)

        # Clean up spacing/line breaks
        text = normalize_text(raw)

        # Skip PDFs that effectively contain no text (could be empty or mostly images)
        if len(text.strip()) < 200:
            continue

        # Split the document into overlapping chunks
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        # Use relative path as the "source" identifier (good for citations later)
        # Example: sf_corpus/logistics/2026-02-11__sftravel__muni.pdf
        source = str(pdf.relative_to(RAW_DIR)).replace("\\", "/")

        # Convert each chunk into a record with ID + metadata
        for i, ch in enumerate(chunks):
            # A unique string for this chunk (used to derive a stable integer ID)
            chunk_uid_str = f"{source}::chunk{i}"

            rec = {
                "id": stable_id(chunk_uid_str),  # stable Qdrant point ID
                "source": source,  # which PDF it came from
                "chunk_index": i,  # which chunk within that PDF
                "text": ch,  # the chunk content
            }
            records.append(rec)

    # Step 2: Save chunk store locally (JSONL)
    # Why we save this:
    # - Debug what was extracted
    # - Rebuild indexes without re-parsing PDFs
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} chunks to {CHUNKS_PATH}")

    # Step 3: Embed chunks (text -> vectors)
    # Load the embedding model once
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Determine embedding dimension (needed to create the Qdrant collection)
    vector_size = model.get_sentence_embedding_dimension()

    # Step 4: Ensure Qdrant collection exists
    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client, vector_size)

    # Get all chunk texts in order
    texts = [r["text"] for r in records]
    # Encode them in batches for speed
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    # Step 5: Prepare Qdrant points (id + vector + payload)
    # Payload is stored with each vector so we can:
    # - show citations ("source" + "chunk_index")
    # - display the original chunk text when retrieved
    points = []
    for r, vec in zip(records, embeddings):
        payload = {
            "source": r["source"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
        }
        points.append(PointStruct(id=r["id"], vector=vec.tolist(), payload=payload))

    # Step 6: Upsert points into Qdrant in batches
    # Why batches:
    # - Faster
    # - More reliable than sending thousands of points at once
    batch_size = 256
    for i in tqdm(range(0, len(points), batch_size), desc="Upserting to Qdrant"):
        client.upsert(collection_name=COLLECTION, points=points[i : i + batch_size])
    print(f"Upserted {len(points)} points to Qdrant collection '{COLLECTION}'")


if __name__ == "__main__":
    # Allows: pythin app/ingest.py (runs main)
    main()
