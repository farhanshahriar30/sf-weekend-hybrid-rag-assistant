# 🌉 SF Weekend Hybrid RAG Assistant (BM25 + Vector + Hybrid)

A Streamlit app that answers San Francisco weekend planning questions **grounded in a PDF corpus**, using **Hybrid Retrieval-Augmented Generation (RAG)**:

- **BM25** keyword search (fast, exact term match)
- **Qdrant** vector search (semantic similarity)
- **Hybrid** retrieval using **RRF (Reciprocal Rank Fusion)**
- **Grounded answers with citations** like `[1] [2]` mapping to retrieved PDF chunks

---

## Demo

- **Live App:** *(add your Streamlit link here)*
- **Qdrant Cloud:** Used as the vector database backend
- **Citations:** Shown under each assistant response (and stored in chat history)

---

## What This Project Does

Given a user question like:

> “Plan a 2-day first-timer weekend in SF with food + transit tips”

The system:

1. Retrieves relevant evidence chunks from your SF PDF corpus using:
   - BM25 (keyword)
   - Qdrant (vector)
   - or Hybrid (BM25 + vector fused via RRF)
2. Builds a context pack like:
   ```text
   [1] source=... chunk=...
   snippet...
   ---
   [2] source=... chunk=...
   snippet...

