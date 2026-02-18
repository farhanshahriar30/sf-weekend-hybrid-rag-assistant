# 🌉 SF Weekend Hybrid RAG Assistant (Hybrid RAG: BM25 + Vector + RRF)

A Streamlit app that answers **San Francisco weekend planning** questions using **Retrieval-Augmented Generation (RAG)** grounded in a PDF corpus. It retrieves evidence from the corpus (keyword + semantic search), then generates an answer **with citations** pointing to the exact chunks used.

**Live demo:** *(paste your Streamlit Community Cloud link here)*

---

## ✅ What this app does

You ask something like:

> “Plan a 2-day first-timer weekend in SF with food + transit tips.”

The app will:

1. Retrieve the most relevant text chunks from the SF PDF corpus using:
   - **BM25** (keyword search)
   - **Qdrant vector search** (semantic similarity)
   - or **Hybrid** (BM25 + vector combined with **RRF**)
2. Build a grounded context pack from the retrieved chunks.
3. Call an OpenAI model with strict grounding rules:
   - **Use ONLY the provided context**
   - If the answer isn’t in the context: **say you don’t know**
   - Add bracket citations like **[1], [2]**
4. Stream the answer to the UI and display citations per message.

---

## ✨ Key features

- **Hybrid retrieval (recommended default)**
  - BM25 catches exact terms like “Clipper”, “Muni”
  - Vector search catches meaning even when wording differs
  - RRF merges both lists reliably
- **Grounded answers**
  - Citations are chunk-based and shown per assistant message
- **Streaming UI**
  - Answer appears token-by-token in the chat bubble
- **Debug mode**
  - Optional retrieval debug panel (if enabled in UI)

---

## 🔎 Retrieval modes (what the UI switch means)

- **bm25**: keyword-only retrieval (fast, exact term match)
- **vector**: semantic-only retrieval using Qdrant
- **hybrid**: BM25 + vector fused via **Reciprocal Rank Fusion (RRF)**

---

## 🧾 How citations work

- Retrieved chunks are numbered in the context sent to the LLM.
- The LLM is instructed to cite chunks as **[n]**.
- After generation, the app filters citations so only those actually used in the final answer remain.
- Citations are stored inside the assistant message history so each answer keeps its own evidence.

---

## 🛠️ Setup (local)

### 1) Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd sf-weekend-hybrid-rag-assistant
```
### 2) Create and activate a virtual environment
Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```
macOS / Linux:
```
python -m venv .venv
source .venv/bin/activate
```
### 3) Install dependencies
```
pip install -r requirements.txt
```
### 4) Environment variables

Create a .env file in the project root:
```
OPENAI_API_KEY=YOUR_OPENAI_KEY
OPENAI_MODEL=gpt-5.2

QDRANT_URL=https://<your-qdrant-cluster-endpoint>
QDRANT_API_KEY=<your-qdrant-api-key>
```
## 🧠 Qdrant

### Option A (recommended for deployment): Qdrant Cloud
1. Create a cluster in Qdrant Cloud
2. Copy the cluster endpoint + API key
3. Put them in:
   - .env (local)
   - Streamlit Secrets (deployment)

### Option B (local development): Qdrant via Docker

Start Qdrant locally:

```
docker compose up -d
```
Then in .env use:
```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

## 📦 Ingestion (build the searchable corpus)

The app needs:
- A local chunk store at data/processed/chunks.jsonl
- A populated Qdrant collection with embeddings

### 1) Put PDFs under
```
data/raw/sf/
```
### 2) Run ingestion
```
python app/ingest.py
```
Expected output includes:
- Found N PDFs
- Wrote N chunks to data/processed/chunks.jsonl
- Upserted ... points to Qdrant collection

Check if Qdrant has your collection
```
python scripts/qdrant_test.py
```
You should see something like: 
```
collections=[CollectionDescription(name='sf_weekend_chunks')]
```
## 🚀 Run the app locally
```
streamlit run app/streamlit_app.py
```

## 🧪 Evaluation
### This project includes a lightweight eval harness that checks:
- whether the model answers
- whether citations are present
- average citation count
- unique sources cited
- retrieval diversity
- refusal rate
- latency

Run:
```
python evals.py
```
It writes results to:
```
data/processed/evals.json
```

## 🌐 Deployment (Streamlit Community Cloud)

### Step 1) Push required files to GitHub

To deploy without ingesting at deploy-time, your repo should contain:
- data/processed/chunks.jsonl

⚠️ Warning

If chunks.jsonl is missing, the deployed app will crash

### Step 2) Create the Streamlit app

1. Go to Streamlit Community Cloud and click New App
2. Select:
   - GitHub repo
   - branch
   - Main file path: app/streamlit_app.py

### Step 3) Add secrets (DO NOT paste keys into code)
App → Settings → Secrets

```
OPENAI_API_KEY = "YOUR_OPENAI_KEY"
OPENAI_MODEL = "gpt-5.2"
QDRANT_URL = "https://<your-qdrant-cluster-endpoint>"
QDRANT_API_KEY = "<your-qdrant-api-key>"
```
### Step 4) Restart
After saving secrets, restart/reboot the app from the Streamlit Cloud UI.

⚠️ Who pays for OpenAI usage?

If you deploy using your OpenAI API key:
   - Any user interacting with the deployed app will use your key
   - Your OpenAI account is billed for their token usage

## 🧰 Tech stack

- **Streamlit**: UI + chat experience  
- **Qdrant**: Vector database for semantic retrieval  
- **rank-bm25**: BM25 keyword retrieval  
- **SentenceTransformers**: Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)  
- **PyPDF**: PDF parsing and text extraction  
- **OpenAI**: Grounded answer generation + streaming responses  
- **RRF (Reciprocal Rank Fusion)**: Hybrid retrieval fusion strategy (BM25 + vector)

## 🗂️ Notes on the dataset

- The PDF corpus was created by collecting public SF travel content (example: visiting pages and saving them as PDFs).
- For public repos, it’s safest to:
  - Include **derived artifacts** like `data/processed/chunks.jsonl`
  - Provide instructions so others can **rebuild the corpus themselves** (scrape/export their own PDFs) if needed








