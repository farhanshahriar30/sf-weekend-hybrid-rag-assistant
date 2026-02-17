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
