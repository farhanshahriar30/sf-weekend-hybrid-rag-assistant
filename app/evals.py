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
