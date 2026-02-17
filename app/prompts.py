"""
prompts.py

Purpose:
- Centralize all prompt text + message-building logic.
- Keeps rag.py focused on retrieval + orchestration.
"""

from __future__ import annotations

from typing import Dict, List


def build_messages(
    question: str, context: str, history: List[Dict] | None = None
) -> List[Dict]:
    """
    Construct the messages we send to the LLM.

    Why:
    - Strong rules keep the model grounded:
      - Use ONLY context
      - Cite chunk numbers
      - Ask follow-up if missing info
    """
    system = (
        "You are a San Francisco weekend planning assistant for first-timers.\n"
        "Use ONLY the provided CONTEXT.\n"
        "If something is not in the context, say you don't know and ask a follow-up question.\n"
        "Cite supporting chunks using bracket citations like [1], [2].\n"
        "Be practical: itinerary bullets, neighborhoods, transit tips, and food suggestions.\n"
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Write an answer grounded in the context. Include citations like [1], [2]."
    )

    messages = [{"role": "system", "content": system}]

    # If we have chat history, insert it after the system message
    if history:
        # keep only last N messages to avoid blowing context window
        last = history[-12:]

        # only allow valid roles + non-empty content
        cleaned = [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in last
            if m.get("role") in {"user", "assistant"}
            and (m.get("content") or "").strip()
        ]
        messages.extend(cleaned)
