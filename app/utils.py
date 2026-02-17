"""
utils.py

Small reusable helpers:
- citation parsing
- text cleanup for PDF-ish chunks
"""

from __future__ import annotations

import re
from typing import Dict, List


_CITE_RE = re.compile(r"\[(\d+)\]")

# Optional: strip common PDF junk and collapse whitespace
_PUA_RE = re.compile(r"[\uE000-\uF8FF]")  # Private Use Area glyphs
_BAD_RE = re.compile(r"[\uFFFC\u200B\u200C\u200D]")  # object replacement + zero-width


def clean_snippet(text: str, limit: int) -> str:
    """
    Light cleanup for chunk text shown to the LLM/UI.
    """
    if not text:
        return ""
    s = str(text).replace("\x00", " ")
    s = _PUA_RE.sub(" ", s)
    s = _BAD_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()  # collapse whitespace/newlines
    return s[:limit]


def filter_citations_used(answer_text: str, citations: List[Dict]) -> List[Dict]:
    """
    Keep only citations whose [n] appears in the model's answer text.
    """
    used = set(int(n) for n in _CITE_RE.findall(answer_text or ""))
    return [c for c in citations if c.get("n") in used]
