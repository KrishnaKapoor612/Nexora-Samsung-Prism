"""Local Okapi BM25 retrieval over complete supplied SIIS records."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.retrieval import STOP_WORDS

#Data Cleaning and Tokenization for BM25
_TOKEN_RE = re.compile(r"(?u)[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SKIP_KEYS = {"id", "url", "deeplink", "_readme"}
_BM25_STOP_WORDS = STOP_WORDS | {
    "do", "does", "did", "can", "could", "would", "should", "please", "help"
}


def tokenize(text: str) -> list[str]:
    """Keep device names, model numbers, hyphenated terms and feature names."""
    return [
        token.lower().replace("’", "'")
        for token in _TOKEN_RE.findall(text)
        if token.lower().replace("’", "'") not in _BM25_STOP_WORDS
    ]


def _string_values(value: Any, parent_key: str = "") -> list[str]:
    if parent_key.lower() in _SKIP_KEYS:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [part for key, child in value.items() for part in _string_values(child, str(key))]
    if isinstance(value, (list, tuple)):
        return [part for child in value for part in _string_values(child)]
    return []


def siis_searchable_text(row: dict[str, Any]) -> str:
    """Flatten all textual SIIS fields, including title and full solution text."""
    return " ".join(_string_values(row)).strip()


@dataclass(frozen=True)
class RetrievalCandidate:
    siis_id: str
    score: float
    searchable_text: str
    metadata: dict[str, Any]
    row: dict[str, Any]


class BM25SIISRetriever:
    """In-memory BM25 index; built once from the actual SIIS asset rows."""

    def __init__(self, rows: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._rows = rows
        self._texts = [siis_searchable_text(row) for row in rows]
        self._tokens = [tokenize(text) for text in self._texts]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / max(len(self._lengths), 1)
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        self._idf = {
            term: math.log(1 + (len(rows) - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search_bm25(self, query: str, top_k: int = 5) -> list[RetrievalCandidate]:
        if top_k <= 0 or not self._rows:
            return []
        terms = tokenize(query)
        if not terms:
            return []
        results: list[RetrievalCandidate] = []
        for index, row in enumerate(self._rows):
            tf = self._term_frequencies[index]
            length = self._lengths[index]
            score = 0.0
            for term in terms:
                frequency = tf.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self._average_length, 1.0)
                )
                score += self._idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1) / denominator
                )
            if not math.isfinite(score):
                score = 0.0
            siis = row.get("siis_response", {})
            results.append(
                RetrievalCandidate(
                    siis_id=str(row.get("id", index)),
                    score=score,
                    searchable_text=self._texts[index],
                    metadata={
                        "title": str(siis.get("title", "")) if isinstance(siis, dict) else "",
                        "original_query": str(row.get("original_query", "")),
                    },
                    row=row,
                )
            )
        results.sort(key=lambda candidate: (-candidate.score, candidate.siis_id))
        return results[:top_k]
