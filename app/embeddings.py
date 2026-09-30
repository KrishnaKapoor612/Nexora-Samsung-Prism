"""In-process TF-IDF vector index used only to *rank* catalog entries.

Design constraints (non-negotiable, see team.md / Provided_Pdfs spec):

1. This module never calls an external embedding API and never generates
   a deeplink URI. It only re-scores the entries that already exist in
   ``student_kit/deeplinks.json``. Catalog integrity is untouched -- the
   ranked "winner" is still just one of the 578 verbatim catalog rows.
2. No new heavy dependency (no sentence-transformers, no torch, no
   network call at request time). TF-IDF + cosine similarity is a
   classic, fully explainable "dense-style" vector-space signal that can
   be built and queried in pure Python, in-process, offline -- consistent
   with the rest of this codebase's offline-safe philosophy.

Why this helps: the previous lexical-only matcher (`len(overlap) /
len(query_tokens)`) divides by the *raw* size of the query text, so a
real match gets diluted whenever the action has a long step list full of
generic instructional words ("navigate", "open", "tap"...). TF-IDF
down-weights those common words automatically (they appear in almost
every catalog entry, so they get a low IDF) and up-weights the rare,
specific words that actually identify a screen -- closing the gap the
spec calls "Screen Resolution Accuracy" without touching catalog data.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.retrieval import content_tokens


def entry_search_text(entry: dict[str, Any]) -> str:
    """The same searchable text fields the lexical matcher already uses."""
    validation = entry.get("validation")
    validation_key = validation.get("key", "") if isinstance(validation, dict) else ""
    return (
        " ".join(
            str(entry.get(field, ""))
            for field in ("description", "message", "qna_description", "originalType")
        )
        + f" {validation_key}"
    )


@dataclass(frozen=True)
class _Vector:
    weights: dict[str, float]
    norm: float


def _vectorize(tokens: list[str], idf: dict[str, float]) -> _Vector:
    if not tokens:
        return _Vector(weights={}, norm=0.0)
    counts = Counter(tokens)
    total = len(tokens)
    weights = {
        token: (count / total) * idf.get(token, 1.0) for token, count in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in weights.values()))
    return _Vector(weights=weights, norm=norm)


class CatalogEmbeddingIndex:
    """Bag-of-words TF-IDF index built once over the deeplink catalog text."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._entries = entries
        tokenized = [list(content_tokens(entry_search_text(entry))) for entry in entries]

        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))

        num_documents = max(len(tokenized), 1)
        self._idf: dict[str, float] = {
            term: math.log((1 + num_documents) / (1 + freq)) + 1.0
            for term, freq in document_frequency.items()
        }
        self._vectors = [_vectorize(tokens, self._idf) for tokens in tokenized]

    def similarity_for_all(self, text: str) -> list[float]:
        """Cosine similarity between `text` and every catalog entry, in order.

        Returns one float per entry passed to the constructor, so callers
        can zip() it back against their own entry list.
        """
        query_tokens = list(content_tokens(text))
        if not query_tokens:
            return [0.0] * len(self._entries)
        query_vector = _vectorize(query_tokens, self._idf)
        if query_vector.norm == 0.0:
            return [0.0] * len(self._entries)

        scores: list[float] = []
        for vector in self._vectors:
            if vector.norm == 0.0:
                scores.append(0.0)
                continue
            dot = sum(
                weight * vector.weights.get(term, 0.0)
                for term, weight in query_vector.weights.items()
            )
            scores.append(dot / (query_vector.norm * vector.norm))
        return scores
