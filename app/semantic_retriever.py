"""Offline semantic-style retrieval for SIIS using concept-normalized vectors.

No pretrained embedding model is present in the project, and downloading one
would add network/model setup requirements. This deterministic sparse vector
model maps common paraphrases to shared concepts, then uses corpus-fitted TF-IDF
weights and cosine similarity. It is a replaceable local embedding interface,
not a pretrained neural embedding model.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from app.bm25_retriever import RetrievalCandidate, siis_searchable_text, tokenize
from app.retrieval import STOP_WORDS


_PHRASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:won't|will not|does not|doesn't|cannot|can't)\s+(?:start|boot|power on|turn on)\b"), "power_on_failure"),
    (re.compile(r"\b(?:won't|will not|does not|doesn't)\s+power\s+up\b"), "power_on_failure"),
    (re.compile(r"\b(?:blank|black|no image|nothing displayed)\b"), "display_not_visible"),
    (re.compile(r"\b(?:flicker|flickering|flashing|flashes)\b"), "display_flicker"),
    (re.compile(r"\b(?:laggy|delayed|slow response|not responsive)\b"), "touch_unresponsive"),
)
_CONCEPT_GROUPS = (
    {"phone", "smartphone", "handset"},
    {"tablet", "tab"},
    {"restart", "reboot", "restarting", "rebooting"},
    {"wifi", "wi-fi", "wireless"},
    {"bluetooth", "bt"},
)
_SEMANTIC_STOP_WORDS = STOP_WORDS | {
    "do", "does", "did", "can", "could", "would", "should", "please", "help"
}
_CONCEPT_MAP = {word: f"concept_{i}" for i, group in enumerate(_CONCEPT_GROUPS) for word in group}


def semantic_features(text: str) -> list[str]:
    lowered = text.lower().replace("’", "'")
    features = [token for token in tokenize(lowered) if token not in _SEMANTIC_STOP_WORDS]
    features.extend(f"phrase_{concept}" for pattern, concept in _PHRASES if pattern.search(lowered))
    features.extend(
        _CONCEPT_MAP[token]
        for token in tokenize(lowered)
        if token in _CONCEPT_MAP
    )
    return features


class LocalSIISSemanticIndex:
    """Corpus-specific TF-IDF concept vectors with cosine similarity."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self._texts = [siis_searchable_text(row) for row in rows]
        tokenized = [semantic_features(text) for text in self._texts]
        document_frequency: Counter[str] = Counter()
        for features in tokenized:
            document_frequency.update(set(features))
        count = len(rows)
        self._idf = {
            term: math.log((1 + count) / (1 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self._vectors = [self._vector(features) for features in tokenized]

    def _vector(self, features: list[str]) -> dict[str, float]:
        counts = Counter(features)
        total = max(sum(counts.values()), 1)
        return {
            feature: (frequency / total) * self._idf.get(feature, 1.0)
            for feature, frequency in counts.items()
        }

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        dot = sum(value * right.get(feature, 0.0) for feature, value in left.items())
        score = dot / (left_norm * right_norm)
        return score if math.isfinite(score) else 0.0

    def search_semantic(self, query: str, top_k: int = 5) -> list[RetrievalCandidate]:
        if top_k <= 0 or not self._rows:
            return []
        features = semantic_features(query)
        if not features:
            return []
        query_vector = self._vector(features)
        results: list[RetrievalCandidate] = []
        for index, row in enumerate(self._rows):
            score = self._cosine(query_vector, self._vectors[index])
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
