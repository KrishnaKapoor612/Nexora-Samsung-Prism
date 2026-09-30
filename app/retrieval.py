"""Deterministic retrieval over the supplied SIIS reference responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from app.config import AppAssets


STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "for",
        "how",
        "i",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "so",
        "the",
        "to",
        "when",
        "with",
    }
)

TERM_ALIASES = {
    # Display
    "black": "blank",
    "flickering": "flicker",
    "flickers": "flicker",
    # Canonicalized to "flicker", not "flash": every downstream decision
    # (intent, troubleshooting_context) already treats flash/flicker as the
    # same symptom, but the raw `symptom` field used exact string equality
    # in validate_context_consistency, so content mentioning both words
    # (e.g. "flickers or flashes") got fingerprinted inconsistently
    # depending on which word appeared first. One canonical form removes
    # the false mismatch.
    "flashing": "flicker",
    "flashes": "flicker",
    "flash": "flicker",
    "display": "screen",
    "touchscreen": "touch",
    "unresponsive": "respond",
    "responsiveness": "respond",
    # Connectivity
    "disconnects": "disconnect",
    "disconnecting": "disconnect",
    "disconnected": "disconnect",
    "bluetooth": "bt",
    "wi-fi": "wifi",
    "pairing": "pair",
    "paired": "pair",
    # App / software
    "crashes": "crash",
    "crashing": "crash",
    "crashed": "crash",
    "freezes": "freeze",
    "freezing": "freeze",
    "frozen": "freeze",
    "hanging": "hang",
    "closing": "close",
    # Battery / charging
    "draining": "drain",
    "drains": "drain",
    "charging": "charge",
    "uncharged": "charge",
    # Thermal
    "overheating": "overheat",
    "overheated": "overheat",
    "heatup": "overheat",
    # Storage
    "storage": "memory",
    "insufficient": "full",
    # Audio
    "speakers": "speaker",
    "microphones": "microphone",
    "earphones": "earphone",
    "headphones": "headphone",
    # Camera
    "blurry": "blur",
    "blurred": "blur",
}

GENERIC_MATCH_TERMS = frozenset(
    {
        "screen",
        "phone",
        "samsung",
        "device",
        "galaxy",
        "mobile",
        "tablet",
        "issue",
        "problem",
        "smartphone",
        "not",
        "working",
    }
)


def normalize_text(text: str) -> str:
    """Normalize user and SIIS text into comparable tokens."""
    lowered = text.lower().replace("’", "'")
    words = re.findall(r"[a-z0-9]+", lowered)
    normalized = [TERM_ALIASES.get(word, word) for word in words]
    return " ".join(normalized)


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 2 and token not in STOP_WORDS
    }


@dataclass(frozen=True)
class RetrievedEvidence:
    row: dict[str, Any]
    score: float
    overlapping_terms: frozenset[str]

    @property
    def title(self) -> str:
        return str(self.row["siis_response"]["title"])

    @property
    def content(self) -> str:
        return str(self.row["siis_response"]["content"])


class SIISRetriever:
    """Small in-memory retriever suitable for the bundled reference corpus."""

    def __init__(self, assets: AppAssets, relevance_threshold: float = 0.20) -> None:
        self.relevance_threshold = relevance_threshold
        self._documents: list[tuple[dict[str, Any], set[str], str]] = []
        for row in assets.siis_rows:
            source_query = str(row.get("original_query", ""))
            siis = row.get("siis_response", {})
            title = str(siis.get("title", ""))
            content = str(siis.get("content", ""))
            searchable = f"{source_query} {title} {content}"
            self._documents.append((row, content_tokens(searchable), normalize_text(searchable)))

    def search(self, query: str, limit: int = 3) -> list[RetrievedEvidence]:
        query_tokens = content_tokens(query)
        if not query_tokens:
            return []

        scored: list[RetrievedEvidence] = []
        normalized_query = normalize_text(query)
        for row, document_tokens, normalized_document in self._documents:
            overlap = query_tokens & document_tokens
            token_score = len(overlap) / len(query_tokens)
            similarity = SequenceMatcher(
                None, normalized_query, normalized_document
            ).ratio()
            score = min(1.0, 0.8 * token_score + 0.2 * similarity)
            scored.append(
                RetrievedEvidence(
                    row=row,
                    score=score,
                    overlapping_terms=frozenset(overlap),
                )
            )

        scored.sort(key=lambda evidence: evidence.score, reverse=True)
        return scored[:limit]

    def relevant(self, query: str, limit: int = 3) -> list[RetrievedEvidence]:
        """Filter to candidates that clear both the score bar and a
        false-positive guard: at least one *specific* shared term is
        required, not just a generic word like "screen" or "phone".

        This directly implements the pitfall the team called out:
        "Do not match only on one common word such as screen, phone, or
        Samsung." A query and a document sharing only generic vocabulary
        is not evidence of a real match. Complements (does not replace)
        the category guard in app/evidence.py::select_evidence -- that
        one rejects cross-category evidence, this one rejects same-
        category evidence that only overlaps on filler words.
        """
        results: list[RetrievedEvidence] = []
        for evidence in self.search(query, limit=limit):
            if evidence.score < self.relevance_threshold:
                continue
            specific_terms = evidence.overlapping_terms - GENERIC_MATCH_TERMS
            if not specific_terms:
                continue
            results.append(evidence)
        return results