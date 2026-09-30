"""Grounded evidence providers used by the troubleshooting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.config import AppAssets
from app.query_understanding import fingerprint_query
from app.hybrid_retriever import HybridSIISRetriever
from app.retrieval import content_tokens


@dataclass(frozen=True)
class Evidence:
    """A retrieved source document, never an LLM-generated answer."""

    source: str
    title: str
    content: str
    score: float
    evidence_id: str | None = None
    url: str | None = None
    source_type: str = "local"
    reliability_score: float = 0.0


class EvidenceProvider(Protocol):
    def search(self, query: str, limit: int = 5) -> list[Evidence]:
        """Return grounded source evidence candidates."""


class LocalSIISProvider:
    """Retrieve only the final hybrid-selected evidence from the SIIS corpus."""

    def __init__(self, assets: AppAssets) -> None:
        self._retriever = HybridSIISRetriever(assets)

    def search(self, query: str, limit: int = 5) -> list[Evidence]:
        matches = self._retriever.search(query)[:limit]
        return [
            Evidence(
                source="local_siis",
                title=str(match.candidate.metadata.get("title", "")),
                content=str(match.candidate.row.get("siis_response", {}).get("content", "")),
                score=0.65 + 0.35 * match.hybrid_score,
                evidence_id=match.candidate.siis_id,
                source_type="local",
                reliability_score=0.9,
            )
            for match in matches
        ]


class StaticEvidenceProvider:
    """Small deterministic provider for tests and local demonstrations."""

    def __init__(self, evidence: Evidence | None = None) -> None:
        self._evidence = evidence

    def search(self, _query: str, limit: int = 5) -> list[Evidence]:
        _ = _query
        return [self._evidence][:limit] if self._evidence is not None else []


class EvidenceProviderChain:
    """Try providers in order and accept the first grounded evidence result."""

    def __init__(self, providers: list[EvidenceProvider]) -> None:
        if not providers:
            raise ValueError("At least one evidence provider is required")
        self._providers = tuple(providers)

    def search(self, query: str, limit: int = 5) -> list[Evidence]:
        for provider in self._providers:
            candidates = [
                evidence
                for evidence in provider.search(query, limit=limit)
                if evidence.content.strip()
            ]
            selected = select_evidence(query, candidates)
            if selected is not None:
                return [selected]
        return []


def evidence_compatibility(
    query: str, evidence: Evidence
) -> float:
    """Score evidence using lexical overlap and query/evidence context signals."""
    query_tokens = content_tokens(query)
    evidence_tokens = content_tokens(f"{evidence.title} {evidence.content}")
    overlap = len(query_tokens & evidence_tokens) / max(len(query_tokens), 1)
    query_context = fingerprint_query(query)
    evidence_context = fingerprint_query(f"{evidence.title} {evidence.content}")
    matches = 0
    known = 0
    for field in ("problem_category", "symptom", "intent", "troubleshooting_context"):
        left = getattr(query_context, field)
        right = getattr(evidence_context, field)
        if left is not None:
            known += 1
            if left == right:
                matches += 1
    context_score = matches / known if known else 0.0
    instruction_score = 1.0 if any(
        marker in evidence.content.lower()
        for marker in ("step 1", "navigate", "tap ", "press ", "contact ")
    ) else 0.0
    return min(
        1.0,
        0.45 * evidence.score
        + 0.30 * overlap
        + 0.15 * context_score
        + 0.10 * instruction_score
        + 0.05 * evidence.reliability_score,
    )


def select_evidence(
    query: str,
    candidates: list[Evidence],
    threshold: float = 0.50,
) -> Evidence | None:
    """Select the highest compatible candidate or return no evidence.

    Category guard: when the query has a known problem_category, local SIIS
    candidates whose fingerprint category strongly conflicts are skipped so
    that cross-category false positives (e.g. display records scoring high
    on a charging query because both mention 'charge') do not block the
    fallback evidence chain.
    """
    query_fp = fingerprint_query(query)

    def _category_ok(candidate: Evidence) -> bool:
        """Return False only for local SIIS evidence with a clear category mismatch."""
        if candidate.source_type != "local":
            return True  # Non-SIIS evidence is never category-filtered here
        if query_fp.problem_category is None:
            return True  # No query category signal — keep all candidates
        # High-confidence retrievals are trusted regardless of category
        if candidate.score >= 0.80:
            return True
        ev_fp = fingerprint_query(f"{candidate.title}\n{candidate.content[:400]}")
        if ev_fp.problem_category is None:
            return True  # Evidence has no category signal — keep it
        return ev_fp.problem_category == query_fp.problem_category

    ranked = sorted(
        [
            candidate
            for candidate in candidates
            if (candidate.source_type != "local" or candidate.score >= 0.65)
            and _category_ok(candidate)
        ],
        key=lambda item: evidence_compatibility(query, item),
        reverse=True,
    )
    if not ranked:
        return None
    best = ranked[0]
    return best if evidence_compatibility(query, best) >= threshold else None
