"""Configurable BM25 + semantic SIIS retrieval and candidate fusion."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

from app.bm25_retriever import BM25SIISRetriever, RetrievalCandidate
from app.config import AppAssets
from app.retrieval import GENERIC_MATCH_TERMS, normalize_text
from app.semantic_retriever import LocalSIISSemanticIndex, semantic_features

logger = logging.getLogger(__name__)
_GENERIC_RETRIEVAL_FEATURES = GENERIC_MATCH_TERMS | {
    "check", "open", "turn", "step", "use", "ensure", "verify", "tap",
    "navigate", "contact", "try", "press", "hold", "support", "troubleshoot",
}


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return min(high, max(low, value)) if math.isfinite(value) else default
    except ValueError:
        return default


def _env_int(name: str, default: int, low: int = 1, high: int = 50) -> int:
    try:
        return min(high, max(low, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def normalize_scores(scores: list[float]) -> list[float]:
    """Deterministically map finite scores to [0, 1], safe for edge cases."""
    if not scores:
        return []
    safe = [score if math.isfinite(score) else 0.0 for score in scores]
    low, high = min(safe), max(safe)
    if high <= 0:
        return [0.0] * len(safe)
    if math.isclose(high, low):
        return [1.0 if score > 0 else 0.0 for score in safe]
    return [(score - low) / (high - low) for score in safe]


@dataclass(frozen=True)
class HybridCandidate:
    candidate: RetrievalCandidate
    bm25_score: float
    semantic_score: float
    bm25_normalized: float
    semantic_normalized: float
    hybrid_score: float


class HybridSIISRetriever:
    """Build both local indices once and fuse their candidates per query."""

    def __init__(self, assets: AppAssets) -> None:
        self.bm25_top_k = _env_int("BM25_TOP_K", 5)
        self.semantic_top_k = _env_int("SEMANTIC_TOP_K", 5)
        self.final_top_k = _env_int("FINAL_TOP_K", 3)
        self.bm25_weight = _env_float("BM25_WEIGHT", 0.5, 0.0, 1.0)
        self.semantic_weight = _env_float("SEMANTIC_WEIGHT", 0.5, 0.0, 1.0)
        total_weight = self.bm25_weight + self.semantic_weight
        if total_weight == 0:
            self.bm25_weight = self.semantic_weight = 0.5
        else:
            self.bm25_weight /= total_weight
            self.semantic_weight /= total_weight
        self.min_confidence = _env_float("SIIS_MIN_CONFIDENCE", 0.25, 0.0, 1.0)
        self.min_bm25_score = _env_float("SIIS_BM25_MIN_SCORE", 1.0, 0.0, 1000.0)
        self.min_semantic_score = _env_float("SIIS_SEMANTIC_MIN_SCORE", 0.08, 0.0, 1.0)
        self.debug = os.getenv("SIIS_RETRIEVAL_DEBUG", "").lower() in {"1", "true", "yes"}
        self.bm25 = BM25SIISRetriever(assets.siis_rows)
        self.semantic = LocalSIISSemanticIndex(assets.siis_rows)

    @staticmethod
    def enrich_query(query: str) -> str:
        # Retain the raw text for phrase/concept matching, and append the
        # existing canonical normalization for aliases and stable token forms.
        return f"{query.strip()} {normalize_text(query)}".strip()

    def search(self, query: str) -> list[HybridCandidate]:
        enriched = self.enrich_query(query)
        lexical = self.bm25.search_bm25(enriched, self.bm25_top_k)
        semantic = self.semantic.search_semantic(enriched, self.semantic_top_k)
        lexical_norm = normalize_scores([candidate.score for candidate in lexical])
        semantic_norm = normalize_scores([candidate.score for candidate in semantic])
        by_id: dict[str, dict[str, Any]] = {}
        for candidate, score in zip(lexical, lexical_norm):
            by_id.setdefault(candidate.siis_id, {"candidate": candidate, "bm25": 0.0, "bm25_norm": 0.0, "semantic": 0.0, "semantic_norm": 0.0})
            by_id[candidate.siis_id].update(bm25=candidate.score, bm25_norm=score)
        for candidate, score in zip(semantic, semantic_norm):
            by_id.setdefault(candidate.siis_id, {"candidate": candidate, "bm25": 0.0, "bm25_norm": 0.0, "semantic": 0.0, "semantic_norm": 0.0})
            by_id[candidate.siis_id].update(candidate=candidate, semantic=candidate.score, semantic_norm=score)

        fused: list[HybridCandidate] = []
        query_features = set(semantic_features(enriched)) - _GENERIC_RETRIEVAL_FEATURES
        for values in by_id.values():
            bm25_score = values["bm25"]
            semantic_score = values["semantic"]
            if bm25_score < self.min_bm25_score and semantic_score < self.min_semantic_score:
                continue
            # A shared specific term or concept is still required. This guards
            # against candidates returned solely for generic words like screen.
            candidate_features = set(semantic_features(values["candidate"].searchable_text))
            if not (query_features & candidate_features):
                continue
            score = self.bm25_weight * values["bm25_norm"] + self.semantic_weight * values["semantic_norm"]
            if not math.isfinite(score) or score < self.min_confidence:
                continue
            fused.append(HybridCandidate(
                candidate=values["candidate"], bm25_score=bm25_score,
                semantic_score=semantic_score, bm25_normalized=values["bm25_norm"],
                semantic_normalized=values["semantic_norm"], hybrid_score=score,
            ))
        fused.sort(key=lambda item: (-item.hybrid_score, item.candidate.siis_id))
        fused = fused[:self.final_top_k]
        if self.debug:
            logger.debug(
                "SIIS retrieval query=%r enriched=%r bm25=%s bm25_normalized=%s semantic=%s semantic_normalized=%s fused=%s selected=%s confidence=%s no_match=%s",
                query, enriched,
                [(c.siis_id, round(c.score, 4)) for c in lexical],
                [(c.siis_id, round(score, 4)) for c, score in zip(lexical, lexical_norm)],
                [(c.siis_id, round(c.score, 4)) for c in semantic],
                [(c.siis_id, round(score, 4)) for c, score in zip(semantic, semantic_norm)],
                [(c.candidate.siis_id, round(c.bm25_normalized, 3), round(c.semantic_normalized, 3), round(c.hybrid_score, 3)) for c in fused],
                [c.candidate.siis_id for c in fused],
                fused[0].hybrid_score if fused else 0.0,
                not bool(fused),
            )
        return fused
