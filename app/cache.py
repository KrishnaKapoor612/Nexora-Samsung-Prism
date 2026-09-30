"""Thread-safe two-stage cache for validated troubleshooting responses."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from app.query_understanding import QueryFingerprint
from app.retrieval import content_tokens


@dataclass(frozen=True)
class CacheLookup:
    value: dict[str, Any]
    matched_key: str
    similarity: float
    compatibility: str


def _similarity(left: str, right: str) -> float:
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def fingerprints_compatible(
    current: QueryFingerprint, cached: QueryFingerprint
) -> bool:
    """Reject candidates that conflict on any known troubleshooting dimension."""
    strict_fields = (
        "device_type",
        "device_family",
        "problem_category",
        "symptom",
        "intent",
        "troubleshooting_context",
        "physical_damage",
        "power_state",
    )
    for field in strict_fields:
        current_value = getattr(current, field)
        cached_value = getattr(cached, field)
        if current_value is not None and cached_value is not None:
            if current_value != cached_value:
                return False
    return True


class ResponseCache:
    """Two-stage cache: lexical candidate search followed by fingerprint gating."""

    def __init__(self, candidate_threshold: float = 0.15) -> None:
        self._values: dict[str, dict[str, Any]] = {}
        self._candidate_threshold = candidate_threshold
        self._lock = Lock()

    def get(
        self,
        key: str,
        fingerprint: QueryFingerprint | None = None,
    ) -> CacheLookup | None:
        with self._lock:
            exact = self._values.get(key)
            if exact is not None:
                if fingerprint is None or self._compatible(exact, fingerprint):
                    return CacheLookup(
                        value=dict(exact["value"]),
                        matched_key=key,
                        similarity=1.0,
                        compatibility="exact",
                    )

            if fingerprint is None:
                return None

            candidates: list[tuple[float, str, dict[str, Any]]] = []
            for candidate_key, entry in self._values.items():
                score = _similarity(key, candidate_key)
                if score >= self._candidate_threshold:
                    candidates.append((score, candidate_key, entry))

            candidates.sort(key=lambda item: item[0], reverse=True)
            for score, candidate_key, entry in candidates:
                if self._compatible(entry, fingerprint):
                    return CacheLookup(
                        value=dict(entry["value"]),
                        matched_key=candidate_key,
                        similarity=score,
                        compatibility="compatible",
                    )
            return None

    def put(
        self,
        key: str,
        value: dict[str, Any],
        fingerprint: QueryFingerprint | None = None,
        cacheable: bool = True,
    ) -> None:
        if not cacheable:
            return
        with self._lock:
            self._values[key] = {
                "value": dict(value),
                "fingerprint": fingerprint.as_dict() if fingerprint else None,
            }

    @staticmethod
    def _compatible(
        entry: dict[str, Any], current: QueryFingerprint
    ) -> bool:
        stored = entry.get("fingerprint")
        if not isinstance(stored, dict):
            return False
        try:
            cached = QueryFingerprint(**stored)
        except TypeError:
            return False
        return fingerprints_compatible(current, cached)

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)
