"""Catalog-only semantic mapping for troubleshooting actions.

HARD RULE (do not weaken this file without re-reading the spec): every
``DeeplinkMatch`` this module returns carries a ``deeplink`` string that is
copied verbatim from ``student_kit/deeplinks.json``. Nothing in this
module -- not the embedding index, not the dummy-positive fallback --
ever constructs, edits, or asks a model to produce a URI. The only free
text this module ever authors itself is the *description/message* pair
for the reserved ``bixby://dummy_positive`` placeholder, and that is done
with a plain regex/template, never an LLM call, exactly as
``deeplinks.json``'s own qna_description for that entry instructs.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from typing import Any

from app.config import AppAssets
from app.constants import DUMMY_POSITIVE_DEEPLINK
from app.embeddings import CatalogEmbeddingIndex, entry_search_text
from app.retrieval import content_tokens

# Matches phrases like "tap Display, and then tap Navigation bar" or
# "navigate to Settings > Accounts and backup" so the dummy-positive
# fallback can name a concrete-looking screen instead of a generic one.
SCREEN_MENTION_PATTERN = re.compile(
    r"\b(?:tap|open|select|go to)\s+([A-Z][A-Za-z0-9 &'/-]{2,40}?)"
    r"(?:\.|,|\s+and\s+then|\s+to\s|$)",
)
SETTINGS_ACTION_PATTERN = re.compile(r"\bsettings\b", re.IGNORECASE)
NAVIGATION_VERB_PATTERN = re.compile(
    r"\b(tap|open|navigate|select|go to)\b", re.IGNORECASE
)
WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")

# Below this blended score a catalog match is not trusted at all -- the
# action falls through to the dummy-positive check (or no deeplink).
# An embedding-only signal (no lexical overlap at all) needs a higher bar
# before we trust it, since it has no exact shared word to anchor on.
MIN_EMBEDDING_ONLY_SCORE = 0.45
# Balanced against the catalog's short action labels: require useful absolute
# similarity and separation from runner-up to avoid a plausible but wrong link.
MIN_CATALOG_MARGIN = float(os.getenv("DEEPLINK_MIN_MARGIN", "0.02"))
CANDIDATE_K = max(1, int(os.getenv("DEEPLINK_CANDIDATE_K", "5")))
MIN_CATALOG_SCORE = float(os.getenv("DEEPLINK_MIN_SCORE", "0.22"))

_ALIASES = {
    "enable": "turnon", "activate": "turnon", "turn": "turn",
    "disable": "turnoff", "deactivate": "turnoff",
    "screen": "display", "touchscreen": "touch", "reboot": "restart",
    "charging": "charge", "wifi": "wifi", "pair": "connect",
    "unpair": "disconnect", "battery": "battery", "saver": "saving",
}


def _tokens(text: str) -> set[str]:
    normalized_text = text.lower().replace("wi-fi", "wifi")
    normalized_text = re.sub(r"\bturn\s+on\b", "turnon", normalized_text)
    normalized_text = re.sub(r"\bturn\s+off\b", "turnoff", normalized_text)
    normalized_text = re.sub(r"\bbattery\s+saver\b", "power saving", normalized_text)
    words = re.findall(r"[a-z0-9]+", normalized_text)
    normalized: list[str] = []
    for word in words:
        # Treat toggle verbs as one intent without erasing on/off polarity.
        if word in {"enable", "activate"}:
            normalized.append("turnon")
        elif word in {"disable", "deactivate"}:
            normalized.append("turnoff")
        else:
            normalized.append(_ALIASES.get(word, word))
    return set(normalized)


@dataclass(frozen=True)
class DeeplinkMatch:
    entry: dict[str, Any]
    score: float
    is_dummy: bool = False


def _guess_screen_name(action_name: str, steps: list[str]) -> str:
    """Best-effort, regex-only guess at the concrete screen a step opens.

    This text is ONLY used to fill in the dummy-positive description/
    message -- never the deeplink URI itself -- and is built from the
    action's own step text, not invented.
    """
    joined = " ".join(steps)
    matches = SCREEN_MENTION_PATTERN.findall(joined)
    if matches:
        return matches[-1].strip()
    return action_name


def _clip_to_words(text: str, min_words: int = 5, max_words: int = 7) -> str:
    words = WORD_PATTERN.findall(text)
    if not words:
        words = ["open", "this", "settings", "screen"]
    if len(words) > max_words:
        words = words[:max_words]
    filler = ["settings", "screen", "options"]
    index = 0
    while len(words) < min_words:
        words.append(filler[index % len(filler)])
        index += 1
    return " ".join(words)


class DeeplinkMapper:
    def __init__(self, assets: AppAssets) -> None:
        all_entries = [
            entry for entry in assets.deeplinks if isinstance(entry.get("deeplink"), str)
        ]
        self._entries = [
            entry for entry in all_entries if entry["deeplink"] != DUMMY_POSITIVE_DEEPLINK
        ]
        self._dummy_entry = next(
            (entry for entry in all_entries if entry["deeplink"] == DUMMY_POSITIVE_DEEPLINK),
            None,
        )
        self._uris = frozenset(entry["deeplink"] for entry in all_entries)
        self._embedding_index = (
            CatalogEmbeddingIndex(self._entries) if self._entries else None
        )

    @staticmethod
    def _search_text(entry: dict[str, Any]) -> str:
        return entry_search_text(entry)

    def _lexical_score(
        self, query_tokens: set[str], entry: dict[str, Any], action_name: str
    ) -> float:
        fields = {
            "qna_description": 1.0,
            "message": 0.9,
            "description": 0.9,
            "validation": 0.55,
            "originalType": 0.25,
        }
        weighted_overlap = 0.0
        total_weight = 0.0
        for field, weight in fields.items():
            value = entry.get(field, "")
            if field == "validation" and isinstance(value, dict):
                value = value.get("key", "")
            metadata_tokens = _tokens(str(value))
            weighted_overlap += weight * len(query_tokens & metadata_tokens) / max(len(query_tokens), 1)
            total_weight += weight
        score = weighted_overlap / max(total_weight, 1e-9)
        if not score:
            return 0.0
        if action_name.lower() in self._search_text(entry).lower():
            score += 0.25
        return min(score, 1.0)

    def match(self, action_name: str, description: str, steps: list[str]) -> DeeplinkMatch | None:
        query_text = " ".join([action_name, description, *steps])
        query_tokens = _tokens(query_text)
        if not query_tokens or not self._entries:
            return self._match_dummy(action_name, steps)

        embedding_scores = (
            self._embedding_index.similarity_for_all(query_text)
            if self._embedding_index is not None
            else [0.0] * len(self._entries)
        )

        matches: list[DeeplinkMatch] = []
        for entry, embed_score in zip(self._entries, embedding_scores):
            lexical_score = self._lexical_score(query_tokens, entry, action_name)
            if lexical_score == 0.0 and embed_score < MIN_EMBEDDING_ONLY_SCORE:
                # No shared word AND weak vector similarity -> not a real
                # candidate. This guards against the embedding signal
                # alone ever pulling in an unrelated catalog row.
                continue
            # Lexical 0.50, TF-IDF 0.35, normalized intent 0.15. Intent is
            # lexical overlap after focused toggle/screen terminology mapping.
            # Intent score: bonus if step text contains settings navigation keywords
            step_text = " ".join([action_name] + steps).lower()
            nav_keywords = {"settings", "display", "battery", "wifi", "bluetooth", "navigation",
                           "brightness", "touch", "sensitivity", "sound", "notification",
                           "connection", "gesture", "general", "management", "apps", "storage",
                           "accounts", "backup", "software", "update", "reset", "power",
                           "mode", "accessibility", "privacy", "security", "lock"}
            entry_text = self._search_text(entry).lower()
            shared_nav = nav_keywords & _tokens(step_text) & _tokens(entry_text)
            intent_score = min(1.0, lexical_score + 0.15 * len(shared_nav))
            blended = 0.50 * lexical_score + 0.35 * embed_score + 0.15 * intent_score
            matches.append(DeeplinkMatch(entry=entry, score=min(blended, 1.0)))

        if not matches:
            return self._match_dummy(action_name, steps)

        matches.sort(key=lambda item: item.score, reverse=True)
        # Candidate retrieval limits reranking to the strongest initial matches.
        matches = matches[:CANDIDATE_K]
        best = matches[0]
        margin = best.score - (matches[1].score if len(matches) > 1 else 0.0)
        if (best.score < MIN_CATALOG_SCORE or margin < MIN_CATALOG_MARGIN
                or best.entry["deeplink"] not in self._uris):
            return self._match_dummy(action_name, steps)
        return best

    def _match_dummy(self, action_name: str, steps: list[str]) -> DeeplinkMatch | None:
        """Reserved catalog fallback for a real Settings screen with no
        dedicated catalog row (student_kit/deeplinks.json, id "DL-DUMMY").

        The deeplink URI is always the exact catalog string. Only the
        description/message are authored here, by template -- never by an
        LLM -- exactly as that entry's own qna_description instructs.
        """
        if self._dummy_entry is None:
            return None
        joined_steps = " ".join(steps)
        opens_settings_screen = bool(
            SETTINGS_ACTION_PATTERN.search(joined_steps)
            and NAVIGATION_VERB_PATTERN.search(joined_steps)
        )
        if not opens_settings_screen:
            return None
        screen = _guess_screen_name(action_name, steps)
        dummy_entry = {
            **self._dummy_entry,
            "description": _clip_to_words(f"Open {screen} settings screen"),
            "message": _clip_to_words(f"Go to {screen} in Settings"),
        }
        return DeeplinkMatch(entry=dummy_entry, score=0.30, is_dummy=True)

    @staticmethod
    def actionable_payload(match: DeeplinkMatch) -> dict[str, Any]:
        entry = match.entry
        return {
            key: entry[key]
            for key in ("deeplink", "description", "message", "classes", "originalType")
            if key in entry and entry[key] is not None
        }

    @staticmethod
    def validation_payload(match: DeeplinkMatch) -> dict[str, Any] | None:
        if match.is_dummy:
            # The generic placeholder has no paired validation deeplink --
            # never invent one.
            return None
        validation = match.entry.get("validation")
        if not isinstance(validation, dict) or not validation.get("deeplink"):
            return None
        payload = {"deeplink": validation["deeplink"], "key": validation.get("key", "")}
        for key in ("resultType", "condition", "value"):
            if validation.get(key) is not None:
                payload[key] = validation[key]
        return payload
