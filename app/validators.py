"""Programmatic safety and contract validation."""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from app.constants import DUMMY_POSITIVE_DEEPLINK
from app.models import ActionCategory, Action, TroubleshootingResponse
from app.query_understanding import QueryFingerprint, fingerprint_query
from app.retrieval import content_tokens


URL_PATTERN = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
GENERIC_LINK_TERMS = frozenset(
    {
        "device",
        "settings",
        "setting",
        "open",
        "view",
        "page",
        "tap",
        "select",
        "use",
        "control",
        "option",
    }
)


class ResponseValidationError(ValueError):
    """Raised when a generated response violates a safety or contract rule."""


def validate_no_urls(value: Any, path: str = "response") -> None:
    if isinstance(value, str):
        if URL_PATTERN.search(value):
            raise ResponseValidationError(f"URL leakage in {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            validate_no_urls(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_no_urls(item, f"{path}[{index}]")


def validate_catalog_links(
    response: TroubleshootingResponse, catalog_uris: frozenset[str]
) -> None:
    for goal in response.contexts:
        for action in goal.actions:
            for group in action.stepGroups:
                for link in (group.actionableDeeplink, group.validationDeeplink):
                    if link is not None and link.deeplink not in catalog_uris:
                        raise ResponseValidationError(
                            f"Deeplink is not in catalog: {link.deeplink}"
                        )


def _normalized_action_signature(action: Action) -> tuple[str, ...]:
    text = " ".join(
        [action.actionName]
        + [step for group in action.stepGroups for step in group.steps]
    )
    return tuple(sorted(content_tokens(text)))


def validate_duplicate_actions(response: TroubleshootingResponse) -> None:
    for goal in response.contexts:
        signatures: set[tuple[str, ...]] = set()
        for action in goal.actions:
            signature = _normalized_action_signature(action)
            if signature in signatures:
                raise ResponseValidationError(
                    f"Duplicate action in goal '{goal.title}': {action.actionName}"
                )
            signatures.add(signature)


def _catalog_entry_by_uri(
    catalog_entries: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    actionable: dict[str, dict[str, Any]] = {}
    validation: dict[str, dict[str, Any]] = {}
    for entry in catalog_entries:
        action_uri = entry.get("deeplink")
        if isinstance(action_uri, str):
            actionable[action_uri] = entry
        validation_data = entry.get("validation")
        if isinstance(validation_data, dict):
            validation_uri = validation_data.get("deeplink")
            if isinstance(validation_uri, str):
                validation[validation_uri] = entry
    return actionable, validation


def validate_deeplink_consistency(
    response: TroubleshootingResponse,
    catalog_entries: list[dict[str, Any]],
) -> None:
    actionable_by_uri, validation_by_uri = _catalog_entry_by_uri(catalog_entries)
    for goal in response.contexts:
        for action in goal.actions:
            action_tokens = content_tokens(
                " ".join(
                    [action.actionName, action.description]
                    + [step for group in action.stepGroups for step in group.steps]
                )
            )
            for group in action.stepGroups:
                actionable = group.actionableDeeplink
                validation = group.validationDeeplink
                if actionable is None:
                    if validation is not None:
                        raise ResponseValidationError(
                            "Validation deeplink cannot exist without actionable deeplink"
                        )
                    continue
                entry = actionable_by_uri.get(actionable.deeplink)
                if entry is None:
                    raise ResponseValidationError(
                        f"Missing catalog metadata for deeplink: {actionable.deeplink}"
                    )
                if actionable.deeplink == DUMMY_POSITIVE_DEEPLINK:
                    # The reserved placeholder is, by design, a generic
                    # catalog row that can legitimately back many
                    # different Settings screens. It is exempt from the
                    # metadata-overlap check below (the catalog's own
                    # description is deliberately generic), but the
                    # response's own description must still be the
                    # action-specific 5-7 word phrase the catalog's
                    # qna_description for this entry requires -- so a
                    # lazy or unchanged generic description still fails.
                    word_count = len(actionable.description.split())
                    if not 5 <= word_count <= 7:
                        raise ResponseValidationError(
                            "dummy_positive actionableDeeplink.description must be "
                            "5 to 7 words naming the concrete screen, per the "
                            "catalog's own instruction for that entry"
                        )
                else:
                    metadata = " ".join(
                        str(entry.get(field, ""))
                        for field in ("description", "message", "qna_description")
                    )
                    meaningful_action_tokens = action_tokens - GENERIC_LINK_TERMS
                    meaningful_metadata_tokens = (
                        content_tokens(metadata) - GENERIC_LINK_TERMS
                    )
                    if not meaningful_action_tokens & meaningful_metadata_tokens:
                        raise ResponseValidationError(
                            f"Deeplink does not match action: {action.actionName}"
                        )
                if validation is not None:
                    # Compare by the matched entry's OWN validation URI, not
                    # object identity against a separately-built lookup dict.
                    # 145 catalog entries share their validation deeplink
                    # with at least one other actionable entry (e.g. two
                    # different toggles both checked by the same "is this
                    # setting on?" validator), so a last-write-wins
                    # {validation_uri: entry} map picks an arbitrary owner
                    # and can reject a perfectly legitimate, catalog-correct
                    # pairing. Checking the entry's own recorded validation
                    # URI is correct regardless of how many other entries
                    # happen to share it.
                    entry_validation = entry.get("validation")
                    entry_validation_uri = (
                        entry_validation.get("deeplink")
                        if isinstance(entry_validation, dict)
                        else None
                    )
                    if entry_validation_uri != validation.deeplink:
                        raise ResponseValidationError(
                            f"Actionable and validation deeplinks do not belong together: "
                            f"{actionable.deeplink}"
                        )


def _fingerprint_text(response: TroubleshootingResponse) -> str:
    parts: list[str] = []
    for goal in response.contexts:
        parts.append(goal.title)
        parts.append(goal.goal)
        for action in goal.actions:
            parts.extend([action.actionName, action.description])
            for group in action.stepGroups:
                parts.extend(group.steps)
    return " ".join(parts)


def validate_context_consistency(
    response: TroubleshootingResponse,
    query_fingerprint: QueryFingerprint,
) -> None:
    if not response.contexts:
        return
    response_fingerprint = fingerprint_query(_fingerprint_text(response))
    for field in (
        "problem_category",
        "symptom",
        "intent",
        "troubleshooting_context",
    ):
        query_value = getattr(query_fingerprint, field)
        response_value = getattr(response_fingerprint, field)
        if field == "troubleshooting_context" and query_fingerprint.symptom is None:
            continue
        if query_value is not None and response_value is not None:
            if query_value != response_value:
                raise ResponseValidationError(
                    f"Response context conflicts on {field}: "
                    f"{query_value!r} != {response_value!r}"
                )


def validate_grounding(
    response: TroubleshootingResponse, evidence_text: str
) -> None:
    evidence_tokens = content_tokens(evidence_text)
    for goal in response.contexts:
        for action in goal.actions:
            for group in action.stepGroups:
                for step in group.steps:
                    step_tokens = content_tokens(step)
                    if step_tokens and not (step_tokens & evidence_tokens):
                        raise ResponseValidationError(
                            f"Step is not grounded in SIIS evidence: {step}"
                        )


def validate_action_order(response: TroubleshootingResponse) -> None:
    category_rank = {
        ActionCategory.auto: 0,
        ActionCategory.manual: 1,
        ActionCategory.critical: 2,
    }
    for goal in response.contexts:
        ranks = [category_rank[action.category] for action in goal.actions]
        if ranks != sorted(ranks):
            raise ResponseValidationError("Actions are not ordered safely")


def validate_response(
    response: TroubleshootingResponse,
    catalog_uris: frozenset[str],
    evidence_text: str | None = None,
    catalog_entries: list[dict[str, Any]] | None = None,
    query_fingerprint: QueryFingerprint | None = None,
) -> TroubleshootingResponse:
    try:
        try:
            validated = TroubleshootingResponse.model_validate(response.model_dump())
        except Exception as exc:
            print("VALIDATION ERROR CAUGHT:", repr(exc))
            raise ResponseValidationError(str(exc)) from exc
        validate_no_urls(validated.model_dump())
        validate_catalog_links(validated, catalog_uris)
        validate_duplicate_actions(validated)
        if catalog_entries is not None:
            validate_deeplink_consistency(validated, catalog_entries)
        validate_action_order(validated)
        if query_fingerprint is not None:
            validate_context_consistency(validated, query_fingerprint)
        if evidence_text is not None:
            validate_grounding(validated, evidence_text)
        return validated
    except ResponseValidationError as e:
        print("VALIDATION ERROR (CUSTOM):", repr(e))
        raise