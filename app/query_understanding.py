"""Conservative query normalization and compatibility fingerprinting."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

from app.retrieval import normalize_text


@dataclass(frozen=True)
class QueryFingerprint:
    """Stable fields used to decide whether two troubleshooting queries are compatible."""

    device_type: str | None
    device_family: str | None
    problem_category: str | None
    symptom: str | None
    intent: str | None
    troubleshooting_context: str | None
    physical_damage: bool | None
    power_state: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_match(patterns: tuple[str, ...], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower() if match.lastindex else match.group(0).lower()
    return None


def _device_family(text: str) -> str | None:
    match = re.search(
        r"\b(galaxy\s+(?:[a-z]\d+\w*|s\d+\w*|z\s*(?:flip|fold)\s*\d*\w*))\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1).lower()).strip()
    if re.search(r"\btablet\b", text, re.IGNORECASE):
        return "tablet"
    if re.search(r"\bphone\b|\bmobile\b", text, re.IGNORECASE):
        return "phone"
    return None


def _device_type(text: str, family: str | None) -> str | None:
    if re.search(r"\b(tv|television)\b", text, re.IGNORECASE):
        return "tv"
    has_tablet = bool(re.search(r"\btablet\b", text, re.IGNORECASE))
    has_phone = bool(
        re.search(r"\b(phone|mobile|galaxy|smartphone)\b", text, re.IGNORECASE)
    )
    if has_tablet and has_phone:
        return None
    if has_tablet or family == "tablet":
        return "tablet"
    if has_phone:
        return "phone"
    return None


def _problem_category(text: str) -> str | None:
    """Map normalized query text to a problem category. Order matters: more specific first."""
    # Email / messaging — check before display so "email screen blank" → email
    if re.search(r"\b(email|gmail|message|sms)\b", text, re.IGNORECASE):
        return "email"
    # Display — check before everything because "black screen"/"touch" are unambiguous
    if re.search(
        r"\b(screen|display|blank|black|white|flicker|flash|crack|touch)\b",
        text,
        re.IGNORECASE,
    ):
        return "display"
    # Thermal — check before battery so "overheating while charging" → thermal not battery
    if re.search(
        r"\b(heat|overheat|overheating|hot|warm|temperature)\b",
        text,
        re.IGNORECASE,
    ):
        return "thermal"
    # Performance — check before battery so "slow/laggy" isn't confused with battery drain
    if re.search(r"\b(slow|lag|performance)\b", text, re.IGNORECASE):
        return "performance"
    # Battery / charging
    if re.search(r"\b(battery|charge|charging|drain)\b", text, re.IGNORECASE):
        return "battery"
    # Camera
    if re.search(r"\b(camera|photo|video|selfie|zoom)\b", text, re.IGNORECASE):
        return "camera"
    # Connectivity — Wi-Fi, Bluetooth, mobile data, hotspot, signal
    if re.search(
        r"\b(wifi|wi-fi|wi\s+fi|bluetooth|hotspot|mobile\s+data|signal|"
        r"internet|network|connect|disconnect|pair)\b",
        text,
        re.IGNORECASE,
    ):
        return "connectivity"
    # Storage — check before app so "tap Apps" inside storage steps doesn't win
    if re.search(
        r"\b(storage|memory|space|full|insufficient|out\s+of\s+space)\b",
        text,
        re.IGNORECASE,
    ):
        return "storage"
    # App / software crashes
    if re.search(
        r"\b(app|application|crash|crashing|force\s+close|not\s+responding|"
        r"freeze|frozen|hang)\b",
        text,
        re.IGNORECASE,
    ):
        return "app"
    # Audio
    if re.search(
        r"\b(sound|audio|speaker|microphone|volume|mute|earphone|headphone)\b",
        text,
        re.IGNORECASE,
    ):
        return "audio"
    return None


def _symptom(text: str, category: str | None) -> str | None:
    """Extract a fine-grained symptom token from the query text."""
    # Display symptoms
    display_symptom = _first_match(
        (
            r"\b(black|blank|white|flicker|flash|crack(?:ed)?|lag(?:gy)?|"
            r"unresponsive|delay(?:ed)?|distort(?:ed)?)\b",
        ),
        text,
    )
    if display_symptom:
        return display_symptom

    # Connectivity symptoms
    if category == "connectivity":
        if re.search(r"\b(disconnect|drop|lost|unstable|weak|slow)\b", text, re.IGNORECASE):
            return "disconnect"
        if re.search(r"\b(pair|pair\s+fail|cannot\s+pair|won.t\s+pair)\b", text, re.IGNORECASE):
            return "pair_fail"
        if re.search(r"\b(not\s+found|not\s+detect|invisible)\b", text, re.IGNORECASE):
            return "not_detected"
        return "no_connection"

    # App symptoms
    if category == "app":
        if re.search(r"\b(crash|crashing|force\s+close)\b", text, re.IGNORECASE):
            return "crash"
        if re.search(r"\b(freeze|frozen|hang|not\s+respond)\b", text, re.IGNORECASE):
            return "freeze"
        return "malfunction"

    # Thermal symptoms
    if category == "thermal":
        return "overheating"

    # Storage symptoms
    if category == "storage":
        if re.search(r"\b(full|insufficient|out\s+of\s+space)\b", text, re.IGNORECASE):
            return "storage_full"
        return "low_storage"

    # Audio symptoms
    if category == "audio":
        if re.search(r"\b(no\s+sound|silent|mute)\b", text, re.IGNORECASE):
            return "no_sound"
        if re.search(r"\b(distort|crackling|static)\b", text, re.IGNORECASE):
            return "distorted"
        return "audio_issue"

    # Battery symptoms
    if category == "battery":
        if re.search(r"\b(drain|draining|fast|quick|rapid)\b", text, re.IGNORECASE):
            return "fast_drain"
        if re.search(r"\b(not\s+charge|won.t\s+charge|charge\s+fail)\b", text, re.IGNORECASE):
            return "charge_fail"
        return "battery_issue"

    # Camera symptoms
    if category == "camera":
        if re.search(r"\b(blur|blurry|focus)\b", text, re.IGNORECASE):
            return "blur"
        if re.search(r"\b(black|blank|not\s+open|crash)\b", text, re.IGNORECASE):
            return "camera_fail"
        return "camera_issue"

    return None


def _intent(symptom: str | None, category: str | None) -> str | None:
    """Map (category, symptom) → user intent label."""
    # Display intents
    if symptom in {"black", "blank", "white"}:
        return "restore_display"
    if symptom in {"flicker", "flash"}:
        return "resolve_display_flicker"
    if symptom == "cracked":
        return "address_physical_screen_damage"
    if symptom in {"unresponsive", "delay", "laggy"}:
        return "restore_touch_responsiveness"

    # Connectivity intents
    if symptom == "disconnect":
        return "restore_connection"
    if symptom == "pair_fail":
        return "resolve_pairing_failure"
    if symptom == "not_detected":
        return "resolve_device_not_found"
    if category == "connectivity" and symptom == "no_connection":
        return "establish_connection"

    # App intents
    if symptom == "crash":
        return "resolve_app_crash"
    if symptom == "freeze":
        return "resolve_app_freeze"
    if category == "app" and symptom == "malfunction":
        return "resolve_app_malfunction"

    # Thermal
    if symptom == "overheating":
        return "resolve_overheating"

    # Storage
    if symptom in {"storage_full", "low_storage"}:
        return "free_storage"

    # Battery
    if symptom == "fast_drain":
        return "resolve_battery_drain"
    if symptom == "charge_fail":
        return "resolve_charging_failure"

    # Camera
    if symptom == "camera_fail":
        return "restore_camera"
    if symptom == "blur":
        return "resolve_camera_quality"

    # Audio
    if symptom == "no_sound":
        return "restore_audio"
    if symptom == "distorted":
        return "resolve_audio_quality"

    return None


def _troubleshooting_context(category: str | None, symptom: str | None) -> str | None:
    """Derive a fine-grained troubleshooting context from category + symptom."""
    if category == "display":
        if symptom in {"black", "blank", "white"}:
            return "display_not_visible"
        if symptom in {"flicker", "flash"}:
            return "display_intermittent"
        if symptom == "cracked":
            return "physical_display_damage"
        if symptom in {"unresponsive", "delay", "laggy"}:
            return "touch_input_problem"
        return "display"

    if category == "connectivity":
        if symptom == "disconnect":
            return "connectivity_unstable"
        if symptom == "pair_fail":
            return "bluetooth_pairing_failure"
        if symptom == "not_detected":
            return "device_not_detected"
        return "connectivity"

    if category == "app":
        if symptom == "crash":
            return "app_crash"
        if symptom == "freeze":
            return "app_freeze"
        return "app_malfunction"

    if category == "thermal":
        return "device_overheating"

    if category == "storage":
        return "storage_full" if symptom == "storage_full" else "low_storage"

    if category == "battery":
        if symptom == "fast_drain":
            return "battery_drain"
        if symptom == "charge_fail":
            return "charging_failure"
        return "battery"

    if category == "camera":
        if symptom == "camera_fail":
            return "camera_failure"
        if symptom == "blur":
            return "camera_quality"
        return "camera"

    if category == "audio":
        if symptom == "no_sound":
            return "audio_silent"
        return "audio"

    # For performance, email, and None — fall back to raw category
    return category


def fingerprint_query(query: str) -> QueryFingerprint:
    """Extract only high-confidence compatibility signals from a user query."""
    text = normalize_text(query)
    family = _device_family(text)

    category = _problem_category(text)
    symptom = _symptom(text, category)
    intent = _intent(symptom, category)
    context = _troubleshooting_context(category, symptom)

    if re.search(r"\b(ring|rings|powers on|turns on|still works)\b", text):
        power_state = "powered_on_display_failed"
    elif re.search(r"\b(will not turn on|won.t turn on|doesn.t turn on|not start)\b", text):
        power_state = "will_not_power_on"
    else:
        power_state = None

    if re.search(r"\b(crack|cracked|damage|damaged|liquid|water)\b", text):
        physical_damage = True
    elif re.search(r"\b(no physical damage|without physical damage)\b", text):
        physical_damage = False
    else:
        physical_damage = None

    return QueryFingerprint(
        device_type=_device_type(text, family),
        device_family=family,
        problem_category=category,
        symptom=symptom,
        intent=intent,
        troubleshooting_context=context,
        physical_damage=physical_damage,
        power_state=power_state,
    )


def normalized_cache_key(query: str) -> str:
    """Return the normalized lexical key used as the first cache lookup key."""
    return normalize_text(query)
