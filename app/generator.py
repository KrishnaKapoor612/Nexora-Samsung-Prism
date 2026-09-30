"""Structured troubleshooting-plan generators."""

from __future__ import annotations

import re
from typing import Protocol

from app.evidence import Evidence
from app.models import Action, ActionCategory, Goal, StepGroup, TroubleshootingResponse
from app.web_provider import _web_to_steps


STEP_HEADING = re.compile(r"^\s*(?:#+\s*)?step\s*\d+\s*:?\s*(.*)$", re.IGNORECASE)
IMPERATIVE = re.compile(
    r"\b(check|ensure|verify|navigate|open|tap|touch|select|press|hold|connect|"
    r"remove|restart|charge|turn|contact|try|inspect|swipe|clear|sign|"
    r"wait|power|plug|unplug|disconnect|toggle|slide|allow|enable|disable|go|"
    r"launch|scroll|adjust|switch|set|force|close|update|install|download|delete|"
    r"confirm|enter|look|place|pull|push|insert|reinsert|shine|run|perform|scan|"
    r"back|drag|drop|visit|replace|clean|wipe|let|access|long)\b",
    re.IGNORECASE,
)

_CRITICAL_TERMS = frozenset(
    {"reset", "factory", "update", "wipe", "service", "contact", "repair", "format"}
)
_AUTO_TERMS = frozenset(
    {"settings", "display", "navigation", "brightness", "sensitivity", "battery", 
     "wifi", "bluetooth", "connection", "sound", "notification", "gesture", "touch"}
)

_DESCRIPTION_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bphysical damage\b|\bliquid\b|\bldi\b", re.I), "It will verify device physical condition"),
    (re.compile(r"\bforce.*restart\b|\bhard.*reset\b|\breboot\b|\brestart\b", re.I), "It will clear temporary software faults"),
    (re.compile(r"\bcharg\b|\bbattery\b|\bpower\b", re.I), "It will help resolve power issues"),
    (re.compile(r"\bsafe.?mode\b", re.I), "It will isolate third party issues"),
    (re.compile(r"\bcache\b|\bapp.?data\b|\bclear.?data\b", re.I), "It will remove corrupted cached data"),
    (re.compile(r"\bwi.?fi\b|\binternet\b|\bnetwork\b|\bconnect\b", re.I), "It will verify network connectivity status"),
    (re.compile(r"\bemail\b|\baccount\b", re.I), "It will refresh account related settings"),
    (re.compile(r"\bsmart.?switch\b|\btransfer\b|\bbackup\b|\bback.?up\b", re.I), "It will help secure device data"),
    (re.compile(r"\bscreen\b|\bdisplay\b|\bbrightness\b", re.I), "It will help diagnose display issues"),
    (re.compile(r"\btouch\b|\bsensitiv\b|\bresponsive\b", re.I), "It will adjust touch input settings"),
    (re.compile(r"\bsoftware\b|\bupdate\b|\bfirmware\b", re.I), "It will apply latest software fixes"),
    (re.compile(r"\bfactory\b|\breset\b|\bwipe\b", re.I), "It will restore original device state"),
    (re.compile(r"\bservice\b|\brepair\b|\bcontact\b|\bsupport\b", re.I), "It will connect you with support"),
    (re.compile(r"\bsim\b|\bmicrosd\b|\btray\b", re.I), "It will check SIM tray condition"),
    (re.compile(r"\bapp\b|\bapplication\b|\binstall\b", re.I), "It will isolate the problematic app"),
    (re.compile(r"\bgesture\b|\bnavigation\b|\bswipe\b", re.I), "It will adjust gesture navigation settings"),
    (re.compile(r"\bmulti.?window\b|\bsplit\b|\bpop.?up\b|\bedge\b", re.I), "It will configure multi window features"),
    (re.compile(r"\bfloat\b|\bcircle\b|\bassistant\b", re.I), "It will manage floating menu options"),
]


def _action_description(action_name: str) -> str:
    """Return a human-readable description derived from the action name."""
    for pattern, hint in _DESCRIPTION_HINTS:
        if pattern.search(action_name):
            return hint
    return "It will help resolve this issue"


def _action_category(action_name: str) -> ActionCategory:
    """Classify an action as auto, manual, or critical based on its name."""
    lower = action_name.lower()
    if any(term in lower for term in _CRITICAL_TERMS):
        return ActionCategory.critical
    if any(term in lower for term in _AUTO_TERMS):
        return ActionCategory.auto
    return ActionCategory.manual


def _goal_title(evidence_title: str) -> str:
    noise_words = {"on", "using", "for", "with", "in", "a", "an", "the", "your", "samsung", "galaxy", "phone", "tablet", "device", "or", "and"}
    words = evidence_title.split()
    meaningful = [w for w in words if w.lower() not in noise_words]
    if not meaningful:
        return "Device Troubleshooting"
    result = " ".join(meaningful[:3])
    return result.capitalize()


def _goal_sentence(query: str, evidence_title: str) -> str:
    topic = _goal_title(evidence_title)
    is_config = any(word in query.lower() for word in ["config", "setup", "set up", "how to set"])
    if is_config:
        return f"Follow these steps to perform this {topic} Configuration"
    return f"Follow these steps to perform this {topic} Troubleshooting"


class StructuredPlanGenerator(Protocol):
    def generate(self, query: str, evidence: Evidence) -> TroubleshootingResponse:
        """Generate a plan grounded only in the supplied evidence."""


def _split_into_atomic_steps(text: str) -> list[str]:
    sentences = re.split(r'\.\s+(?=[A-Z])|\n', text)
    result = []
    
    for sentence in sentences:
        if re.match(r"^(Note|Important|There are|Here is|Please be aware|It's also|This is especially|You can also):?\b", sentence, re.IGNORECASE):
            continue
            
        parts = re.split(r',\s+(?:and\s+)?(?:then\s+)?(?=tap\b|select\b|navigate\b|open\b)', sentence, flags=re.IGNORECASE)
        
        for part in parts:
            part = part.strip()
            if IMPERATIVE.search(part):
                if len(part) > 250 and not re.search(r'\b(tap|select|open|navigate)\b', part, re.IGNORECASE):
                    continue
                    
                if part:
                    part = part[0].upper() + part[1:]
                    if not part.endswith('.'):
                        part += '.'
                    result.append(part)
                    
    return result


def parse_steps(content: str) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_steps: list[str] = []

    def flush() -> None:
        nonlocal current_name, current_steps
        if current_name and current_steps:
            final_steps = []
            for step in current_steps:
                if len(step) > 120:
                    split_steps = _split_into_atomic_steps(step)
                    if split_steps:
                        final_steps.extend(split_steps)
                    else:
                        final_steps.append(step)
                else:
                    final_steps.append(step)
            groups.append((current_name, final_steps))
        current_name = None
        current_steps = []

    for raw_line in content.splitlines():
        line = re.sub(r"^\s*[-*]\s*", "", raw_line).strip()
        if not line:
            continue
        heading = STEP_HEADING.match(line)
        if heading:
            flush()
            current_name = heading.group(1).strip() or "Troubleshooting step"
            continue
        if current_name is not None and IMPERATIVE.search(line):
            current_steps.append(line)
    flush()
    return groups


class DeterministicPlanGenerator:
    """Grounded baseline generator used offline and as a test oracle."""

    def generate(self, query: str, evidence: Evidence) -> TroubleshootingResponse:
        actions: list[Action] = []
        parsed = parse_steps(evidence.content)
        if not parsed:
            normalized = _web_to_steps(evidence.content)
            if normalized != evidence.content:
                parsed = parse_steps(normalized)
        for action_name, steps in parsed:
            titled = action_name.title()
            
            final_steps = []
            for step in steps:
                if len(step) > 120:
                    split_steps = _split_into_atomic_steps(step)
                    if split_steps:
                        final_steps.extend(split_steps)
                    else:
                        final_steps.append(step)
                else:
                    final_steps.append(step)
            
            actions.append(
                Action(
                    actionName=titled,
                    description=_action_description(action_name),
                    stepGroups=[StepGroup(steps=final_steps)],
                    category=_action_category(action_name),
                )
            )

        if not actions:
            return TroubleshootingResponse(contexts=[])

        # Guarantee safe ordering: auto < manual < critical
        category_rank = {
            ActionCategory.auto: 0,
            ActionCategory.manual: 1,
            ActionCategory.critical: 2,
        }
        actions.sort(key=lambda a: category_rank[a.category])

        return TroubleshootingResponse(
            contexts=[
                Goal(
                    goal=_goal_sentence(query, evidence.title),
                    title=_goal_title(evidence.title),
                    actions=actions,
                    score=round(min(evidence.score, 1.0), 2),
                )
            ]
        )
