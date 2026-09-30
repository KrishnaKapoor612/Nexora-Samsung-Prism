"""Application configuration and startup asset loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
STUDENT_KIT_DIR = ROOT_DIR / "student_kit"


@dataclass(frozen=True)
class AppAssets:
    siis_rows: list[dict[str, Any]]
    deeplinks: list[dict[str, Any]]

    @property
    def deeplink_uris(self) -> frozenset[str]:
        uris: set[str] = set()
        for entry in self.deeplinks:
            if isinstance(entry.get("deeplink"), str):
                uris.add(entry["deeplink"])
            validation = entry.get("validation")
            if isinstance(validation, dict) and isinstance(
                validation.get("deeplink"), str
            ):
                uris.add(validation["deeplink"])
        return frozenset(uris)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_assets(student_kit_dir: Path = STUDENT_KIT_DIR) -> AppAssets:
    """Load and validate the static SIIS and deeplink assets."""
    siis_payload = _load_json(student_kit_dir / "siis_responses.json")
    deeplink_payload = _load_json(student_kit_dir / "deeplinks.json")

    siis_rows = siis_payload.get("responses")
    deeplinks = deeplink_payload.get("deeplinks")
    if not isinstance(siis_rows, list) or not isinstance(deeplinks, list):
        raise ValueError("Student-kit assets have invalid list fields")
    if len(siis_rows) == 0:
        raise ValueError("SIIS asset is empty")
    if len(deeplinks) == 0:
        raise ValueError("Deeplink catalog is empty")

    return AppAssets(siis_rows=siis_rows, deeplinks=deeplinks)