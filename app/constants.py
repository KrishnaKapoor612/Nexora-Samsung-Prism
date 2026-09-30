"""Shared constants used across the pipeline.

Kept in one place so app/deeplink_mapper.py and app/validators.py never
drift out of sync on the literal URI string.
"""

from __future__ import annotations

# Reserved, catalog-provided placeholder (see student_kit/deeplinks.json,
# entry id "DL-DUMMY"). It is the ONLY deeplink URI ever used when a step
# opens a real Settings screen that has no dedicated entry in the catalog.
# The URI string itself always comes verbatim from deeplinks.json -- never
# from an LLM and never invented at runtime. Only the accompanying
# description/message text is authored programmatically, exactly as the
# catalog's own qna_description instructs.
DUMMY_POSITIVE_DEEPLINK = "bixby://dummy_positive"
