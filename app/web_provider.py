"""Credential-free web evidence provider with source allowlisting.

The provider is intentionally injectable: tests can supply deterministic search
and fetch clients, while deployments can replace the HTTP clients with an API
backed implementation without changing the evidence contract.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen

from app.evidence import Evidence
from app.retrieval import content_tokens


ALLOWED_HOST_SUFFIXES = (
    "samsung.com",
    "samsungmobilepress.com",
)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""


class SearchClient(Protocol):
    def search(self, query: str, limit: int) -> list[SearchResult]:
        ...


class PageFetcher(Protocol):
    def fetch(self, url: str) -> str:
        ...


def is_reliable_source(url: str) -> bool:
    host = urlparse(url).hostname
    if not host:
        return False
    host = host.lower().removeprefix("www.")
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in ALLOWED_HOST_SUFFIXES)


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0
        self._ignored_tags = {"script", "style", "nav", "footer", "header", "aside"}

    def handle_starttag(
        self, tag: str, _attrs: list[tuple[str, str | None]]
    ) -> None:
        _ = _attrs
        if tag.lower() in self._ignored_tags:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            cleaned = _clean_text(data)
            if cleaned:
                self.parts.append(cleaned)


class HttpSearchClient:
    """Small DuckDuckGo HTML adapter used when no search SDK is configured."""

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int) -> list[SearchResult]:
        headers = {"User-Agent": "Mozilla/5.0 SamsungTroubleshooting/1.0"}
        engines = (
            ("https://html.duckduckgo.com/html/?q=", _SearchResultParser),
            ("https://www.bing.com/search?q=", _BingResultParser),
        )
        for endpoint, parser_type in engines:
            try:
                request = Request(
                    f"{endpoint}{quote(query)}",
                    headers=headers,
                )
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    html = response.read().decode("utf-8", errors="replace")
                parser = parser_type()
                parser.feed(html)
                if parser.results:
                    return parser.results[:limit]
            except (OSError, ValueError):
                continue
        return []


class _SearchResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._current_url: str | None = None
        self._current_title: list[str] = []
        self._in_result = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = attributes.get("class", "") or ""
        if tag == "a" and "result__a" in classes:
            self._current_url = _decode_search_url(attributes.get("href"))
            self._current_title = []
            self._in_result = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_result and self._current_url:
            title = _clean_text("".join(self._current_title))
            if title:
                self.results.append(SearchResult(title=title, url=self._current_url))
            self._current_url = None
            self._current_title = []
            self._in_result = False

    def handle_data(self, data: str) -> None:
        if self._in_result:
            self._current_title.append(data)


class _BingResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._in_result = False
        self._current_url: str | None = None
        self._current_title: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "li" and "b_algo" in classes:
            self._in_result = True
        elif self._in_result and tag == "a" and self._current_url is None:
            self._current_url = _decode_search_url(attributes.get("href"))

    def handle_endtag(self, tag: str) -> None:
        if tag == "li" and self._in_result:
            title = _clean_text("".join(self._current_title))
            if title and self._current_url:
                self.results.append(
                    SearchResult(title=title, url=self._current_url)
                )
            self._in_result = False
            self._current_url = None
            self._current_title = []

    def handle_data(self, data: str) -> None:
        if self._in_result and self._current_url is not None:
            self._current_title.append(data)


def _decode_search_url(url: str | None) -> str | None:
    """Extract the destination URL from a search-engine redirect."""
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.path == "/l/":
        destination = parse_qs(parsed.query).get("uddg", [])
        if destination:
            return unquote(destination[0])
    if parsed.hostname and parsed.hostname.endswith("bing.com"):
        encoded = parse_qs(parsed.query).get("u", [])
        if encoded and encoded[0].startswith("a1"):
            try:
                padded = encoded[0][2:] + "==="
                return base64.urlsafe_b64decode(padded).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                pass
    return url


# Per-category search terms appended to Samsung support queries so the
# search engine is steered toward the right article family.
_CATEGORY_TERMS: dict[str, str] = {
    "display":      "screen display blank black flicker",
    "battery":      "battery drain charging power",
    "thermal":      "overheating hot temperature",
    "performance":  "slow lag performance speed",
    "connectivity": "wifi bluetooth network connect",
    "app":          "app crash freeze force close",
    "storage":      "storage memory space full",
    "camera":       "camera photo video",
    "audio":        "sound speaker audio volume",
    "email":        "email account sync",
}

# Imperative verbs that signal an actionable troubleshooting step.
_WEB_IMPERATIVE = re.compile(
    r"\b(navigate|tap|press|hold|check|select|turn|open|go|enable|disable|"
    r"clear|remove|connect|restart|charge|verify|ensure|inspect|sign|swipe)\b",
    re.IGNORECASE,
)

# Numbered list: "1. Do this" or "1) Do this"
_NUMBERED_STEP = re.compile(r"(?m)^[ \t]*(\d+)[.)][ \t]+(.+)$")
# Explicit "Step N:" or "Step N -" headings
_EXPLICIT_STEP = re.compile(r"(?mi)^[ \t]*step[ \t]*(\d+)[ \t]*[:\-\u2013]?[ \t]*(.*)$")
# Markdown numbered sub-headings: "### 1. Check for Physical Damage" -- distinct
# from _NUMBERED_STEP because the digit follows one or more '#' marks rather
# than starting the line, and from _EXPLICIT_STEP because it has no "Step"
# word at all.
_MARKDOWN_NUMBERED_HEADING = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]*(\d+)[.)][ \t]*(.+)$")


def _web_to_steps(content: str) -> str:
    """Convert raw web-page text into the ## Step N: format parse_steps expects.

    Tries these strategies in order:
    1. Numbered list items ("1. Navigate to Settings")
    2. Explicit "Step N:" / "Step N -" headings already present in the text
    2.5. Markdown numbered sub-headings ("### 1. Check for Physical Damage")
    3. Imperative-verb sentences used as synthetic step bodies
    Returns the original content unchanged if fewer than 3 steps are found.
    """
    # ── Strategy 1: numbered list items ────────────────────────────────────
    numbered = _NUMBERED_STEP.findall(content)
    if len(numbered) >= 3:
        parts = []
        for step_num, (_, text) in enumerate(numbered[:10], 1):
            text = text.strip()
            heading = " ".join(text.split()[:8]).rstrip(".,;")
            parts.append(f"## Step {step_num}: {heading}\n{text}")
        return "\n".join(parts)

    # ── Strategy 2: explicit Step N headings ───────────────────────────────
    positions = list(_EXPLICIT_STEP.finditer(content))
    if len(positions) >= 3:
        parts = []
        for i, match in enumerate(positions[:10]):
            heading_text = match.group(2).strip() or f"Step {match.group(1)}"
            start = match.end()
            end = positions[i + 1].start() if i + 1 < len(positions) else len(content)
            body = re.sub(r"\s+", " ", content[start:end].strip())
            parts.append(f"## Step {i + 1}: {heading_text}\n{body}")
        return "\n".join(parts)

    # ── Strategy 2.5: markdown numbered sub-headings ("### 1. Title") ──────
    # Handles SIIS/support content that groups steps under numbered H2/H3
    # headings instead of literal "Step N" text. Extracting by heading keeps
    # each group's real body together, instead of falling through to
    # Strategy 3 and flattening the whole article into individual sentences
    # (which can duplicate a sentence that legitimately appears twice under
    # two different headings, e.g. device-specific restart instructions).
    markdown_positions = list(_MARKDOWN_NUMBERED_HEADING.finditer(content))
    if len(markdown_positions) >= 3:
        parts = []
        for i, match in enumerate(markdown_positions[:10]):
            heading_text = match.group(2).strip() or f"Step {match.group(1)}"
            start = match.end()
            end = (
                markdown_positions[i + 1].start()
                if i + 1 < len(markdown_positions)
                else len(content)
            )
            body = re.sub(r"\s+", " ", content[start:end].strip())
            parts.append(f"## Step {i + 1}: {heading_text}\n{body}")
        return "\n".join(parts)

    # ── Strategy 3: imperative sentences as synthetic steps ────────────────
    step_sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", content)
        if len(s.strip()) > 20 and _WEB_IMPERATIVE.search(s)
    ]
    if len(step_sentences) >= 3:
        parts = []
        for step_num, text in enumerate(step_sentences[:10], 1):
            heading = " ".join(text.split()[:8]).rstrip(".,;")
            parts.append(f"## Step {step_num}: {heading}\n{text}")
        return "\n".join(parts)

    return content  # No pattern found — return as-is


def _search_queries(query: str, category: str | None = None) -> list[str]:
    """Generate category-aware search variants for Samsung support search."""
    extra = _CATEGORY_TERMS.get(category or "", "troubleshooting fix")
    return list(
        dict.fromkeys(
            [
                f"site:samsung.com/support Samsung Galaxy {query}",
                f"site:samsung.com/support {query} {extra}",
                f"site:samsung.com/support {query} troubleshooting",
                f"site:samsung.com {query} Samsung support fix",
            ]
        )
    )


class HttpPageFetcher:
    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch(self, url: str) -> str:
        request = Request(
            url,
            headers={"User-Agent": "SamsungTroubleshooting/1.0"},
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            html = response.read().decode("utf-8", errors="replace")
        parser = _VisibleTextParser()
        parser.feed(html)
        return _clean_text(" ".join(parser.parts))


class WebEvidenceProvider:
    """Retrieve and filter grounded evidence from allowlisted web sources."""

    def __init__(
        self,
        search_client: SearchClient | None = None,
        page_fetcher: PageFetcher | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.search_client = search_client or HttpSearchClient(timeout_seconds)
        self.page_fetcher = page_fetcher or HttpPageFetcher(timeout_seconds)

    def search(self, query: str, limit: int = 5) -> list[Evidence]:
        # Detect category once so all search variants are steered correctly.
        from app.query_understanding import fingerprint_query
        category = fingerprint_query(query).problem_category

        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for search_query in _search_queries(query, category=category):
            try:
                batch = self.search_client.search(
                    search_query, limit=max(limit * 2, limit)
                )
            except (OSError, ValueError, TimeoutError):
                continue
            for result in batch:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    results.append(result)
            if len(results) >= limit * 2:
                break

        evidence: list[Evidence] = []
        query_tokens = content_tokens(query)
        for rank, result in enumerate(results):
            if not is_reliable_source(result.url):
                continue
            try:
                raw_content = self.page_fetcher.fetch(result.url)
            except (OSError, ValueError):
                continue
            if len(raw_content) < 80:
                continue

            # Normalize web prose into ## Step N: format required by parse_steps.
            content = _web_to_steps(raw_content)

            # Skip pages where normalization couldn't extract any steps —
            # the generator would reject them with generator_returned_no_steps.
            if "## Step" not in content:
                continue

            overlap = len(query_tokens & content_tokens(f"{result.title} {content}"))
            # Score is overlap-driven with a modest floor; avoids giving
            # every samsung.com page a free 0.65 regardless of relevance.
            score = min(1.0, 0.30 + (0.10 * overlap) + (0.05 / (rank + 1)))
            evidence.append(
                Evidence(
                    source="web",
                    title=result.title,
                    content=content,
                    score=score,
                    evidence_id=result.url,
                    url=result.url,
                    source_type="web",
                    reliability_score=1.0,
                )
            )
            if len(evidence) >= limit:
                break
        return evidence

