from urllib.parse import quote

from app.web_provider import (
    SearchResult,
    WebEvidenceProvider,
    _decode_search_url,
    is_reliable_source,
)


class FakeSearchClient:
    def search(self, query: str, limit: int) -> list[SearchResult]:
        assert "battery drains" in query
        assert limit >= 1
        return [
            SearchResult(
                title="Samsung battery troubleshooting",
                url="https://www.samsung.com/us/support/troubleshooting/battery",
            ),
            SearchResult(
                title="Untrusted result",
                url="https://example.com/battery",
            ),
        ]


class FakePageFetcher:
    def fetch(self, url: str) -> str:
        assert url.startswith("https://www.samsung.com/")
        return (
            "Step 1: Open Settings. Step 2: Tap Battery and device care. "
            "Check battery usage and install available software updates. "
            "Contact Samsung Support if the issue continues."
        )


def test_source_allowlist_rejects_untrusted_domains() -> None:
    assert is_reliable_source("https://www.samsung.com/support") is True
    assert is_reliable_source("https://example.com/support") is False


def test_web_provider_filters_sources_and_returns_evidence() -> None:
    provider = WebEvidenceProvider(FakeSearchClient(), FakePageFetcher())
    results = provider.search("my battery drains quickly", limit=2)
    assert len(results) == 1
    assert results[0].source_type == "web"
    assert results[0].url == "https://www.samsung.com/us/support/troubleshooting/battery"
    assert results[0].reliability_score == 1.0


def test_web_provider_retries_query_variants_after_empty_search() -> None:
    class VariantSearchClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, limit: int) -> list[SearchResult]:
            del limit
            self.queries.append(query)
            if len(self.queries) == 1:
                return []
            return [
                SearchResult(
                    title="Samsung battery troubleshooting",
                    url="https://www.samsung.com/support/battery",
                )
            ]

    client = VariantSearchClient()
    results = WebEvidenceProvider(client, FakePageFetcher()).search(
        "my battery drains quickly", limit=1
    )
    assert results
    assert len(client.queries) >= 2


def test_decode_search_redirect_url() -> None:
    destination = "https://www.samsung.com/us/support/troubleshooting/battery"
    redirect = "https://duckduckgo.com/l/?uddg=" + quote(destination)
    assert _decode_search_url(redirect) == destination
