from app.config import load_assets
from app.evidence import Evidence, StaticEvidenceProvider, select_evidence
from app.models import TroubleshootRequest
from app.pipeline import TroubleshootingPipeline
from app.query_understanding import fingerprint_query


def test_known_query_returns_grounded_context() -> None:
    pipeline = TroubleshootingPipeline(load_assets())
    result = pipeline.run(
        TroubleshootRequest(
            query="My Galaxy S24 Ultra screen is completely black and will not turn on"
        )
    )
    assert result.response.contexts
    assert result.meta.fallback is None
    assert result.response.contexts[0].actions


def test_unrelated_query_returns_controlled_no_match() -> None:
    pipeline = TroubleshootingPipeline(load_assets())
    result = pipeline.run(TroubleshootRequest(query="How do I cook rice?"))
    assert result.response.contexts == []
    assert result.meta.fallback == "no_match"


def test_pipeline_uses_retrieval_without_request_siis_response() -> None:
    provider = StaticEvidenceProvider(
        Evidence(
            source="test",
            title="Display troubleshooting",
            content="## Step 1: Check display\nOpen Settings and tap Display.",
            score=0.95,
        )
    )
    pipeline = TroubleshootingPipeline(
        load_assets(),
        evidence_provider=provider,
    )
    result = pipeline.run(TroubleshootRequest(query="screen issue"))
    assert result.response.contexts


def test_repeated_normalized_query_uses_cache() -> None:
    pipeline = TroubleshootingPipeline(load_assets())
    query = "My Galaxy S24 Ultra screen is completely black and will not turn on"
    pipeline.run(TroubleshootRequest(query=query))
    result = pipeline.run(TroubleshootRequest(query=query.upper()))
    assert result.meta.cache_hit is True


def test_query_fingerprint_separates_black_screen_and_flickering() -> None:
    black = fingerprint_query("My Galaxy S24 screen is black but the phone still rings")
    flickering = fingerprint_query("My Galaxy S24 screen is flickering whenever I open it")

    assert black.problem_category == flickering.problem_category == "display"
    assert black.symptom != flickering.symptom
    assert black.intent != flickering.intent
    assert black.troubleshooting_context != flickering.troubleshooting_context


def test_semantically_similar_but_incompatible_queries_are_cache_misses() -> None:
    pipeline = TroubleshootingPipeline(load_assets())
    black_query = "My Galaxy S24 screen is black but the phone still rings"
    flicker_query = "My Galaxy S24 screen is flickering whenever I open it"

    first = pipeline.run(TroubleshootRequest(query=black_query))
    second = pipeline.run(TroubleshootRequest(query=flicker_query))

    assert first.response.contexts
    assert second.meta.cache_hit is False
    assert second.meta.reason in {
        "evidence_not_found_or_incompatible",
        "generator_returned_no_steps",
        "generated_response_failed_validation",
    }


def test_similar_compatible_query_can_reuse_cache() -> None:
    pipeline = TroubleshootingPipeline(load_assets())
    pipeline.run(
        TroubleshootRequest(
            query="My Galaxy S24 screen is black while the phone still rings"
        )
    )
    result = pipeline.run(
        TroubleshootRequest(
            query="Galaxy S24 display is black, device still rings"
        )
    )
    assert result.meta.cache_hit is True
    assert result.meta.reason.startswith("compatible_cache:")


def test_fallback_provider_supplies_grounded_evidence() -> None:
    fallback = StaticEvidenceProvider(
        Evidence(
            source="test_fallback",
            title="Blank display fallback",
            content=(
                "## Step 1: Force a restart\n"
                "Press and hold the Power button and Volume down button."
            ),
            score=0.9,
            reliability_score=0.9,
        )
    )
    pipeline = TroubleshootingPipeline(
        load_assets(),
        fallback_provider=fallback,
    )
    result = pipeline.run(
        TroubleshootRequest(query="A completely unfamiliar display problem")
    )
    assert result.response.contexts
    assert result.meta.fallback is None


def test_evidence_selection_rejects_wrong_display_symptom() -> None:
    candidates = [
        Evidence(
            source="wrong",
            title="Blank or black display",
            content="## Step 1: Restart\nPress the power button.",
            score=0.95,
            reliability_score=0.9,
        ),
        Evidence(
            source="right",
            title="Screen flickers",
            content="## Step 1: Check display\nCheck the display when it flickers.",
            score=0.75,
            reliability_score=0.9,
        ),
    ]
    selected = select_evidence("My phone screen flickers", candidates)
    assert selected is not None
    assert selected.source == "right"


def test_siis_miss_attempts_web_fallback_before_no_match() -> None:
    class EmptyProvider:
        def __init__(self) -> None:
            self.called = False

        def search(self, query: str, limit: int = 5) -> list[Evidence]:
            del query, limit
            self.called = True
            return []

    class WebHitProvider:
        def __init__(self) -> None:
            self.called = False

        def search(self, query: str, limit: int = 5) -> list[Evidence]:
            del query, limit
            self.called = True
            return [Evidence(
                source="web", source_type="web", title="Bluetooth pairing support",
                content=("## Step 1: Pair Bluetooth\nOpen Settings and pair the Bluetooth device."),
                score=1.0, url="https://www.samsung.com/support/bluetooth",
                reliability_score=1.0,
            )]

    siis, web = EmptyProvider(), WebHitProvider()
    pipeline = TroubleshootingPipeline(
        load_assets(), evidence_provider=siis, fallback_provider=web
    )
    result = pipeline.run(TroubleshootRequest(query="Bluetooth pairing problem"))
    assert siis.called and web.called
    assert result.response.contexts


def test_web_disabled_miss_returns_no_match_without_static_answer() -> None:
    empty = StaticEvidenceProvider(None)
    pipeline = TroubleshootingPipeline(load_assets(), evidence_provider=empty)
    result = pipeline.run(TroubleshootRequest(query="Unlisted obscure phone issue"))
    assert result.response.contexts == []
    assert result.meta.fallback == "no_match"
