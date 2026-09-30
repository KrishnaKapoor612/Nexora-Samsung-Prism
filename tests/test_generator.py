import pytest

from app.config import load_assets
from app.evidence import Evidence, StaticEvidenceProvider
from app.generator import DeterministicPlanGenerator, parse_steps
from app.models import TroubleshootingResponse, TroubleshootRequest
from app.nova_generator import NovaGenerationError, NovaLitePlanGenerator
from app.pipeline import TroubleshootingPipeline


def test_parse_steps_extracts_grounded_instructions() -> None:
    groups = parse_steps(
        "## Step 1: Check the device\n"
        "Inspect the charger.\n"
        "This is explanatory text.\n"
        "## Step 2: Restart\n"
        "Press and hold the Power button."
    )
    assert groups == [
        ("Check the device", ["Inspect the charger."]),
        ("Restart", ["Press and hold the Power button."]),
    ]


def test_deterministic_generator_does_not_create_deeplinks() -> None:
    response = DeterministicPlanGenerator().generate(
        "screen issue",
        Evidence(
            source="test",
            title="Display troubleshooting",
            content="## Step 1: Check display\nOpen Settings and tap Display.",
            score=0.8,
        ),
    )
    group = response.contexts[0].actions[0].stepGroups[0]
    assert group.actionableDeeplink is None
    assert group.validationDeeplink is None


class StubGenerator:
    called = False

    def generate(self, query: str, evidence: Evidence) -> TroubleshootingResponse:
        self.called = True
        assert query == "screen issue"
        assert evidence.source == "test"
        return TroubleshootingResponse(contexts=[])


def test_pipeline_accepts_injected_generator() -> None:
    generator = StubGenerator()
    pipeline = TroubleshootingPipeline(
        load_assets(),
        evidence_provider=StaticEvidenceProvider(
            Evidence(
                source="test",
                title="Display",
                content="Check the display.",
                score=1.0,
            )
        ),
        generator=generator,
    )
    result = pipeline.run(TroubleshootRequest(query="screen issue"))
    assert generator.called is True
    assert result.meta.fallback == "no_match"


class FailingNovaGenerator:
    def generate(
        self, _query: str, _evidence: Evidence
    ) -> TroubleshootingResponse:
        del _query, _evidence
        raise NovaGenerationError("simulated Bedrock failure")


def test_pipeline_returns_no_match_when_nova_fails() -> None:
    pipeline = TroubleshootingPipeline(
        load_assets(),
        evidence_provider=StaticEvidenceProvider(
            Evidence(
                source="test",
                title="Display",
                content="Check the display.",
                score=1.0,
            )
        ),
        generator=FailingNovaGenerator(),
    )
    result = pipeline.run(TroubleshootRequest(query="screen issue"))
    assert result.response.contexts == []
    assert result.meta.fallback == "no_match"


class FakeBedrockClient:
    def converse(self, **kwargs):
        assert kwargs["modelId"] == "test-model"
        assert kwargs["inferenceConfig"]["temperature"] == 0.0
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": (
                                '{"contexts":[{"goal":"Follow these steps to perform '
                                'this Display Troubleshooting","title":"Display issue",'
                                '"score":0.9,"actions":[{"actionName":"Check Display",'
                                '"description":"It will check the display",'
                                '"category":"manual","stepGroups":[{"steps":'
                                '["Check the display."],"actionableDeeplink":null,'
                                '"validationDeeplink":null}]}]}]}'
                            )
                        }
                    ]
                }
            }
        }


class InvalidBedrockClient:
    def converse(self, **kwargs):
        assert kwargs["modelId"]
        return {
            "output": {
                "message": {
                    "content": [{"text": "not json"}],
                }
            }
        }


def test_nova_generator_parses_fake_bedrock_response() -> None:
    generator = NovaLitePlanGenerator(
        client=FakeBedrockClient(),
        model_id="test-model",
    )
    result = generator.generate(
        "My display is not working",
        Evidence(
            source="test",
            title="Display troubleshooting",
            content="Check the display.",
            score=0.9,
        ),
    )
    assert result.contexts[0].actions[0].stepGroups[0].steps == [
        "Check the display."
    ]


def test_nova_generator_rejects_invalid_json() -> None:
    generator = NovaLitePlanGenerator(client=InvalidBedrockClient())
    with pytest.raises(NovaGenerationError):
        generator.generate(
            "Display issue",
            Evidence(
                source="test",
                title="Display",
                content="Check the display.",
                score=0.9,
            ),
        )
