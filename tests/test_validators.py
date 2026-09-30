import pytest

from app.config import load_assets
from app.models import (
    Action,
    ActionCategory,
    ActionableDeeplink,
    Goal,
    StepGroup,
    TroubleshootingResponse,
    ValidationDeeplink,
)
from app.query_understanding import fingerprint_query
from app.validators import (
    ResponseValidationError,
    validate_no_urls,
    validate_response,
)


def test_catalog_and_schema_validation_accepts_empty_response() -> None:
    assets = load_assets()
    response = validate_response(
        TroubleshootingResponse(contexts=[]),
        assets.deeplink_uris,
    )
    assert response.contexts == []


def test_url_leakage_is_rejected() -> None:
    with pytest.raises(ResponseValidationError):
        validate_no_urls({"steps": ["Open https://example.com"]})


def test_unsafe_action_order_is_rejected() -> None:
    assets = load_assets()
    response = TroubleshootingResponse(
        contexts=[
            Goal(
                goal="Follow these steps to perform this Test Troubleshooting",
                title="Test issue",
                score=0.8,
                actions=[
                    Action(
                        actionName="Reset",
                        description="It will reset the device",
                        category=ActionCategory.critical,
                        stepGroups=[StepGroup(steps=["Reset the device."])],
                    ),
                    Action(
                        actionName="Check",
                        description="It will check the device",
                        category=ActionCategory.manual,
                        stepGroups=[StepGroup(steps=["Check the device."])],
                    ),
                ],
            )
        ]
    )
    with pytest.raises(ResponseValidationError):
        validate_response(response, assets.deeplink_uris, "Reset the device. Check the device.")


def test_duplicate_actions_are_rejected() -> None:
    assets = load_assets()
    action = Action(
        actionName="Check display",
        description="It will check the display",
        stepGroups=[StepGroup(steps=["Check the display."])],
    )
    response = TroubleshootingResponse(
        contexts=[
            Goal(
                goal="Follow these steps to perform this Display Troubleshooting",
                title="Display issue",
                score=0.8,
                actions=[action, action.model_copy(deep=True)],
            )
        ]
    )
    with pytest.raises(ResponseValidationError, match="Duplicate action"):
        validate_response(response, assets.deeplink_uris)


def test_mismatched_action_and_validation_deeplinks_are_rejected() -> None:
    assets = load_assets()
    first = assets.deeplinks[0]
    second = next(
        entry
        for entry in assets.deeplinks
        if isinstance(entry.get("validation"), dict)
        and entry["validation"].get("deeplink")
        and entry["deeplink"] != first["deeplink"]
    )
    action = Action(
        actionName="Switch Time Format",
        description="It will switch the time format",
        stepGroups=[
            StepGroup(
                steps=["Open time format settings."],
                actionableDeeplink=ActionableDeeplink(
                    deeplink=first["deeplink"],
                    description=first["description"],
                    message=first.get("message", ""),
                ),
                validationDeeplink=ValidationDeeplink(
                    deeplink=second["validation"]["deeplink"],
                    key=second["validation"].get("key", ""),
                ),
            )
        ],
    )
    response = TroubleshootingResponse(
        contexts=[
            Goal(
                goal="Follow these steps to perform this Display Troubleshooting",
                title="Display issue",
                score=0.8,
                actions=[action],
            )
        ]
    )
    with pytest.raises(ResponseValidationError, match="do not belong"):
        validate_response(response, assets.deeplink_uris, catalog_entries=assets.deeplinks)


def test_catalog_valid_but_wrong_action_deeplink_is_rejected() -> None:
    assets = load_assets()
    entry = assets.deeplinks[0]
    response = TroubleshootingResponse(
        contexts=[
            Goal(
                goal="Follow these steps to perform this Battery Troubleshooting",
                title="Battery issue",
                score=0.8,
                actions=[
                    Action(
                        actionName="Battery settings",
                        description="It will manage battery settings",
                        stepGroups=[
                            StepGroup(
                                steps=["Open battery settings."],
                                actionableDeeplink=ActionableDeeplink(
                                    deeplink=entry["deeplink"],
                                    description=entry["description"],
                                    message=entry.get("message", ""),
                                ),
                            )
                        ],
                    )
                ],
            )
        ]
    )
    with pytest.raises(ResponseValidationError, match="does not match action"):
        validate_response(
            response,
            assets.deeplink_uris,
            catalog_entries=assets.deeplinks,
        )


def test_context_mismatch_is_rejected() -> None:
    assets = load_assets()
    response = TroubleshootingResponse(
        contexts=[
            Goal(
                goal="Follow these steps to perform this Battery Troubleshooting",
                title="Battery drain",
                score=0.8,
                actions=[
                    Action(
                        actionName="Battery settings",
                        description="It will improve battery life",
                        stepGroups=[StepGroup(steps=["Open battery settings."])],
                    )
                ],
            )
        ]
    )
    with pytest.raises(ResponseValidationError, match="problem_category"):
        validate_response(
            response,
            assets.deeplink_uris,
            query_fingerprint=fingerprint_query("My Galaxy S24 screen is black"),
        )