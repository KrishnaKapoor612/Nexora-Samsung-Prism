from app.config import load_assets
from app.config import AppAssets
from app.constants import DUMMY_POSITIVE_DEEPLINK
from app.deeplink_mapper import DeeplinkMapper
from app.embeddings import CatalogEmbeddingIndex


def test_every_match_deeplink_is_verbatim_from_catalog() -> None:
    """Non-negotiable: no deeplink is ever invented, only ranked."""
    assets = load_assets()
    mapper = DeeplinkMapper(assets)
    samples = [
        ("Back Up Phone Data", "It will back up your data", [
            "Navigate to and open Settings.",
            "Tap Accounts and backup.",
            "Select Back up data to secure your personal files.",
        ]),
        ("Configure Navigation Bar Settings", "It will let you choose navigation type", [
            "Navigate to and open Settings.",
            "Tap Display, and then tap Navigation bar.",
            "Select your preferred navigation type between Buttons and Swipe gestures.",
        ]),
        ("Contact Samsung Support", "It will help resolve this issue", [
            "Contact Samsung Support or visit an authorized Samsung Service Center.",
        ]),
    ]
    for action_name, description, steps in samples:
        match = mapper.match(action_name, description, steps)
        if match is not None:
            assert match.entry["deeplink"] in assets.deeplink_uris


def test_dummy_positive_used_only_for_unindexed_settings_screen() -> None:
    assets = load_assets()
    mapper = DeeplinkMapper(assets)
    match = mapper.match(
        "Configure Made Up Feature Toggle",
        "It will do something new",
        [
            "Navigate to and open Settings.",
            "Tap Display, and then tap Totally Fictional Toggle.",
        ],
    )
    assert match is not None
    assert match.is_dummy is True
    assert match.entry["deeplink"] == DUMMY_POSITIVE_DEEPLINK
    word_count = len(match.entry["description"].split())
    assert 5 <= word_count <= 7


def test_dummy_positive_not_used_for_non_settings_manual_action() -> None:
    assets = load_assets()
    mapper = DeeplinkMapper(assets)
    match = mapper.match(
        "Schedule Screen Repair Service",
        "It will help you locate a service center",
        ["Contact Samsung Support or visit an authorized Samsung Service Center."],
    )
    assert match is None


def test_embedding_index_ranks_shared_rare_terms_over_shared_common_terms() -> None:
    entries = [
        {"id": "A", "deeplink": "bixby://a", "description": "Enable warranty status lookup"},
        {"id": "B", "deeplink": "bixby://b", "description": "Open general device settings screen"},
    ]
    index = CatalogEmbeddingIndex(entries)
    scores = index.similarity_for_all("check warranty status for this device")
    assert scores[0] > scores[1]


def test_toggle_synonym_matches_catalog_entry_and_returns_only_catalog_uri() -> None:
    assets = AppAssets(siis_rows=[{}], deeplinks=[
        {"id": "touch", "deeplink": "bixby://catalog/touch",
         "message": "Enable Touch sensitivity",
         "qna_description": "Turn on touch sensitivity"},
        {"id": "wifi", "deeplink": "bixby://catalog/wifi",
         "message": "Connect to Wi-Fi", "qna_description": "Join wireless network"},
    ])
    match = DeeplinkMapper(assets).match(
        "Turn on touch sensitivity", "", ["Enable touch sensitivity"]
    )
    assert match is not None
    assert match.entry["deeplink"] == "bixby://catalog/touch"
    assert match.entry["deeplink"] in assets.deeplink_uris


def test_ambiguous_and_low_confidence_deeplink_queries_return_none() -> None:
    assets = AppAssets(siis_rows=[{}], deeplinks=[
        {"id": "a", "deeplink": "bixby://catalog/a", "message": "Open display settings"},
        {"id": "b", "deeplink": "bixby://catalog/b", "message": "Open display settings"},
    ])
    mapper = DeeplinkMapper(assets)
    assert mapper.match("display settings", "", []) is None
    assert mapper.match("unrelated unknown task", "", []) is None
