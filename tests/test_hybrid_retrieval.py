from app.bm25_retriever import BM25SIISRetriever
from app.config import AppAssets, load_assets
from app.hybrid_retriever import HybridSIISRetriever, normalize_scores
from app.semantic_retriever import LocalSIISSemanticIndex


def _row(row_id: str, title: str, content: str, query: str = "") -> dict:
    return {
        "id": row_id,
        "original_query": query,
        "siis_response": {"title": title, "content": content},
    }


def test_bm25_preserves_exact_model_and_technical_terms() -> None:
    rows = [
        _row("s22", "Galaxy S22 Safe Mode", "Connect to Wi-Fi and test Bluetooth."),
        _row("s23", "Galaxy S23 Ultra", "Open Settings and review the display."),
    ]
    results = BM25SIISRetriever(rows).search_bm25("Galaxy S22 Safe Mode Wi-Fi", top_k=1)
    assert results[0].siis_id == "s22"
    assert "Wi-Fi" in results[0].searchable_text


def test_bm25_indexes_title_and_complete_body_and_honors_top_k() -> None:
    rows = [
        _row("title", "Galaxy screen support", "A generic display article."),
        _row("body", "Display support", "Use Safe Mode to diagnose a screen flicker."),
    ]
    retriever = BM25SIISRetriever(rows)
    assert retriever.search_bm25("Safe Mode", top_k=1)[0].siis_id == "body"
    assert len(retriever.search_bm25("screen", top_k=1)) == 1
    assert BM25SIISRetriever([]).search_bm25("screen") == []


def test_semantic_index_retrieves_paraphrased_power_issue() -> None:
    rows = [_row("power", "Galaxy phone will not power on", "The device will not start."),
            _row("wifi", "Galaxy Wi-Fi disconnects", "Reconnect the wireless network.")]
    results = LocalSIISSemanticIndex(rows).search_semantic("My phone won't start", top_k=1)
    assert results[0].siis_id == "power"
    assert results[0].score > 0


def test_hybrid_fuses_bm25_and_semantic_candidates() -> None:
    rows = [
        _row("exact", "Galaxy S22 Safe Mode", "Start the phone in Safe Mode."),
        _row("paraphrase", "Phone will not power on", "The handset will not start."),
        _row("other", "Bluetooth pairing", "Pair a Bluetooth accessory."),
    ]
    retriever = HybridSIISRetriever(AppAssets(siis_rows=rows, deeplinks=[]))
    query = "Galaxy S22 Safe Mode phone won't start"
    candidates = retriever.search(query)
    ids = {candidate.candidate.siis_id for candidate in candidates}
    assert "exact" in ids
    assert "paraphrase" in ids


def test_hybrid_returns_empty_when_evidence_is_unrelated() -> None:
    assets = load_assets()
    retriever = HybridSIISRetriever(assets)
    assert retriever.search("How do I cook rice?") == []


def test_score_normalization_handles_empty_equal_zero_and_nonfinite_values() -> None:
    assert normalize_scores([]) == []
    assert normalize_scores([0]) == [0.0]
    assert normalize_scores([3, 3]) == [1.0, 1.0]
    assert normalize_scores([0, 0]) == [0.0, 0.0]
    values = normalize_scores([2.0, float("nan"), float("inf")])
    assert values == [1.0, 0.0, 0.0]

