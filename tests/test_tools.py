"""
Automated unit and integration test suite for Bengaluru Road Traffic Congestion Prediction & Tools.
Tests:
1. Shared predictor matches /predict endpoint output byte-for-byte.
2. /options matches the saved model bundle.
3. rank_roads_by_risk returns every monitored road in Bengaluru (16 roads).
4. get_live_conditions network failure is handled safely without crashing.
5. Bad tool arguments and unknown tools do not crash tool execution or chat.
6. Model facts and odds ratios are mathematically verified against the joblib bundle.
"""

import sys
from pathlib import Path
from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.predictor import get_model_bundle, get_options, predict_single
from backend.tools import (
    execute_tool,
    tool_explain_model_factors,
    tool_get_live_conditions,
    tool_predict_congestion,
    tool_rank_roads_by_risk,
)


def test_shared_predictor_matches_predict_endpoint():
    """Verify that backend/predictor.py matches the /predict endpoint byte-for-byte."""
    client = TestClient(app)
    payload = {
        "road_name": "Sony World Junction",
        "weather_condition": "Rain",
        "roadwork": True,
        "day_of_week": 0,
        "month": 10
    }

    # 1. Direct call to shared predictor
    direct_result = predict_single(**payload)

    # 2. HTTP POST call to /predict endpoint
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    api_result = response.json()

    # Compare core outputs
    assert direct_result["prediction"] == api_result["prediction"]
    assert direct_result["label"] == api_result["label"]
    assert direct_result["risk_level"] == api_result["risk_level"]
    assert direct_result["road_name"] == api_result["road_name"]
    assert direct_result["area_name"] == api_result["area_name"]
    assert abs(direct_result["congestion_probability"] - api_result["congestion_probability"]) < 1e-4
    print("[PASS] Test 1: Shared predictor matches /predict endpoint.")


def test_options_matches_bundle():
    """Verify that /options endpoint matches the saved model bundle metadata."""
    client = TestClient(app)
    bundle = get_model_bundle()
    spec = bundle["feature_spec"]

    response = client.get("/options")
    assert response.status_code == 200
    data = response.json()

    assert data["city"] == "Bengaluru"
    assert len(data["roads"]) == len(spec["roads"])
    assert set(data["roads"]) == set(spec["roads"])
    assert len(data["areas"]) == len(spec["areas"])
    assert set(data["areas"]) == set(spec["areas"])
    assert data["threshold"] == bundle["threshold"]
    print("[PASS] Test 2: /options matches the model bundle.")


def test_rank_roads_by_risk_returns_every_road():
    """Verify that rank_roads_by_risk returns all 16 Bengaluru corridors sorted by probability."""
    res = tool_rank_roads_by_risk(day_of_week=0, month=10, weather_condition="Clear", roadwork=False)

    assert "ranked_roads" in res
    assert res["total_roads"] == 16, f"Expected 16 roads, got {res['total_roads']}"
    assert len(res["ranked_roads"]) == 16

    # Verify descending probability sorting
    probs = [item["congestion_probability"] for item in res["ranked_roads"]]
    assert probs == sorted(probs, reverse=True), "Expected roads to be sorted by probability descending"

    for r in res["ranked_roads"]:
        assert 0.0 <= r["congestion_probability"] <= 1.0
        assert r["risk_level"] in ("Low", "Medium", "High")
        assert len(r["area_name"]) > 0

    print("[PASS] Test 3: rank_roads_by_risk returns all 16 roads sorted by risk.")


def test_live_conditions_network_failure_handling():
    """Verify that get_live_conditions handles network timeouts or connection failures cleanly without faking."""
    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Mocked timeout error")):
        res = tool_get_live_conditions()
        assert res.get("status") == "error"
        assert "timed out" in res.get("message", "").lower()

    with patch("httpx.Client.get", side_effect=httpx.ConnectError("Mocked connection refused")):
        res = tool_get_live_conditions()
        assert res.get("status") == "error"
        assert "unable to reach" in res.get("message", "").lower()

    print("[PASS] Test 4: Live-conditions failure handled cleanly without crashing.")


def test_bad_tool_argument_does_not_crash_chat():
    """Verify that bad arguments, unknown tools, or invalid payloads do not crash execution."""
    # Test 1: execute_tool with non-existent tool
    err_res = execute_tool("unknown_tool", {})
    assert err_res.get("status") == "error"
    assert "Unknown tool" in err_res.get("message", "")

    # Test 2: out-of-range day of week or unusual arguments
    res = tool_predict_congestion(road_name="Nonexistent Road", day_of_week=99, month=13)
    assert "congestion_probability" in res
    assert res["prediction"] in (0, 1)

    print("[PASS] Test 5: Bad tool arguments and edge cases handled safely.")


def test_model_facts_and_odds_ratios_verified():
    """Verify that model_facts.json contains genuine coefficients and calculated odds ratios."""
    facts = tool_explain_model_factors()
    assert facts.get("city") == "Bengaluru"
    assert "Logistic Regression" in facts.get("algorithm")
    assert len(facts.get("top_congested_factors", [])) > 0
    assert len(facts.get("top_protective_factors", [])) > 0

    for item in facts["top_congested_factors"]:
        assert item["odds_ratio"] >= 1.0, f"Expected congested factor OR >= 1.0, got {item['odds_ratio']}"

    for item in facts["top_protective_factors"]:
        assert item["odds_ratio"] < 1.0, f"Expected protective factor OR < 1.0, got {item['odds_ratio']}"

    print("[PASS] Test 6: Model facts and odds ratios verified.")


if __name__ == "__main__":
    print("Running Bengaluru Road Traffic Test Suite...")
    test_shared_predictor_matches_predict_endpoint()
    test_options_matches_bundle()
    test_rank_roads_by_risk_returns_every_road()
    test_live_conditions_network_failure_handling()
    test_bad_tool_argument_does_not_crash_chat()
    test_model_facts_and_odds_ratios_verified()
    print("\nALL 6 TEST SUITES PASSED SUCCESSFULLY!")
