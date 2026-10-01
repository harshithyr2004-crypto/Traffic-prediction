"""
Centralized ML Prediction & Feature Engineering Module for Bengaluru Road Traffic Congestion.
Loads the calibrated Logistic Regression model from model/bengaluru_traffic_model.joblib.
Zero data leakage, fully reproducible, single source of truth for both REST API and AI Chatbot tools.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Paths to model artifacts
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "bengaluru_traffic_model.joblib"
FACTS_PATH = BASE_DIR / "model" / "model_facts.json"

# In-memory cached artifacts
MODEL_BUNDLE: Dict[str, Any] = {}
MODEL_FACTS: Dict[str, Any] = {}

if MODEL_PATH.exists():
    MODEL_BUNDLE = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Required Bengaluru model bundle not found at {MODEL_PATH}")

if FACTS_PATH.exists():
    with open(FACTS_PATH, "r", encoding="utf-8") as f:
        MODEL_FACTS = json.load(f)

# Extract core metadata from the bundle
FEATURE_SPEC: Dict[str, Any] = MODEL_BUNDLE.get("feature_spec", {})
AVAILABLE_ROADS: List[str] = FEATURE_SPEC.get("roads", [
    "100 Feet Road", "Anil Kumble Circle", "Ballari Road", "CMH Road",
    "Hebbal Flyover", "Hosur Road", "ITPL Main Road", "Jayanagar 4th Block",
    "Marathahalli Bridge", "Sarjapur Road", "Silk Board Junction",
    "Sony World Junction", "South End Circle", "Trinity Circle",
    "Tumkur Road", "Yeshwanthpur Circle"
])
AVAILABLE_AREAS: List[str] = FEATURE_SPEC.get("areas", [
    "Electronic City", "Hebbal", "Indiranagar", "Jayanagar",
    "Koramangala", "M.G. Road", "Whitefield", "Yeshwanthpur"
])
ROAD_TO_AREA: Dict[str, str] = FEATURE_SPEC.get("road_to_area", {
    "100 Feet Road": "Indiranagar",
    "CMH Road": "Indiranagar",
    "Marathahalli Bridge": "Whitefield",
    "ITPL Main Road": "Whitefield",
    "Sony World Junction": "Koramangala",
    "Sarjapur Road": "Koramangala",
    "Trinity Circle": "M.G. Road",
    "Anil Kumble Circle": "M.G. Road",
    "Jayanagar 4th Block": "Jayanagar",
    "South End Circle": "Jayanagar",
    "Hebbal Flyover": "Hebbal",
    "Ballari Road": "Hebbal",
    "Yeshwanthpur Circle": "Yeshwanthpur",
    "Tumkur Road": "Yeshwanthpur",
    "Silk Board Junction": "Electronic City",
    "Hosur Road": "Electronic City"
})
AREA_TO_ROADS: Dict[str, List[str]] = FEATURE_SPEC.get("area_to_roads", {
    area: [r for r, a in ROAD_TO_AREA.items() if a == area]
    for area in AVAILABLE_AREAS
})
WEATHER_CONDITIONS: List[str] = FEATURE_SPEC.get("weather_conditions", ["Clear", "Overcast", "Fog", "Rain", "Windy"])
TRAINING_COLUMNS: List[str] = MODEL_BUNDLE.get("columns", [])
NUM_COLS: List[str] = MODEL_BUNDLE.get("num_cols", ["day_of_week", "month"])
DEFAULT_THRESHOLD: float = MODEL_BUNDLE.get("threshold", 0.45)
MODEL_METRICS: Dict[str, Any] = MODEL_BUNDLE.get("metrics", {})

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def get_model_bundle() -> Dict[str, Any]:
    """Return the loaded Bengaluru model bundle."""
    return MODEL_BUNDLE


def get_model_facts() -> Dict[str, Any]:
    """Return model facts and odds ratios from model_facts.json."""
    return MODEL_FACTS


def get_options() -> Dict[str, Any]:
    """Return all dropdown and filter options for frontend and tools."""
    return {
        "city": "Bengaluru",
        "areas": AVAILABLE_AREAS,
        "roads": AVAILABLE_ROADS,
        "area_to_roads": AREA_TO_ROADS,
        "road_to_area": ROAD_TO_AREA,
        "weather_conditions": WEATHER_CONDITIONS,
        "roadwork_options": [False, True],
        "day_of_week_options": [
            {"value": i, "label": name, "is_weekend": i in [5, 6]}
            for i, name in enumerate(DAY_NAMES)
        ],
        "months": [
            {"value": m, "label": MONTH_NAMES[m]}
            for m in range(1, 13)
        ],
        "threshold": DEFAULT_THRESHOLD,
        "metrics": MODEL_METRICS
    }


def build_input_row(
    road_name: str,
    weather_condition: str = "Clear",
    roadwork: bool = False,
    day_of_week: int = 0,
    month: int = 10
) -> pd.DataFrame:
    """
    Construct a single-row feature vector conforming exactly to the 23 columns
    used during Logistic Regression training.
    """
    row = pd.DataFrame(0, index=[0], columns=TRAINING_COLUMNS)

    # Continuous and binary numeric values
    row["roadwork"] = 1 if roadwork else 0
    row["is_weekend"] = 1 if day_of_week in [5, 6] else 0
    row["day_of_week"] = int(day_of_week)
    row["month"] = int(month)

    # One-hot encoded road
    road_col = f"road_{road_name}"
    if road_col in row.columns:
        row[road_col] = 1

    # One-hot encoded weather
    weather_col = f"weather_{weather_condition}"
    if weather_col in row.columns:
        row[weather_col] = 1

    # Scale the continuous columns using the training scaler
    row_scaled = row.copy()
    scaler = MODEL_BUNDLE["scaler"]
    row_scaled[NUM_COLS] = scaler.transform(row[NUM_COLS])

    return row_scaled


def predict_single(
    road_name: str,
    weather_condition: str = "Clear",
    roadwork: bool = False,
    day_of_week: int = 0,
    month: int = 10,
    area_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute single-corridor prediction using calibrated Logistic Regression.
    """
    # Sanitize inputs
    matched_road = None
    for r in AVAILABLE_ROADS:
        if r.lower() == road_name.strip().lower():
            matched_road = r
            break
    if not matched_road:
        matched_road = AVAILABLE_ROADS[0]

    matched_weather = None
    for w in WEATHER_CONDITIONS:
        if w.lower() == weather_condition.strip().lower():
            matched_weather = w
            break
    if not matched_weather:
        matched_weather = "Clear"

    day_of_week = max(0, min(6, int(day_of_week)))
    month = max(1, min(12, int(month)))

    # Inferred area
    inferred_area = ROAD_TO_AREA.get(matched_road, area_name or "Bengaluru Urban")

    # Build row and run model
    X_row = build_input_row(
        road_name=matched_road,
        weather_condition=matched_weather,
        roadwork=roadwork,
        day_of_week=day_of_week,
        month=month
    )

    clf = MODEL_BUNDLE["model"]
    prob = float(clf.predict_proba(X_row)[0, 1])
    thresh = MODEL_BUNDLE.get("threshold", DEFAULT_THRESHOLD)
    prediction = 1 if prob >= thresh else 0
    label = "High Congestion" if prediction == 1 else "Normal Flow"

    # Risk tier classification
    if prob < 0.40:
        risk_level = "Low"
    elif prob < 0.70:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Realistic civil transportation speed and delay modeling
    # Free-flow speed on arterial roads ~45 km/h
    speed_drop_prob = prob * 22.0  # Up to 22 km/h drop at severe saturation
    speed_drop_work = 5.0 if roadwork else 0.0
    speed_drop_rain = 4.0 if matched_weather == "Rain" else (2.0 if matched_weather in ["Overcast", "Windy"] else 0.0)
    est_speed = max(12.0, round(45.0 - speed_drop_prob - speed_drop_work - speed_drop_rain, 1))

    # Queue delay estimation in minutes
    base_delay = 5.0
    cong_delay = 30.0 * (prob ** 1.3)
    work_delay = 8.0 if roadwork else 0.0
    rain_delay = 6.0 if matched_weather == "Rain" else 0.0
    est_delay = int(round(base_delay + cong_delay + work_delay + rain_delay))

    # Practical commute recommendation
    if risk_level == "High":
        if roadwork:
            rec = f"SEVERE BOTTLENECK ALERT on {matched_road} ({inferred_area}). Roadwork has constricted effective carriageway width. Seek an alternate parallel arterial."
        elif matched_weather == "Rain":
            rec = f"HIGH CONGESTION ADVISORY on {matched_road}. Monsoon rain and surface waterlogging are causing heavy braking and slow speeds ({est_speed} km/h). Expect ~{est_delay} min delay."
        else:
            rec = f"ELEVATED GRIDLOCK RISK on {matched_road} ({inferred_area}). Corridor is operating near saturated capacity. Consider departing earlier or using an alternate route."
    elif risk_level == "Medium":
        rec = f"MODERATE CONGESTION on {matched_road}. Traffic is moving steadily ({est_speed} km/h) but corridor density is elevated. Allow an extra {est_delay} minutes buffer."
    else:
        rec = f"FREE FLOW CONDITIONS on {matched_road}. Expected speed is optimal ({est_speed} km/h) with minimal corridor delay (~{est_delay} min)."

    return {
        "city": "Bengaluru",
        "prediction": prediction,
        "label": label,
        "congestion_probability": round(prob, 4),
        "risk_level": risk_level,
        "threshold": thresh,
        "road_name": matched_road,
        "area_name": inferred_area,
        "day_name": DAY_NAMES[day_of_week],
        "month_name": MONTH_NAMES[month],
        "estimated_speed_kmh": est_speed,
        "estimated_delay_min": est_delay,
        "recommendation": rec,
        "inputs_used": {
            "road_name": matched_road,
            "area_name": inferred_area,
            "weather_condition": matched_weather,
            "roadwork": roadwork,
            "day_of_week": day_of_week,
            "day_name": DAY_NAMES[day_of_week],
            "month": month,
            "month_name": MONTH_NAMES[month],
            "is_weekend": day_of_week in [5, 6]
        }
    }


def rank_all_roads(
    weather_condition: str = "Clear",
    roadwork: bool = False,
    day_of_week: int = 0,
    month: int = 10
) -> List[Dict[str, Any]]:
    """
    Score and rank all 16 Bengaluru corridors from highest to lowest risk under given conditions.
    """
    ranked = []
    for road in AVAILABLE_ROADS:
        res = predict_single(
            road_name=road,
            weather_condition=weather_condition,
            roadwork=roadwork,
            day_of_week=day_of_week,
            month=month
        )
        ranked.append({
            "road_name": road,
            "area_name": res["area_name"],
            "congestion_probability": res["congestion_probability"],
            "risk_level": res["risk_level"],
            "estimated_speed_kmh": res["estimated_speed_kmh"],
            "estimated_delay_min": res["estimated_delay_min"]
        })

    # Sort descending by probability
    ranked.sort(key=lambda x: x["congestion_probability"], reverse=True)
    return ranked
