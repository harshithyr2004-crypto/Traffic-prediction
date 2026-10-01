"""
Python Execution Tools & OpenAI Function Schemas for Bengaluru Traffic AI Chatbot.
Provides 5 mathematical calculation tools:
1. predict_congestion
2. compare_scenarios
3. rank_roads_by_risk
4. get_live_conditions (Open-Meteo for Bengaluru: 12.9716 N, 77.5946 E)
5. explain_model_factors (Reads model_facts.json)
Zero hallucinated calculations. Zero fake hour-of-day features.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import zoneinfo

import httpx

from backend.predictor import (
    AVAILABLE_AREAS,
    AVAILABLE_ROADS,
    DEFAULT_THRESHOLD,
    ROAD_TO_AREA,
    WEATHER_CONDITIONS,
    get_model_facts,
    predict_single,
    rank_all_roads
)

BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946
BENGALURU_TZ = "Asia/Kolkata"


def map_wmo_code_to_weather(wmo_code: int, wind_speed_kmh: float = 0.0) -> str:
    """Map standard WMO weather codes to the Bengaluru dataset's 5 weather categories."""
    if wind_speed_kmh >= 30.0:
        return "Windy"
    if wmo_code == 0:
        return "Clear"
    elif wmo_code in [1, 2, 3]:
        return "Overcast"
    elif wmo_code in [45, 48]:
        return "Fog"
    elif wmo_code in [51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99]:
        return "Rain"
    return "Clear"


def tool_predict_congestion(
    road_name: str,
    weather_condition: str = "Clear",
    roadwork: bool = False,
    day_of_week: int = 0,
    month: int = 10
) -> Dict[str, Any]:
    """Calculate congestion risk probability for a specific Bengaluru road."""
    try:
        return predict_single(
            road_name=road_name,
            weather_condition=weather_condition,
            roadwork=roadwork,
            day_of_week=day_of_week,
            month=month
        )
    except Exception as e:
        return {"status": "error", "message": f"Prediction failed: {str(e)}"}


def tool_compare_scenarios(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare multiple travel scenarios across different Bengaluru roads, days, or conditions."""
    if not isinstance(scenarios, list) or len(scenarios) == 0:
        return {"status": "error", "message": "Expected a non-empty list of scenario dictionaries."}

    results = []
    for sc in scenarios[:5]:  # limit to 5 scenarios
        rname = sc.get("road_name", "Sony World Junction")
        wcond = sc.get("weather_condition", "Clear")
        rwork = bool(sc.get("roadwork", False))
        dow = int(sc.get("day_of_week", 0))
        m = int(sc.get("month", 10))

        pred = predict_single(
            road_name=rname,
            weather_condition=wcond,
            roadwork=rwork,
            day_of_week=dow,
            month=m
        )
        results.append({
            "road_name": pred["road_name"],
            "area_name": pred["area_name"],
            "day_name": pred["day_name"],
            "weather": wcond,
            "roadwork": rwork,
            "congestion_probability": pred["congestion_probability"],
            "risk_level": pred["risk_level"],
            "estimated_speed_kmh": pred["estimated_speed_kmh"],
            "estimated_delay_min": pred["estimated_delay_min"]
        })

    # Find scenario with lowest risk
    best_scenario = min(results, key=lambda x: x["congestion_probability"])

    return {
        "scenarios_compared": len(results),
        "results": results,
        "recommendation": f"Lowest risk scenario is {best_scenario['road_name']} on {best_scenario['day_name']} with {best_scenario['congestion_probability']*100:.1f}% congestion risk."
    }


def tool_rank_roads_by_risk(
    day_of_week: int = 0,
    month: int = 10,
    weather_condition: str = "Clear",
    roadwork: bool = False
) -> Dict[str, Any]:
    """Rank all 16 monitored Bengaluru corridors from highest to lowest congestion risk."""
    try:
        ranked = rank_all_roads(
            weather_condition=weather_condition,
            roadwork=roadwork,
            day_of_week=day_of_week,
            month=month
        )
        return {
            "day_of_week": day_of_week,
            "month": month,
            "weather_condition": weather_condition,
            "roadwork": roadwork,
            "total_roads": len(ranked),
            "ranked_roads": ranked,
            "highest_risk_road": ranked[0]["road_name"],
            "lowest_risk_road": ranked[-1]["road_name"]
        }
    except Exception as e:
        return {"status": "error", "message": f"Ranking failed: {str(e)}"}


def tool_get_live_conditions() -> Dict[str, Any]:
    """Fetch live real-time ambient weather and current date in Bengaluru via Open-Meteo."""
    try:
        kolkata_tz = zoneinfo.ZoneInfo(BENGALURU_TZ)
        now = datetime.now(kolkata_tz)
    except Exception:
        now = datetime.now()

    day_of_week = now.weekday()  # 0=Monday, 6=Sunday
    month = now.month

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={BENGALURU_LAT}&longitude={BENGALURU_LON}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,cloud_cover,wind_speed_10m"
        f"&timezone=Asia%2FKolkata"
    )

    try:
        with httpx.Client(timeout=6.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        return {
            "status": "error",
            "message": "Open-Meteo live weather service timed out for Bengaluru. Please supply conditions manually.",
            "today_day_of_week": day_of_week,
            "today_month": month
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Unable to reach live weather service for Bengaluru: {str(exc)}. Please specify conditions manually.",
            "today_day_of_week": day_of_week,
            "today_month": month
        }

    current = data.get("current", {})
    temp_c = current.get("temperature_2m", 25.0)
    wmo_code = int(current.get("weather_code", 0))
    wind_kmh = float(current.get("wind_speed_10m", 0.0))
    rain_mm = float(current.get("rain", 0.0))

    mapped_weather = map_wmo_code_to_weather(wmo_code, wind_kmh)

    return {
        "status": "success",
        "city": "Bengaluru",
        "coordinates": {"lat": BENGALURU_LAT, "lon": BENGALURU_LON},
        "temperature_c": temp_c,
        "rain_mm": rain_mm,
        "wind_speed_kmh": wind_kmh,
        "weather_condition": mapped_weather,
        "day_of_week": day_of_week,
        "day_name": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_of_week],
        "month": month,
        "formatted_time": now.strftime("%Y-%m-%d %I:%M %p IST")
    }


def tool_explain_model_factors() -> Dict[str, Any]:
    """Retrieve top positive and protective road congestion odds ratios from model_facts.json."""
    facts = get_model_facts()
    if not facts:
        return {"status": "error", "message": "Model facts artifact unavailable."}

    return {
        "city": facts.get("city", "Bengaluru"),
        "algorithm": facts.get("algorithm"),
        "threshold": facts.get("threshold"),
        "metrics": facts.get("metrics"),
        "top_congested_factors": facts.get("top_congested_factors", []),
        "top_protective_factors": facts.get("top_protective_factors", [])
    }


# Map tool name strings to executable Python functions
TOOL_REGISTRY = {
    "predict_congestion": tool_predict_congestion,
    "compare_scenarios": tool_compare_scenarios,
    "rank_roads_by_risk": tool_rank_roads_by_risk,
    "get_live_conditions": tool_get_live_conditions,
    "explain_model_factors": tool_explain_model_factors
}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Safely invoke a tool function by name with parsed arguments."""
    func = TOOL_REGISTRY.get(tool_name)
    if not func:
        return {
            "status": "error",
            "message": f"Unknown tool '{tool_name}'. Available tools: {list(TOOL_REGISTRY.keys())}"
        }
    try:
        return func(**arguments)
    except TypeError as te:
        return {"status": "error", "message": f"Invalid arguments for {tool_name}: {str(te)}"}
    except Exception as exc:
        return {"status": "error", "message": f"Execution error in {tool_name}: {str(exc)}"}


# OpenAI Function Calling JSON Schemas for Groq LLM
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_congestion",
            "description": "Calculate road traffic congestion probability for a specific Bengaluru road/corridor under given weather, roadwork, and calendar conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "road_name": {
                        "type": "string",
                        "description": "Name of the Bengaluru road or junction (e.g., 'Sony World Junction', '100 Feet Road', 'Hebbal Flyover', 'Silk Board Junction', 'Marathahalli Bridge').",
                        "enum": AVAILABLE_ROADS
                    },
                    "weather_condition": {
                        "type": "string",
                        "description": "Meteorological condition: 'Clear', 'Overcast', 'Fog', 'Rain', or 'Windy'.",
                        "enum": WEATHER_CONDITIONS,
                        "default": "Clear"
                    },
                    "roadwork": {
                        "type": "boolean",
                        "description": "True if active carriageway construction, roadwork, or lane constriction is present.",
                        "default": False
                    },
                    "day_of_week": {
                        "type": "integer",
                        "description": "Day of the week: 0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday, 4 = Friday, 5 = Saturday, 6 = Sunday.",
                        "minimum": 0,
                        "maximum": 6,
                        "default": 0
                    },
                    "month": {
                        "type": "integer",
                        "description": "Month of the year (1 = January to 12 = December).",
                        "minimum": 1,
                        "maximum": 12,
                        "default": 10
                    }
                },
                "required": ["road_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_scenarios",
            "description": "Compare multiple travel options across different Bengaluru roads, days, or weather conditions side-by-side.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scenarios": {
                        "type": "array",
                        "description": "List of scenario objects to compare.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "road_name": {"type": "string", "enum": AVAILABLE_ROADS},
                                "weather_condition": {"type": "string", "enum": WEATHER_CONDITIONS},
                                "roadwork": {"type": "boolean"},
                                "day_of_week": {"type": "integer", "minimum": 0, "maximum": 6},
                                "month": {"type": "integer", "minimum": 1, "maximum": 12}
                            },
                            "required": ["road_name"]
                        }
                    }
                },
                "required": ["scenarios"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rank_roads_by_risk",
            "description": "Rank all 16 monitored Bengaluru arterial roads and junctions from highest to lowest congestion risk under specified conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day_of_week": {
                        "type": "integer",
                        "description": "Day of week: 0 = Monday ... 6 = Sunday",
                        "minimum": 0,
                        "maximum": 6,
                        "default": 0
                    },
                    "month": {
                        "type": "integer",
                        "description": "Month: 1-12",
                        "minimum": 1,
                        "maximum": 12,
                        "default": 10
                    },
                    "weather_condition": {
                        "type": "string",
                        "enum": WEATHER_CONDITIONS,
                        "default": "Clear"
                    },
                    "roadwork": {
                        "type": "boolean",
                        "default": False
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_conditions",
            "description": "Fetch real-time ambient weather (temperature, rain, cloud cover, wind) and current day/month for Bengaluru, India.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_model_factors",
            "description": "Retrieve the exact learned Logistic Regression coefficients, Odds Ratios (exp(coef)), and top influential factors from model_facts.json.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]
