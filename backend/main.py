"""
FastAPI Backend Application for Bengaluru Road Traffic Congestion Prediction.
Provides REST endpoints:
- GET  /health: Service status, model type, roads & areas monitored
- GET  /options: Dynamic metadata for dropdowns, roads grouped by area, metrics
- GET  /scenarios: Real-world Bengaluru commute presets
- POST /predict: Single-corridor prediction using calibrated Logistic Regression
- POST /chat: Server-Sent Events (SSE) streaming AI Operations Assistant
"""

from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.chat import ChatRequest, generate_chat_stream
from backend.predictor import (
    AVAILABLE_AREAS,
    AVAILABLE_ROADS,
    get_options,
    predict_single,
    rank_all_roads
)
from backend.schemas import HealthResponse, OptionsResponse, PredictRequest, PredictResponse

app = FastAPI(
    title="Bengaluru Road Traffic Congestion Prediction API",
    description="Operational machine learning service for predicting surface road traffic congestion across Bengaluru's arterial corridors.",
    version="4.0.0"
)

# CORS configuration strictly allowing the frontend application
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

# Real-world Bengaluru scenario presets
PRESET_SCENARIOS = [
    {
        "id": "sony-world-rush",
        "title": "Koramangala Sony World Peak",
        "description": "High-density tech corridor transit through Koramangala.",
        "road_name": "Sony World Junction",
        "area_name": "Koramangala",
        "day_of_week": 0,
        "month": 10,
        "weather_condition": "Clear",
        "roadwork": False
    },
    {
        "id": "sarjapur-roadwork",
        "title": "Sarjapur Road Active Construction",
        "description": "Carriageway constriction due to civic digging & lane closure.",
        "road_name": "Sarjapur Road",
        "area_name": "Koramangala",
        "day_of_week": 1,
        "month": 11,
        "weather_condition": "Overcast",
        "roadwork": True
    },
    {
        "id": "indiranagar-weekend",
        "title": "Indiranagar 100 Feet Rd Saturday Evening",
        "description": "Weekend shopping and dining leisure traffic.",
        "road_name": "100 Feet Road",
        "area_name": "Indiranagar",
        "day_of_week": 5,
        "month": 12,
        "weather_condition": "Clear",
        "roadwork": False
    },
    {
        "id": "marathahalli-monsoon",
        "title": "Marathahalli Bridge Monsoon Rain",
        "description": "Rain-induced surface waterlogging and heavy braking.",
        "road_name": "Marathahalli Bridge",
        "area_name": "Whitefield",
        "day_of_week": 2,
        "month": 7,
        "weather_condition": "Rain",
        "roadwork": False
    },
    {
        "id": "hebbal-morning",
        "title": "Hebbal Flyover Airport Inflow",
        "description": "Morning inbound highway corridor merging with airport traffic.",
        "road_name": "Hebbal Flyover",
        "area_name": "Hebbal",
        "day_of_week": 0,
        "month": 10,
        "weather_condition": "Fog",
        "roadwork": False
    },
    {
        "id": "tumkur-offpeak",
        "title": "Tumkur Road Smooth Sunday Cruise",
        "description": "Optimal low-risk weekend travel along northern arterial corridor.",
        "road_name": "Tumkur Road",
        "area_name": "Yeshwanthpur",
        "day_of_week": 6,
        "month": 1,
        "weather_condition": "Clear",
        "roadwork": False
    }
]


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint polled periodically by the dashboard."""
    return {
        "status": "healthy",
        "service": "Bengaluru Road Traffic Congestion Risk API",
        "city": "Bengaluru",
        "model": "Calibrated Logistic Regression (L2 Regularized)",
        "roads_monitored": len(AVAILABLE_ROADS),
        "areas_monitored": len(AVAILABLE_AREAS)
    }


@app.get("/options", response_model=OptionsResponse)
def options_endpoint():
    """Return all dropdown and filter options dynamically derived from the model bundle."""
    return get_options()


@app.get("/scenarios")
def scenarios_endpoint():
    """Return realistic Bengaluru preset travel scenarios for 1-click testing."""
    return PRESET_SCENARIOS


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    """Predict road traffic congestion risk for a specific Bengaluru corridor."""
    try:
        result = predict_single(
            road_name=req.road_name,
            weather_condition=req.weather_condition,
            roadwork=req.roadwork,
            day_of_week=req.day_of_week,
            month=req.month,
            area_name=req.area_name
        )
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(exc)}"
        )


@app.get("/roads/rank")
def rank_roads_endpoint(
    day_of_week: int = 0,
    month: int = 10,
    weather_condition: str = "Clear",
    roadwork: bool = False
):
    """Rank all 16 Bengaluru corridors from highest to lowest congestion risk."""
    return rank_all_roads(
        weather_condition=weather_condition,
        roadwork=roadwork,
        day_of_week=day_of_week,
        month=month
    )


@app.post("/chat")
async def chat_endpoint(chat_req: ChatRequest, request: Request):
    """
    Stream AI responses with Server-Sent Events (SSE).
    Uses official OpenAI client pointed at Groq with manual tool calling loop.
    """
    client_ip = request.client.host if request.client else "127.0.0.1"

    return StreamingResponse(
        generate_chat_stream(chat_req, client_ip=client_ip),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )