"""
Pydantic Request and Response Schemas for Bengaluru Road Traffic Prediction API.
Strict validation, clear type annotations, and descriptive documentation for OpenAPI docs.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

WeatherType = Literal["Clear", "Overcast", "Fog", "Rain", "Windy"]


class PredictRequest(BaseModel):
    """Input parameters for predicting traffic congestion on a Bengaluru road."""
    road_name: str = Field(
        "Sony World Junction",
        description="Name of the Bengaluru corridor or major intersection"
    )
    area_name: Optional[str] = Field(
        None,
        description="Optional urban area/locality (e.g. Koramangala, Indiranagar)"
    )
    weather_condition: str = Field(
        "Clear",
        description="Meteorological condition: Clear, Overcast, Fog, Rain, Windy"
    )
    roadwork: bool = Field(
        False,
        description="Active carriageway roadwork, civic digging, or lane constriction"
    )
    day_of_week: int = Field(
        0,
        ge=0,
        le=6,
        description="Day of week: 0 = Monday ... 6 = Sunday"
    )
    month: int = Field(
        10,
        ge=1,
        le=12,
        description="Month of year: 1 = January ... 12 = December"
    )


class PredictResponse(BaseModel):
    """Comprehensive output payload returned by the Bengaluru road prediction engine."""
    city: str = Field("Bengaluru", description="Monitored metropolis")
    prediction: int = Field(..., description="Binary classification (1 = Congested, 0 = Normal Flow)")
    label: str = Field(..., description="High Congestion or Normal Flow")
    congestion_probability: float = Field(..., description="Calibrated posterior probability (0.0 to 1.0)")
    risk_level: str = Field(..., description="Low (<0.40), Medium (0.40-0.69), or High (>=0.70)")
    threshold: float = Field(..., description="Operational decision cutoff (0.45)")
    road_name: str = Field(..., description="Monitored road/intersection")
    area_name: str = Field(..., description="Urban area/zone")
    day_name: str = Field(..., description="Full day name (Monday - Sunday)")
    month_name: str = Field(..., description="Full month name (January - December)")
    estimated_speed_kmh: float = Field(..., description="Estimated average corridor speed in km/h")
    estimated_delay_min: int = Field(..., description="Estimated queue delay in minutes")
    recommendation: str = Field(..., description="Practical commute action guidance")
    inputs_used: Dict[str, Any] = Field(..., description="Echo of normalized inputs used")


class DayOption(BaseModel):
    value: int
    label: str
    is_weekend: bool


class MonthOption(BaseModel):
    value: int
    label: str


class OptionsResponse(BaseModel):
    """Dynamic metadata payload providing all available options for the frontend."""
    city: str
    areas: List[str]
    roads: List[str]
    area_to_roads: Dict[str, List[str]]
    road_to_area: Dict[str, str]
    weather_conditions: List[str]
    roadwork_options: List[bool]
    day_of_week_options: List[DayOption]
    months: List[MonthOption]
    threshold: float
    metrics: Dict[str, Any]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    """Payload for the Groq streaming AI Operations Assistant."""
    message: str = Field(..., description="User's query text")
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional preceding conversation turns for context"
    )


class HealthResponse(BaseModel):
    """Health check payload."""
    status: str
    service: str
    city: str
    model: str
    roads_monitored: int
    areas_monitored: int