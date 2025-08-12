"""
MARS-GIS API Router Configuration

This module organizes and configures all API endpoints according to the
ISO/IEC 29148:2011 requirements specification.
"""
# flake8: noqa
# mypy: ignore-errors

try:
    from datetime import datetime
    from typing import Any, Dict, List, Optional

    from fastapi import APIRouter, HTTPException, Path, Query
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None
    HTTPException = None
    BaseModel = object
    def Field(**kwargs):  # type: ignore
        return None

import asyncio
import json  # noqa: F401
from pathlib import Path as PathLib  # noqa: F401

# Import MARS-GIS core modules
from mars_gis.core.config import settings

# Import data processing modules
try:
    from mars_gis.data.nasa_client import NASADataClient
    from mars_gis.data.usgs_client import USGSDataClient
    DATA_CLIENTS_AVAILABLE = True
except ImportError:
    DATA_CLIENTS_AVAILABLE = False
    NASADataClient = None
    USGSDataClient = None

# Import ML modules
try:
    from mars_gis.models.earth_mars_transfer import EarthMarsTransferModel
    from mars_gis.models.landing_site_optimization import LandingSiteOptimizer
    from mars_gis.models.terrain_models import TerrainClassifier
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False
    EarthMarsTransferModel = None
    LandingSiteOptimizer = None
    TerrainClassifier = None

# Import mission planning modules
try:
    from mars_gis.database.models import MissionPlan
    from mars_gis.geospatial.path_planning import MarsPathPlanner
    MISSION_MODULES_AVAILABLE = True
except ImportError:
    MISSION_MODULES_AVAILABLE = False
    MarsPathPlanner = None
    MissionPlan = None

# Pydantic models for API requests/responses
if FASTAPI_AVAILABLE:

    class MarsDataQuery(BaseModel):
        """Mars data query parameters - Requirement FR-DM-001"""
        lat_min: float = Field(..., ge=-90, le=90, description="Minimum latitude")
        lat_max: float = Field(..., ge=-90, le=90, description="Maximum latitude")
        lon_min: float = Field(..., ge=-180, le=180, description="Minimum longitude")
        lon_max: float = Field(..., ge=-180, le=180, description="Maximum longitude")
        data_types: List[str] = Field(default=["imagery", "elevation"], description="Types of data to retrieve")
        resolution: Optional[str] = Field(default="medium", description="Data resolution: low, medium, high")
        start_date: Optional[datetime] = Field(default=None, description="Start date for temporal filtering")
        end_date: Optional[datetime] = Field(default=None, description="End date for temporal filtering")

    class InferenceRequest(BaseModel):
        """ML model inference request - Requirement FR-ML-001, FR-ML-002, FR-ML-003"""
        model_type: str = Field(..., description="Model type: terrain_classification, landing_site, transfer_learning")
        input_data: Dict[str, Any] = Field(..., description="Input data for inference")
        coordinates: Optional[List[float]] = Field(default=None, description="[lat, lon] coordinates if applicable")
        confidence_threshold: Optional[float] = Field(default=0.8, ge=0, le=1, description="Minimum confidence threshold")

    class MissionCreateRequest(BaseModel):
        """Mission creation request - Requirement FR-MP-001"""
        name: str = Field(..., min_length=1, max_length=255, description="Mission name")
        description: str = Field(..., min_length=1, description="Mission description")
        mission_type: str = Field(..., description="Mission type: landing, rover, orbital")
        target_coordinates: List[float] = Field(..., description="Target [lat, lon] coordinates")
        planned_date: datetime = Field(..., description="Planned mission date")
        priority: str = Field(default="medium", description="Mission priority: low, medium, high, critical")
        constraints: Optional[Dict[str, Any]] = Field(default=None, description="Mission constraints and requirements")

    class MissionResponse(BaseModel):
        """Mission response model - Requirement FR-MP-001"""
        id: str
        name: str
        description: str
        mission_type: str
        target_coordinates: List[float]
        status: str
        safety_score: Optional[float]
        scientific_priority: Optional[float]
        created_at: datetime
        updated_at: datetime

    class StreamSubscribeRequest(BaseModel):
        """Stream subscription request - Requirement FR-DM-003"""
        stream_type: str = Field(..., description="Stream type: satellite_imagery, weather_data, mission_telemetry")
        coordinates: Optional[List[float]] = Field(default=None, description="[lat, lon] for location-based streams")
        filters: Optional[Dict[str, Any]] = Field(default=None, description="Stream filtering criteria")
        update_frequency: Optional[str] = Field(default="real-time", description="Update frequency: real-time, hourly, daily")

    class APIResponse(BaseModel):
        """Standard API response wrapper"""
        success: bool
        data: Any
        message: str
        timestamp: datetime = Field(default_factory=datetime.utcnow)
        trace_id: Optional[str] = None

else:
    # Fallback classes if FastAPI not available
    class MarsDataQuery:
        pass

    class InferenceRequest:
        pass

    class MissionCreateRequest:
        pass

    class MissionResponse:
        pass

    class StreamSubscribeRequest:
        pass

    class APIResponse:
        pass

# Export router creation function
def create_api_router():
    """Create and configure API router with all endpoints."""
    if not FASTAPI_AVAILABLE:
        return None

    # Import the implementation from routes module
    try:
        from .routes import create_api_router as _create_router
        return _create_router()
    except ImportError:
        return None

# Export for easy importing
__all__ = [
    "create_api_router",
    "FASTAPI_AVAILABLE", "APIRouter", "HTTPException", "Query", "Path",
    "StreamingResponse", "asyncio", "datetime", "Optional",
    "MarsDataQuery", "InferenceRequest", "MissionCreateRequest",
    "StreamSubscribeRequest", "APIResponse",
    "DATA_CLIENTS_AVAILABLE", "ML_MODELS_AVAILABLE", "MISSION_MODULES_AVAILABLE",
    "NASADataClient", "USGSDataClient", "MarsPathPlanner",
    "EarthMarsTransferModel", "LandingSiteOptimizer", "TerrainClassifier",
    # Expose module-level router from routes for tests expecting it
    "router",
]

# Backward-compat import for `router` symbol used by some tests
try:
    from .routes import router
except Exception:
    router = None
