"""
Pydantic schemas for API request/response bodies.

Kept separate from the ORM models (app/db/models.py) on purpose: the API
contract and the storage schema are allowed to drift independently. Right
now they happen to look similar, which is fine for a take-home.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ClassificationResponse(BaseModel):
    tile_id: str
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: str
    model_version: str
    processing_time_ms: float


class ClassificationResultOut(BaseModel):
    """Shape returned by GET /results and GET /results/{tile_id}."""

    model_config = ConfigDict(from_attributes=True)

    tile_id: str
    filename: str
    predicted_class: str
    confidence: float
    status: str
    model_version: str
    processing_time_ms: float
    processed_at: datetime
    image_hash: str


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
