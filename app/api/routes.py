"""
API routes.

Endpoints:
  GET  /health              liveness + model-loaded check
  POST /tiles/classify      the required "thin slice": classify + store one tile
  GET  /results             analyst querying (filterable list) - see DESIGN.md section 7
  GET  /results/{tile_id}   fetch a single stored result by id
"""

from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.database import get_db
from app.db.models import ClassificationResult
from app.ml.classifier import classifier as default_classifier
from app.schemas.classification import (
    ClassificationResponse,
    ClassificationResultOut,
    HealthResponse,
)
from app.services.classification_service import (
    FileTooLargeError,
    UnsupportedContentTypeError,
    classify_and_store,
)
from app.ml.preprocessing import InvalidImageError

logger = get_logger(__name__)
router = APIRouter()


def get_classifier():
    """FastAPI dependency so tests can override this with a mock classifier
    without touching disk or importing scikit-learn."""
    return default_classifier


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health(clf=Depends(get_classifier)) -> HealthResponse:
    from app.core.config import settings

    return HealthResponse(
        status="ok",
        model_loaded=clf.is_loaded,
        model_version=settings.model_version,
    )


@router.post(
    "/tiles/classify",
    response_model=ClassificationResponse,
    status_code=status.HTTP_200_OK,
    tags=["tiles"],
)
async def classify_tile(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    clf=Depends(get_classifier),
) -> ClassificationResponse:
    if not clf.is_loaded:
        # Model artifact missing/not loaded is a server-side problem, not
        # a bad request from the client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Classifier model is not available on the server",
        )

    raw_bytes = await file.read()

    try:
        result = classify_and_store(
            db=db,
            classifier=clf,
            raw_bytes=raw_bytes,
            filename=file.filename or "unknown",
            content_type=file.content_type,
        )
    except (UnsupportedContentTypeError, InvalidImageError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        )
    except Exception:
        # Anything unexpected: log the real cause server-side, never leak
        # internals (stack traces, file paths, SQL) to the client.
        logger.exception("Unexpected failure while classifying tile")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while classifying tile",
        )

    return ClassificationResponse(
        tile_id=result.tile_id,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        status=result.status,
        model_version=result.model_version,
        processing_time_ms=result.processing_time_ms,
    )


@router.get("/results", response_model=list[ClassificationResultOut], tags=["results"])
def list_results(
    db: Session = Depends(get_db),
    predicted_class: str | None = Query(None, description="Filter by exact class name"),
    status_filter: str | None = Query(
        None, alias="status", description="Filter by 'accepted' or 'uncertain'"
    ),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    start_date: datetime | None = Query(None, description="Inclusive lower bound (ISO 8601)"),
    end_date: datetime | None = Query(None, description="Inclusive upper bound (ISO 8601)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[ClassificationResult]:
    stmt = select(ClassificationResult)

    if predicted_class is not None:
        stmt = stmt.where(ClassificationResult.predicted_class == predicted_class)
    if status_filter is not None:
        stmt = stmt.where(ClassificationResult.status == status_filter)
    if min_confidence is not None:
        stmt = stmt.where(ClassificationResult.confidence >= min_confidence)
    if start_date is not None:
        stmt = stmt.where(ClassificationResult.processed_at >= start_date)
    if end_date is not None:
        stmt = stmt.where(ClassificationResult.processed_at <= end_date)

    stmt = stmt.order_by(ClassificationResult.processed_at.desc()).limit(limit).offset(offset)
    return list(db.scalars(stmt).all())


@router.get(
    "/results/{tile_id}", response_model=ClassificationResultOut, tags=["results"]
)
def get_result(tile_id: str, db: Session = Depends(get_db)) -> ClassificationResult:
    stmt = select(ClassificationResult).where(ClassificationResult.tile_id == tile_id)
    result = db.scalars(stmt).first()
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No result found for tile_id '{tile_id}'",
        )
    return result
