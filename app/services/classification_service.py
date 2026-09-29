"""
Orchestrates a single classification request:

  validate bytes -> decode image -> run classifier -> apply confidence
  policy -> persist result -> return it

This is the one place that ties the ML layer (app/ml) to the storage
layer (app/db) together, so it's the natural seam for unit testing with
a mocked classifier (see tests/test_classification.py).
"""

import hashlib
import time

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import ClassificationResult
from app.ml.classifier import TileClassifier
from app.ml.preprocessing import InvalidImageError, load_image_from_bytes

logger = get_logger(__name__)


class UnsupportedContentTypeError(ValueError):
    pass


class FileTooLargeError(ValueError):
    pass


def _validate_upload(raw_bytes: bytes, content_type: str | None) -> None:
    if content_type is not None and content_type not in settings.allowed_content_types:
        raise UnsupportedContentTypeError(
            f"Unsupported content type '{content_type}'. "
            f"Allowed: {', '.join(settings.allowed_content_types)}"
        )
    if len(raw_bytes) > settings.max_upload_size_bytes:
        raise FileTooLargeError(
            f"File exceeds maximum allowed size of "
            f"{settings.max_upload_size_bytes} bytes"
        )


def classify_and_store(
    db: Session,
    classifier: TileClassifier,
    raw_bytes: bytes,
    filename: str,
    content_type: str | None,
) -> ClassificationResult:
    """Run the full pipeline for one uploaded tile and persist the result.

    Raises UnsupportedContentTypeError, FileTooLargeError, or
    InvalidImageError on bad input — callers (the API layer) translate
    these into the appropriate HTTP status codes.
    """
    _validate_upload(raw_bytes, content_type)

    image_hash = hashlib.sha256(raw_bytes).hexdigest()

    start = time.perf_counter()
    image = load_image_from_bytes(raw_bytes)  # raises InvalidImageError
    predicted_class, confidence = classifier.predict(image)
    elapsed_ms = (time.perf_counter() - start) * 1000

    status = "accepted" if confidence >= settings.confidence_threshold else "uncertain"

    result = ClassificationResult(
        filename=filename,
        image_hash=image_hash,
        predicted_class=predicted_class,
        confidence=confidence,
        status=status,
        model_version=settings.model_version,
        processing_time_ms=elapsed_ms,
    )
    db.add(result)
    db.commit()
    db.refresh(result)

    logger.info(
        "Classified tile_id=%s filename=%s class=%s confidence=%.3f status=%s "
        "took_ms=%.1f",
        result.tile_id,
        filename,
        predicted_class,
        confidence,
        status,
        elapsed_ms,
    )
    return result
