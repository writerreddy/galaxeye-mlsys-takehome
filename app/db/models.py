"""
ORM models.

One table for the take-home: classification_results. See DESIGN.md
section 4 for why these particular columns and indexes were chosen
(in short: everything an analyst would filter or group by is indexed,
and we store metadata rather than the raw image bytes).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _new_tile_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ClassificationResult(Base):
    __tablename__ = "classification_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Public identifier returned to clients. Generated server-side so it's
    # stable even if the original filename is reused or missing.
    tile_id: Mapped[str] = mapped_column(
        String(36), unique=True, index=True, default=_new_tile_id
    )

    filename: Mapped[str] = mapped_column(String(255))

    # SHA-256 of the raw uploaded bytes. Lets an analyst (or us, during
    # debugging) confirm two "different" tiles are actually byte-identical,
    # and gives a cheap dedup key without needing to store the image itself.
    image_hash: Mapped[str] = mapped_column(String(64), index=True)

    predicted_class: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float, index=True)

    # "accepted" | "uncertain" — see app/services/classification_service.py
    status: Mapped[str] = mapped_column(String(16), index=True)

    model_version: Mapped[str] = mapped_column(String(32), index=True)
    processing_time_ms: Mapped[float] = mapped_column(Float)

    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    __table_args__ = (
        # The query patterns we expect from an analyst (see GET /results)
        # are "give me class X above confidence Y" and "give me everything
        # in this date range" — this composite index serves the first, and
        # the per-column indexes above serve the rest without over-indexing
        # a table that will, realistically, stay small for a take-home.
        Index("ix_class_confidence", "predicted_class", "confidence"),
    )
