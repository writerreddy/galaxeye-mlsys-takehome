"""Tests for the persistence layer, independent of the HTTP API."""

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import ClassificationResult
from app.services.classification_service import classify_and_store
from tests.conftest import FakeClassifier, make_test_png_bytes


def _make_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)()


def test_classify_and_store_persists_a_row():
    db = _make_session()
    classifier = FakeClassifier(predicted_class="SeaLake", confidence=0.95)

    result = classify_and_store(
        db=db,
        classifier=classifier,
        raw_bytes=make_test_png_bytes(),
        filename="lake.png",
        content_type="image/png",
    )

    stored = db.scalars(
        select(ClassificationResult).where(ClassificationResult.tile_id == result.tile_id)
    ).first()

    assert stored is not None
    assert stored.predicted_class == "SeaLake"
    assert stored.confidence == 0.95
    assert stored.status == "accepted"
    assert stored.filename == "lake.png"
    assert len(stored.image_hash) == 64  # sha256 hex digest length
    assert stored.processing_time_ms >= 0


def test_classify_and_store_marks_low_confidence_uncertain():
    db = _make_session()
    classifier = FakeClassifier(predicted_class="Highway", confidence=0.2)

    result = classify_and_store(
        db=db,
        classifier=classifier,
        raw_bytes=make_test_png_bytes(),
        filename="road.png",
        content_type="image/png",
    )

    assert result.status == "uncertain"


def test_same_image_bytes_produce_same_hash():
    db = _make_session()
    classifier = FakeClassifier()
    image_bytes = make_test_png_bytes()

    result1 = classify_and_store(
        db=db, classifier=classifier, raw_bytes=image_bytes,
        filename="a.png", content_type="image/png",
    )
    result2 = classify_and_store(
        db=db, classifier=classifier, raw_bytes=image_bytes,
        filename="b.png", content_type="image/png",
    )

    assert result1.image_hash == result2.image_hash
    assert result1.tile_id != result2.tile_id  # each upload is still its own row


def test_tile_id_is_unique_per_row():
    db = _make_session()
    classifier = FakeClassifier()

    ids = set()
    for _ in range(5):
        result = classify_and_store(
            db=db, classifier=classifier, raw_bytes=make_test_png_bytes(),
            filename="x.png", content_type="image/png",
        )
        ids.add(result.tile_id)

    assert len(ids) == 5
