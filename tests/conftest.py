"""
Shared pytest fixtures.

Key design choice: tests use an isolated in-memory SQLite database and a
mocked classifier (predictable outputs, no scikit-learn dependency),
overriding the app's FastAPI dependencies. This means:
  - tests never touch the real models/land_use_classifier.joblib file
  - tests never touch the real data/app.db file
  - tests are fast and deterministic

A separate integration test (test_classification.py::test_full_pipeline_with_real_model)
opts back into the real trained model to prove the actual end-to-end path works.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import get_classifier
from app.db.database import Base, get_db
from app.main import app


class FakeClassifier:
    """Deterministic stand-in for TileClassifier, used in unit tests."""

    def __init__(self, predicted_class: str = "Forest", confidence: float = 0.9):
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.is_loaded = True

    def predict(self, image):
        return self.predicted_class, self.confidence


@pytest.fixture()
def db_session_factory():
    """A fresh in-memory SQLite database per test."""
    # StaticPool is essential here: without it, every new connection to a
    # ':memory:' SQLite database gets its own separate, empty database,
    # so a session opened by the API request wouldn't see the table
    # created below.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def fake_classifier():
    return FakeClassifier()


@pytest.fixture()
def client(db_session_factory, fake_classifier):
    """TestClient with DB and classifier dependencies overridden."""

    def override_get_db():
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    def override_get_classifier():
        return fake_classifier

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_classifier] = override_get_classifier

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def make_test_png_bytes(size=(64, 64), color=(34, 139, 34)) -> bytes:
    """A small in-memory PNG, so tests don't depend on files outside the repo."""
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
