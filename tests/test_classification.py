"""
Tests for POST /tiles/classify.

Unit tests below use the fake, deterministic classifier from conftest.py
so they verify our API/service/storage logic, not model accuracy. The
one integration test at the bottom uses the real trained model to prove
the full pipeline (upload -> classifier -> database -> response) works
end-to-end, which is what the assignment explicitly asks for.
"""

from tests.conftest import make_test_png_bytes


def test_classify_accepted_prediction(client, fake_classifier):
    fake_classifier.predicted_class = "Forest"
    fake_classifier.confidence = 0.91

    files = {"file": ("tile.png", make_test_png_bytes(), "image/png")}
    response = client.post("/tiles/classify", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "Forest"
    assert body["confidence"] == 0.91
    assert body["status"] == "accepted"
    assert body["model_version"]
    assert body["processing_time_ms"] >= 0
    assert body["tile_id"]


def test_classify_low_confidence_is_marked_uncertain(client, fake_classifier):
    fake_classifier.predicted_class = "River"
    fake_classifier.confidence = 0.35  # below the default 0.60 threshold

    files = {"file": ("tile.png", make_test_png_bytes(), "image/png")}
    response = client.post("/tiles/classify", files=files)

    assert response.status_code == 200
    assert response.json()["status"] == "uncertain"


def test_classify_rejects_invalid_image(client):
    garbage_bytes = b"this is not a png or jpeg"
    files = {"file": ("tile.png", garbage_bytes, "image/png")}

    response = client.post("/tiles/classify", files=files)

    assert response.status_code == 400
    assert "detail" in response.json()


def test_classify_rejects_empty_file(client):
    files = {"file": ("tile.png", b"", "image/png")}
    response = client.post("/tiles/classify", files=files)
    assert response.status_code == 400


def test_classify_rejects_unsupported_content_type(client):
    files = {"file": ("tile.txt", b"hello world", "text/plain")}
    response = client.post("/tiles/classify", files=files)
    assert response.status_code == 400


def test_classify_persists_result_retrievable_by_id(client, fake_classifier):
    fake_classifier.predicted_class = "Industrial"
    fake_classifier.confidence = 0.8

    files = {"file": ("tile.png", make_test_png_bytes(), "image/png")}
    post_response = client.post("/tiles/classify", files=files)
    tile_id = post_response.json()["tile_id"]

    get_response = client.get(f"/results/{tile_id}")
    assert get_response.status_code == 200
    body = get_response.json()
    assert body["tile_id"] == tile_id
    assert body["predicted_class"] == "Industrial"
    assert body["filename"] == "tile.png"


def test_get_unknown_result_returns_404(client):
    response = client.get("/results/does-not-exist")
    assert response.status_code == 404


def test_full_pipeline_with_real_model():
    """Integration test: real model, real (temp) database, no mocks.

    Proves: upload -> preprocessing -> real classifier -> real DB write
    -> API response, end to end. Requires models/land_use_classifier.joblib
    to exist (run `python scripts/train.py` first, same as README says).
    """
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db.database import Base, get_db
    from app.main import app
    from app.ml.classifier import TileClassifier

    real_classifier = TileClassifier()
    real_classifier.load()

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    from app.api.routes import get_classifier

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_classifier] = lambda: real_classifier

    try:
        with TestClient(app) as test_client:
            # A real, plausible "forest-ish" solid dark-green tile.
            image_bytes = make_test_png_bytes(color=(20, 90, 30))
            files = {"file": ("integration_tile.png", image_bytes, "image/png")}
            response = test_client.post("/tiles/classify", files=files)

            assert response.status_code == 200
            body = response.json()
            assert body["predicted_class"] in [
                "AnnualCrop",
                "Forest",
                "Highway",
                "Industrial",
                "Residential",
                "River",
                "SeaLake",
            ]
            assert 0.0 <= body["confidence"] <= 1.0
            assert body["status"] in ("accepted", "uncertain")

            # Confirm it's actually queryable afterwards.
            list_response = test_client.get("/results")
            assert list_response.status_code == 200
            assert any(r["tile_id"] == body["tile_id"] for r in list_response.json())
    finally:
        app.dependency_overrides.clear()
