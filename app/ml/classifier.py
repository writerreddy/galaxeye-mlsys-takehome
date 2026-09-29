"""
Thin wrapper around the trained model artifact.

The app never trains a model at startup or at request time — it only
loads a pre-trained artifact from disk (see scripts/train.py for how
that artifact is produced). This keeps request latency low and
deployment reproducible: the exact model bytes that were validated
during training are the ones running in production.
"""

from pathlib import Path

import joblib
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.preprocessing import extract_features
from PIL import Image

logger = get_logger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but no model artifact is available."""


class TileClassifier:
    """Loads a scikit-learn pipeline (features -> class probabilities) from disk."""

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or settings.model_path
        self._pipeline = None
        self._classes: list[str] | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            logger.error("Model artifact not found at %s", self.model_path)
            raise ModelNotLoadedError(
                f"Model artifact not found at {self.model_path}. "
                "Run `python scripts/train.py` first."
            )
        artifact = joblib.load(self.model_path)
        self._pipeline = artifact["pipeline"]
        self._classes = list(artifact["classes"])
        logger.info(
            "Loaded model from %s (classes=%s)", self.model_path, self._classes
        )

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def predict(self, image: Image.Image) -> tuple[str, float]:
        """Return (predicted_class, confidence) for a single PIL image."""
        if self._pipeline is None:
            raise ModelNotLoadedError("Classifier used before load() was called")

        features = extract_features(image).reshape(1, -1)
        probabilities = self._pipeline.predict_proba(features)[0]

        best_index = int(np.argmax(probabilities))
        predicted_class = self._classes[best_index]
        confidence = float(probabilities[best_index])
        return predicted_class, confidence


# Module-level singleton, loaded once at app startup (see app/main.py)
# and reused across requests — this is the "don't train/load per request"
# requirement from the assignment.
classifier = TileClassifier()
