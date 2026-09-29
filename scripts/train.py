#!/usr/bin/env python3
"""
Train the tile classifier from the labelled candidate_tiles dataset and
save a local model artifact.

Usage:
    python scripts/train.py

This script:
  1. Loads images from data/be-mlsys-assignment-dataset/candidate_tiles/<ClassName>/*.png
  2. Extracts features with app.ml.preprocessing.extract_features (the
     SAME function used at inference time, so train/serve skew isn't
     possible by construction).
  3. Splits into train/validation (80/20, stratified).
  4. Trains a RandomForestClassifier.
  5. Prints validation accuracy + a per-class report.
  6. Saves {pipeline, classes, trained_at, feature_length} to
     models/land_use_classifier.joblib.

Note: the eval_set/ directory (with eval_labels.csv) is intentionally
NOT used here — it's held out for scripts/evaluate.py so that the
reported evaluation numbers are on data the model never trained on.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.ml.preprocessing import extract_features, load_image_from_bytes  # noqa: E402

RANDOM_STATE = 42


def load_dataset(train_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders found under {train_dir}")

    classes = [p.name for p in class_dirs]
    features, labels = [], []

    for class_dir in class_dirs:
        image_paths = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
        print(f"  {class_dir.name}: {len(image_paths)} images")
        for image_path in image_paths:
            raw_bytes = image_path.read_bytes()
            image = load_image_from_bytes(raw_bytes)
            features.append(extract_features(image))
            labels.append(class_dir.name)

    return np.stack(features), np.array(labels), classes


def main() -> None:
    print(f"Loading training data from {settings.train_dir}")
    X, y, classes = load_dataset(settings.train_dir)
    print(f"Loaded {len(X)} images across {len(classes)} classes: {classes}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}  Validation: {len(X_val)}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"Trained in {time.perf_counter() - start:.2f}s")

    y_pred = model.predict(X_val)
    print("\nValidation classification report:")
    print(classification_report(y_val, y_pred, digits=3))

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": model,
        # IMPORTANT: must be in the exact order of model.classes_, since
        # that's the order predict_proba() returns probabilities in.
        "classes": model.classes_.tolist(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_length": X.shape[1],
        "model_version": settings.model_version,
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
    }
    joblib.dump(artifact, settings.model_path)
    print(f"\nSaved model artifact to {settings.model_path}")


if __name__ == "__main__":
    main()
