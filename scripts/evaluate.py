#!/usr/bin/env python3
"""
Evaluate the trained model against the held-out eval_set/ + eval_labels.csv.

Usage:
    python scripts/evaluate.py

This is a separate, never-trained-on dataset (see README.txt shipped
with the dataset), so these numbers are a genuine held-out estimate,
not a training-set score dressed up as one. As the assignment notes,
accuracy is not the primary grading criterion here — this script exists
to produce honest numbers to discuss, not to chase a high score.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.ml.classifier import TileClassifier  # noqa: E402
from app.ml.preprocessing import load_image_from_bytes  # noqa: E402


def load_eval_labels(csv_path: Path) -> dict[str, str]:
    labels = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["filename"]] = row["true_label"]
    return labels


def main() -> None:
    classifier = TileClassifier()
    classifier.load()

    labels_by_filename = load_eval_labels(settings.eval_labels_csv)
    print(f"Loaded {len(labels_by_filename)} ground-truth labels from {settings.eval_labels_csv}")

    y_true, y_pred, confidences = [], [], []

    for filename, true_label in sorted(labels_by_filename.items()):
        image_path = settings.eval_dir / filename
        if not image_path.exists():
            print(f"  WARNING: {filename} listed in labels but not found in eval_set/, skipping")
            continue
        image = load_image_from_bytes(image_path.read_bytes())
        predicted_class, confidence = classifier.predict(image)

        y_true.append(true_label)
        y_pred.append(predicted_class)
        confidences.append(confidence)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    confidences_arr = np.array(confidences)

    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    print(f"\nOverall accuracy on eval_set: {accuracy:.3f} ({len(y_true)} samples)")

    print("\nPer-class report:")
    print(classification_report(y_true_arr, y_pred_arr, digits=3))

    print("Confusion matrix (rows=true, cols=predicted):")
    class_labels = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=class_labels)
    header = "".join(f"{c[:10]:>12}" for c in class_labels)
    print(f"{'':>14}{header}")
    for label, row in zip(class_labels, cm):
        row_str = "".join(f"{v:>12}" for v in row)
        print(f"{label[:12]:>14}{row_str}")

    threshold = settings.confidence_threshold
    accepted_mask = confidences_arr >= threshold
    n_accepted = int(accepted_mask.sum())
    if n_accepted:
        accepted_accuracy = accuracy_score(y_true_arr[accepted_mask], y_pred_arr[accepted_mask])
    else:
        accepted_accuracy = float("nan")

    print(
        f"\nAt confidence_threshold={threshold}: "
        f"{n_accepted}/{len(y_true)} predictions marked 'accepted' "
        f"(accuracy among them: {accepted_accuracy:.3f})"
    )
    print(
        f"{len(y_true) - n_accepted}/{len(y_true)} predictions marked 'uncertain' "
        f"(accuracy among them: "
        f"{accuracy_score(y_true_arr[~accepted_mask], y_pred_arr[~accepted_mask]) if (~accepted_mask).sum() else float('nan'):.3f})"
    )

    # Save a confusion matrix image for the write-up / interview discussion.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(
            ax=ax, xticks_rotation=45, colorbar=False
        )
        out_path = Path(__file__).resolve().parent.parent / "models" / "confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(out_path)
        print(f"\nSaved confusion matrix plot to {out_path}")
    except ImportError:
        print("\n(matplotlib not installed — skipping confusion matrix plot)")


if __name__ == "__main__":
    main()
