"""
Image validation and feature extraction.

This module is imported by the API, the training script, and the
evaluation script, so "how we turn an image into a feature vector" is
defined in exactly one place. That matters: if training and inference
ever preprocess images differently, the model silently breaks.

Feature strategy (deliberately simple — see DESIGN.md section 2/6):

  1. Decode + resize to a fixed size (handles tiles that aren't exactly
     64x64, even though the supplied dataset is).
  2. Per-channel color histograms (16 bins x 3 channels = 48 features).
     Land-use classes in EuroSAT-style imagery (forest, water, urban,
     crop) are strongly color-correlated, so this carries most of the
     signal.
  3. Per-channel mean + std (6 features) — cheap global brightness/
     contrast signal.
  4. A coarse edge-density histogram (16 bins) from a grayscale edge
     map — a rough texture proxy that helps separate visually "busy"
     classes (Residential, Industrial, Highway) from "smooth" ones
     (Forest, SeaLake).

No deep learning, no GPU, no pretrained weights to manage — the whole
feature extractor is a few dozen lines of NumPy/Pillow and is easy to
read, test, and explain line-by-line in an interview.
"""

import io

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

IMAGE_SIZE = (64, 64)
N_COLOR_BINS = 16
N_EDGE_BINS = 16

FEATURE_VECTOR_LENGTH = (N_COLOR_BINS * 3) + 6 + N_EDGE_BINS


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes are not a decodable, supported image."""


def load_image_from_bytes(raw_bytes: bytes) -> Image.Image:
    """Decode raw bytes into a PIL Image, validating it's a real image.

    Raises InvalidImageError with a client-safe message on any failure
    (corrupt file, unsupported/truncated data, zero-byte upload, etc.).
    """
    if not raw_bytes:
        raise InvalidImageError("Uploaded file is empty")

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()  # force full decode now, not lazily later
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Invalid or corrupt image file") from exc

    return image.convert("RGB")


def extract_features(image: Image.Image) -> np.ndarray:
    """Turn a PIL RGB image into a fixed-length float feature vector."""
    image = image.resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3), scaled 0..1

    features: list[float] = []

    # 1. Per-channel color histograms
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=N_COLOR_BINS, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-8  # normalize -> robust to nothing else changing
        features.extend(hist.tolist())

    # 2. Per-channel mean + std
    for channel in range(3):
        features.append(float(arr[:, :, channel].mean()))
        features.append(float(arr[:, :, channel].std()))

    # 3. Coarse edge-density histogram (texture proxy)
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edges, dtype=np.float32) / 255.0
    edge_hist, _ = np.histogram(edge_arr, bins=N_EDGE_BINS, range=(0.0, 1.0))
    edge_hist = edge_hist.astype(np.float32)
    edge_hist /= edge_hist.sum() + 1e-8
    features.extend(edge_hist.tolist())

    vector = np.asarray(features, dtype=np.float32)
    assert vector.shape[0] == FEATURE_VECTOR_LENGTH
    return vector
