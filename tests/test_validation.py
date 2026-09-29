"""Tests for the lower-level validation and preprocessing logic."""

import numpy as np
import pytest
from PIL import Image

from app.ml.preprocessing import (
    FEATURE_VECTOR_LENGTH,
    InvalidImageError,
    extract_features,
    load_image_from_bytes,
)
from app.services.classification_service import (
    FileTooLargeError,
    UnsupportedContentTypeError,
    _validate_upload,
)
from tests.conftest import make_test_png_bytes


def test_load_image_from_bytes_accepts_valid_png():
    image = load_image_from_bytes(make_test_png_bytes())
    assert image.mode == "RGB"


def test_load_image_from_bytes_rejects_garbage():
    with pytest.raises(InvalidImageError):
        load_image_from_bytes(b"not an image at all")


def test_load_image_from_bytes_rejects_empty():
    with pytest.raises(InvalidImageError):
        load_image_from_bytes(b"")


def test_extract_features_returns_fixed_length_vector():
    image = Image.new("RGB", (64, 64), color=(10, 200, 50))
    vector = extract_features(image)
    assert vector.shape == (FEATURE_VECTOR_LENGTH,)
    assert np.all(np.isfinite(vector))


def test_extract_features_is_deterministic():
    image = Image.new("RGB", (64, 64), color=(10, 200, 50))
    v1 = extract_features(image)
    v2 = extract_features(image)
    np.testing.assert_array_equal(v1, v2)


def test_extract_features_handles_non_standard_size():
    # Real tiles are 64x64, but the extractor should not assume that.
    image = Image.new("RGB", (128, 96), color=(5, 5, 200))
    vector = extract_features(image)
    assert vector.shape == (FEATURE_VECTOR_LENGTH,)


def test_validate_upload_rejects_unsupported_content_type():
    with pytest.raises(UnsupportedContentTypeError):
        _validate_upload(b"whatever", "application/pdf")


def test_validate_upload_rejects_oversized_file():
    from app.core.config import settings

    oversized = b"0" * (settings.max_upload_size_bytes + 1)
    with pytest.raises(FileTooLargeError):
        _validate_upload(oversized, "image/png")


def test_validate_upload_accepts_normal_png():
    _validate_upload(make_test_png_bytes(), "image/png")  # should not raise
