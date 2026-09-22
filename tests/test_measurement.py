"""
Pytest Unit Tests - Defect & Crack Measurement Module
======================================================
Tests pixel-to-cm calibration scaling and minAreaRect contour measurement logic.
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.crack_measurement import pixels_to_cm, measure_crack
from src.config import DEFAULT_RAIL_GAUGE_CM


def test_pixels_to_cm_standard_conversion():
    """Tests basic linear pixel to centimeter physical unit scaling."""
    # 240 pixels represents 143.5 cm rail gauge -> scale factor ~0.5979 cm/px
    result_cm = pixels_to_cm(pixel_measurement=120.0, reference_object_width_px=240.0, reference_object_width_cm=143.5)
    expected = 143.5 / 2.0  # 71.75 cm
    assert pytest.approx(result_cm, rel=1e-3) == expected


def test_pixels_to_cm_zero_or_negative_inputs():
    """Ensures invalid zero or negative reference inputs raise ValueError."""
    with pytest.raises(ValueError):
        pixels_to_cm(50.0, reference_object_width_px=0, reference_object_width_cm=143.5)

    with pytest.raises(ValueError):
        pixels_to_cm(50.0, reference_object_width_px=100, reference_object_width_cm=-10)


def test_measure_crack_synthetic_contour():
    """Tests measure_crack on a synthetic canvas containing a rectangle of known pixel dimensions."""
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)

    # Draw a white rectangle of 60px length x 20px width on dark background
    # Bounding box crop coordinates: [100, 100, 200, 200]
    cv2.rectangle(canvas, (120, 140), (180, 160), (255, 255, 255), -1)

    bbox = [100, 100, 200, 200]
    result = measure_crack(canvas, bbox, ref_width_px=240.0, ref_width_cm=143.5)

    assert result["contour_found"] is True
    # Length should be approximately 60 pixels, width approximately 20 pixels
    assert pytest.approx(result["length_px"], abs=3.0) == 60.0
    assert pytest.approx(result["width_px"], abs=3.0) == 20.0
    assert result["length_cm"] > 0
    assert result["width_cm"] > 0


def test_measure_crack_empty_bbox():
    """Ensures degenerate zero-area bounding boxes return graceful fallback defaults."""
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = [50, 50, 50, 50]  # Zero width/height

    result = measure_crack(canvas, bbox)
    assert result["contour_found"] is False
    assert result["length_px"] == 0.0
    assert result["width_px"] == 0.0
