"""
RailGuard Vision - Defect & Crack Measurement Module
=====================================================
Extracts defect region contours, calculates minimum-area rectangle dimensions in pixels,
and applies physical calibration scaling to compute real-world length and width in centimeters.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DEFAULT_RAIL_GAUGE_CM, DEFAULT_RAIL_HEAD_WIDTH_CM


def pixels_to_cm(
    pixel_measurement: float,
    reference_object_width_px: float,
    reference_object_width_cm: float = DEFAULT_RAIL_GAUGE_CM,
) -> float:
    """
    Converts pixel measurement to real-world centimeters using physical reference calibration.

    Scale Factor = reference_object_width_cm / reference_object_width_px (cm/pixel)

    Args:
        pixel_measurement (float): Length or width in pixels.
        reference_object_width_px (float): Measured width of reference object in pixels.
        reference_object_width_cm (float): Known physical width of reference object in centimeters.

    Returns:
        float: Calculated measurement in centimeters.
    """
    if reference_object_width_px <= 0:
        raise ValueError("reference_object_width_px must be greater than zero.")
    if reference_object_width_cm <= 0:
        raise ValueError("reference_object_width_cm must be greater than zero.")

    scale_factor = reference_object_width_cm / float(reference_object_width_px)
    return float(pixel_measurement * scale_factor)


def measure_crack(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    ref_width_px: float = 240.0,
    ref_width_cm: float = DEFAULT_RAIL_GAUGE_CM,
) -> Dict[str, Any]:
    """
    Crops defect ROI, detects crack contour via adaptive thresholding, fits a minimum-area
    rectangle, and calculates pixel + physical (cm) length and width.

    Args:
        image (np.ndarray): Full BGR image.
        bbox (Tuple[int, int, int, int]): Bounding box array [x1, y1, x2, y2].
        ref_width_px (float): Measured rail gauge distance in pixels (calibration reference).
        ref_width_cm (float): Known physical rail gauge distance in centimeters.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'length_px': float
            - 'width_px': float
            - 'length_cm': float
            - 'width_cm': float
            - 'contour_found': bool
            - 'box_points': np.ndarray (4 corner points of minAreaRect for drawing)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided to measure_crack.")

    x1, y1, x2, y2 = bbox
    h_img, w_img = image.shape[:2]

    # Clip coordinates to image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    crop_w = x2 - x1
    crop_h = y2 - y1

    # Fallback response if bounding box is degenerate
    if crop_w <= 2 or crop_h <= 2:
        return {
            "length_px": 0.0,
            "width_px": 0.0,
            "length_cm": 0.0,
            "width_cm": 0.0,
            "contour_found": False,
            "box_points": np.array([]),
        }

    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()

    # Step 1: Preprocess crop via Otsu Adaptive Thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 2: Morphological closing to join broken crack segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 3: Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours that span the full crop border (background mask artifact)
    valid_contours = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw >= crop_w - 2 and bh >= crop_h - 2:
            continue
        valid_contours.append(cnt)

    if not valid_contours:
        # Try non-inverted thresholding in case of bright feature on dark background
        _, thresh_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_bin = cv2.morphologyEx(thresh_bin, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < crop_w - 2 or bh < crop_h - 2:
                valid_contours.append(cnt)

    if not valid_contours:
        # Graceful fallback: use bounding box dimensions if no fine contour is extracted
        length_px = float(max(crop_w, crop_h))
        width_px = float(min(crop_w, crop_h))
        length_cm = pixels_to_cm(length_px, ref_width_px, ref_width_cm)
        width_cm = pixels_to_cm(width_px, ref_width_px, ref_width_cm)
        return {
            "length_px": round(length_px, 2),
            "width_px": round(width_px, 2),
            "length_cm": round(length_cm, 2),
            "width_cm": round(width_cm, 2),
            "contour_found": False,
            "box_points": np.array([]),
        }

    # Extract largest valid contour corresponding to the defect structure
    largest_contour = max(valid_contours, key=cv2.contourArea)

    # Step 4: Minimum-Area Oriented Bounding Rectangle
    rect = cv2.minAreaRect(largest_contour)
    (center_x, center_y), (rect_w, rect_h), angle = rect

    length_px = max(rect_w, rect_h)
    width_px = min(rect_w, rect_h)

    # Convert rect corners relative to full image space
    box_points_crop = cv2.boxPoints(rect)
    box_points_full = box_points_crop + np.array([x1, y1])

    # Step 5: Convert pixel dimensions to physical centimeters
    length_cm = pixels_to_cm(length_px, ref_width_px, ref_width_cm)
    width_cm = pixels_to_cm(width_px, ref_width_px, ref_width_cm)

    return {
        "length_px": round(length_px, 2),
        "width_px": round(width_px, 2),
        "length_cm": round(length_cm, 2),
        "width_cm": round(width_cm, 2),
        "contour_found": True,
        "box_points": box_points_full.astype(int),
    }


if __name__ == "__main__":
    from src.preprocessing import create_synthetic_rail_sample
    from src.detect import YOLODetector
    import matplotlib.pyplot as plt

    print("[Crack Measurement] Running end-to-end detection + measurement test...")
    sample_img = create_synthetic_rail_sample()

    detector = YOLODetector()
    detections = detector.detect(sample_img)

    for i, det in enumerate(detections):
        bbox = det["bbox"]
        measurement = measure_crack(sample_img, bbox, ref_width_px=240.0, ref_width_cm=DEFAULT_RAIL_GAUGE_CM)
        print(f"Defect #{i+1} ({det['class_name']}):")
        print(f"  Bounding Box: {bbox}")
        print(f"  Pixel Dimensions: {measurement['length_px']}px x {measurement['width_px']}px")
        print(f"  Physical Dimensions: {measurement['length_cm']}cm (Length) x {measurement['width_cm']}cm (Width)")
        print(f"  Contour Extracted: {measurement['contour_found']}")

        # Draw contour minAreaRect onto visualization copy
        vis_img = detector.draw_annotations(sample_img, detections)
        if measurement["contour_found"] and len(measurement["box_points"]) > 0:
            cv2.drawContours(vis_img, [measurement["box_points"]], 0, (255, 255, 0), 2)

        output_demo = Path(__file__).resolve().parent.parent / "reports" / "measurement_demo.png"
        cv2.imwrite(str(output_demo), vis_img)
        print(f"[Crack Measurement] Visual demonstration saved to: {output_demo}")
