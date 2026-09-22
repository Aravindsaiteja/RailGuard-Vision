"""
RailGuard Vision - Track Region Segmentation Module
===================================================
Provides edge detection, Probabilistic Hough Line Transformation, and
Region of Interest (ROI) polygon masking to isolate railway track beds.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.preprocessing import create_synthetic_rail_sample, enhance_contrast, denoise_image


def detect_track_lines(
    image: np.ndarray,
    canny_threshold1: int = 50,
    canny_threshold2: int = 150,
    hough_threshold: int = 40,
    min_line_length: int = 80,
    max_line_gap: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects prominent linear rail structures using Canny edge detection and
    Probabilistic Hough Line Transform (HoughLinesP).

    Args:
        image (np.ndarray): Input BGR or grayscale image array.
        canny_threshold1 (int): Lower hysteresis threshold for Canny edge detector.
        canny_threshold2 (int): Upper hysteresis threshold for Canny edge detector.
        hough_threshold (int): Accumulator threshold for HoughLinesP.
        min_line_length (int): Minimum vector length to accept as a line segment.
        max_line_gap (int): Maximum allowable gap between segments to merge.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - edges (np.ndarray): Binary Canny edge map.
            - lines (np.ndarray): Array of line segments (shape: N x 1 x 4) containing [x1, y1, x2, y2].
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image provided to detect_track_lines.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Step 1: Compute Canny Edges
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    # Step 2: Probabilistic Hough Line Detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        lines = np.array([])

    return edges, lines


def segment_track_region(
    image: np.ndarray, lines: np.ndarray, default_padding: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a convex polygonal Region of Interest (ROI) mask enclosing the detected track lines.

    Args:
        image (np.ndarray): Source BGR image.
        lines (np.ndarray): Line segments array returned by detect_track_lines.
        default_padding (int): Pixel margin added to bounding region boundaries.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - segmented_image (np.ndarray): Image with background outside track region masked black.
            - mask (np.ndarray): Binary single-channel uint8 polygon mask (255 for track ROI, 0 elsewhere).
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image provided to segment_track_region.")

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if len(lines) > 0:
        # Extract all (x, y) line endpoints
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([[x1, y1], [x2, y2]])
        points_arr = np.array(points, dtype=np.int32)

        # Compute Convex Hull around rail line points to form ROI polygon
        hull = cv2.convexHull(points_arr)
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        # Fallback default trapezoidal ROI covering the central 60% corridor of rail tracks
        pts = np.array(
            [
                [int(w * 0.15), h],
                [int(w * 0.35), int(h * 0.2)],
                [int(w * 0.65), int(h * 0.2)],
                [int(w * 0.85), h],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 255)

    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image, mask


if __name__ == "__main__":
    print("[RailGuard Segmentation] Running standalone pipeline test...")

    raw_img = create_synthetic_rail_sample()
    denoised_img = denoise_image(enhance_contrast(raw_img))

    # Detect lines and segment track
    edges, lines = detect_track_lines(denoised_img)
    segmented_track, roi_mask = segment_track_region(raw_img, lines)

    # Draw detected Hough lines on visualization canvas
    lines_visual = raw_img.copy()
    if len(lines) > 0:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(lines_visual, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Plot pipeline steps side by side
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Input Track Image")
    axes[0].axis("off")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title(f"2. Canny Edges")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(lines_visual, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"3. Hough Rail Lines ({len(lines)} detected)")
    axes[2].axis("off")

    axes[3].imshow(cv2.cvtColor(segmented_track, cv2.COLOR_BGR2RGB))
    axes[3].set_title("4. Masked Track ROI")
    axes[3].axis("off")

    plt.tight_layout()
    output_path = Path(__file__).resolve().parent.parent / "reports" / "segmentation_demo.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[RailGuard Segmentation] Success! Visual output saved to: {output_path}")
