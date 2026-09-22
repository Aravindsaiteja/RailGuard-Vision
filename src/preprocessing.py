"""
RailGuard Vision - Image Preprocessing Pipeline
================================================
Provides computer vision functions for image resizing, normalization,
contrast enhancement via CLAHE, and edge-preserving noise reduction.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union
from pathlib import Path
import sys

# Add parent directory to system path if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import TARGET_IMAGE_SIZE


def preprocess_image(
    image: np.ndarray, target_size: Tuple[int, int] = TARGET_IMAGE_SIZE
) -> np.ndarray:
    """
    Resizes input image to target dimensions and normalizes pixel intensities to [0, 1].

    Args:
        image (np.ndarray): Input image in BGR or RGB format.
        target_size (Tuple[int, int]): Desired width and height (default: 640x640).

    Returns:
        np.ndarray: Preprocessed image array normalized to float32 range [0.0, 1.0].
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image provided to preprocess_image.")

    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def enhance_contrast(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance
    fine track cracks and surface defect details.

    Args:
        image (np.ndarray): Input image (grayscale or 3-channel uint8 array).
        clip_limit (float): Threshold for contrast limiting (default: 2.0).
        tile_grid_size (Tuple[int, int]): Grid size for histogram equalization (default: 8x8).

    Returns:
        np.ndarray: CLAHE contrast-enhanced image.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image provided to enhance_contrast.")

    # Convert normalized float [0, 1] to uint8 if necessary
    if image.dtype != np.uint8:
        image_uint8 = (image * 255.0).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        image_uint8 = image.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
        # Convert to LAB color space and apply CLAHE to L-channel to preserve original color tones
        lab = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        enhanced_image = clahe.apply(image_uint8)

    return enhanced_image


def denoise_image(
    image: np.ndarray, gaussian_kernel: Tuple[int, int] = (5, 5), median_ksize: int = 5
) -> np.ndarray:
    """
    Applies Gaussian blur followed by Median filtering to reduce visual noise while
    preserving structural edge boundaries of rails and cracks.

    Args:
        image (np.ndarray): Input image array (uint8).
        gaussian_kernel (Tuple[int, int]): Kernel dimensions for Gaussian filtering.
        median_ksize (int): Odd integer kernel size for Median filtering.

    Returns:
        np.ndarray: Denoised image array.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image provided to denoise_image.")

    # Ensure input is uint8
    if image.dtype != np.uint8:
        img_uint8 = (image * 255.0).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        img_uint8 = image.copy()

    # Step 1: Gaussian filter to smooth high-frequency background noise
    gaussian_filtered = cv2.GaussianBlur(img_uint8, gaussian_kernel, sigmaX=0)

    # Step 2: Median filter to eliminate impulse salt-and-pepper noise while preserving sharp boundaries
    denoised = cv2.medianBlur(gaussian_filtered, median_ksize)

    return denoised


def create_synthetic_rail_sample() -> np.ndarray:
    """Generates a synthetic railway track image for standalone debugging and demonstration."""
    canvas = np.full((640, 640, 3), 120, dtype=np.uint8)
    # Draw dark ballast noise
    noise = np.random.randint(-30, 30, (640, 640, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw two vertical parallel steel rails (bright silver/metallic)
    cv2.rectangle(canvas, (180, 0), (220, 640), (180, 185, 190), -1)
    cv2.rectangle(canvas, (420, 0), (460, 640), (180, 185, 190), -1)

    # Draw rail ties / sleepers horizontally
    for y in range(40, 640, 100):
        cv2.rectangle(canvas, (100, y), (540, y + 30), (50, 40, 30), -1)

    # Re-draw rails over ties
    cv2.rectangle(canvas, (180, 0), (220, 640), (180, 185, 190), -1)
    cv2.rectangle(canvas, (420, 0), (460, 640), (180, 185, 190), -1)

    # Draw a synthetic crack on the left rail
    pts = np.array([[195, 250], [202, 270], [198, 290], [205, 310]], np.int32)
    cv2.polylines(canvas, [pts], isClosed=False, color=(20, 20, 20), thickness=3)

    return canvas


if __name__ == "__main__":
    print("[RailGuard Preprocessing] Running standalone pipeline test...")

    # Create synthetic test image
    raw_image = create_synthetic_rail_sample()

    # Run processing pipeline steps
    norm_image = preprocess_image(raw_image, target_size=(640, 640))
    enhanced_img = enhance_contrast(raw_image)
    denoised_img = denoise_image(enhanced_img)

    # Visual Inspection via Matplotlib
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Original Raw Image")
    axes[0].axis("off")

    axes[1].imshow(norm_image)
    axes[1].set_title("2. Resized & Normalized")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title("3. CLAHE Enhanced")
    axes[2].axis("off")

    axes[3].imshow(cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB))
    axes[3].set_title("4. Denoised (Gaussian+Median)")
    axes[3].axis("off")

    plt.tight_layout()
    output_demo = Path(__file__).resolve().parent.parent / "reports" / "preprocessing_demo.png"
    plt.savefig(output_demo, dpi=150)
    plt.close()
    print(f"[RailGuard Preprocessing] Success! Pipeline debug figure saved to: {output_demo}")
