# RailGuard Vision — Technical Report: Phase 4 (Image Processing Pipeline)

## 1. Executive Summary
Phase 4 implements classical computer vision pre-processing and region-of-interest segmentation algorithms in `src/preprocessing.py` and `src/segmentation.py`.

## 2. Algorithms & Implementation Details

### 2.1 Contrast Limited Adaptive Histogram Equalization (CLAHE)
In `enhance_contrast(image)`:
- Image converted to LAB color space.
- CLAHE algorithm ($clip\_limit=2.0$, $grid=(8,8)$) applied to the luminance ($L$) channel.
- Re-merged with $A, B$ color channels and converted back to BGR.
- Enhances micro-fissure visibility under shadow or rust conditions.

### 2.2 Dual Filtering (Gaussian + Median Denoising)
In `denoise_image(image)`:
- $5\times 5$ Gaussian kernel smooths background ballast grain.
- $5\times 5$ Median filter removes impulse noise while retaining sharp crack boundary edges.

### 2.3 Hough Line Rail Detection & Region Masking
In `detect_track_lines(image)` and `segment_track_region(image)`:
- Canny edge detection ($T_1=50, T_2=150$).
- Probabilistic Hough Transform (`cv2.HoughLinesP`, $min\_length=80, max\_gap=20$).
- Convex hull ROI polygon mask generation isolating steel rail beds from background terrain.
