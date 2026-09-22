# RailGuard Vision — Technical Report: Phase 5 (Detection & Crack Measurement)

## 1. Executive Summary
Phase 5 implements YOLOv8 object detection inference and real-world defect measurement algorithms in `src/detect.py` and `src/crack_measurement.py`.

## 2. Technical Architecture

### 2.1 YOLODetector Wrapper Class
- Encapsulates Ultralytics YOLO inference (`models/best.pt`).
- Returns structured JSON/dictionary bounding boxes, class labels, and confidence scores.
- Includes a robust fallback mock detector to enable local testing without pre-downloaded weights.

### 2.2 Sub-Pixel Contour Extraction & `minAreaRect` Fitting
- Crops detected defect region.
- Applies Otsu adaptive thresholding and morphological closing (`cv2.morphologyEx`).
- Extracts defect contour and fits a minimum-area oriented bounding rectangle (`cv2.minAreaRect`).
- Extracts pixel length ($\text{length}_{\text{px}}$) and width ($\text{width}_{\text{px}}$).

### 2.3 Physical Unit Calibration Scaling
Scale conversion formula:
$$\text{Scale Factor} = \frac{\text{Reference Width}_{\text{cm}}}{\text{Reference Width}_{\text{px}}} \quad (\text{cm/pixel})$$
$$\text{Length}_{\text{cm}} = \text{Length}_{\text{px}} \times \text{Scale Factor}$$
Calibrated against standard rail gauge ($143.5\text{ cm}$).
