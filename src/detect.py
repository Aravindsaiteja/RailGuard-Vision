"""
RailGuard Vision - YOLOv8 Defect Detector Module
================================================
Provides the YOLODetector wrapper class for deep-learning inference on rail images,
returning structured bounding box detections, class names, and confidence scores.
Includes a robust fallback mock detector for local testing prior to model training.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Union, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DEFAULT_MODEL_PATH, YOLO_CONF_THRESHOLD, CLASS_NAMES


class YOLODetector:
    """Wrapper class for YOLOv8 railway defect detection."""

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH, conf_threshold: float = YOLO_CONF_THRESHOLD):
        """
        Initializes the YOLOv8 detector.

        Args:
            model_path (Union[str, Path]): Path to trained YOLOv8 weights (.pt file).
            conf_threshold (float): Minimum confidence score threshold for valid detections.
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.model = None
        self.is_mock = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads Ultralytics YOLO model or falls back to mock detector if weights are absent."""
        if self.model_path.exists():
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(self.model_path))
                print(f"[YOLODetector] Successfully loaded YOLO weights from {self.model_path}")
            except Exception as e:
                print(f"[YOLODetector] Warning: Could not initialize YOLO model ({e}). Using mock detector mode.")
                self.is_mock = True
        else:
            print(f"[YOLODetector] Notice: Weight file '{self.model_path}' not found. Initializing mock detector mode.")
            self.is_mock = True

    def detect(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Runs object detection on the provided image.

        Args:
            image (np.ndarray): Input image in BGR format.
            conf_threshold (Optional[float]): Overrides instance confidence threshold if specified.

        Returns:
            List[Dict[str, Any]]: List of detection dictionaries containing:
                - 'bbox': [x1, y1, x2, y2] (ints)
                - 'class_id': int
                - 'class_name': str
                - 'confidence': float
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image provided to YOLODetector.detect()")

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        detections: List[Dict[str, Any]] = []

        if self.is_mock:
            return self._run_mock_detection(image, threshold)

        try:
            results = self.model(image, conf=threshold, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = CLASS_NAMES.get(cls_id, f"defect_{cls_id}")

                    detections.append(
                        {
                            "bbox": xyxy,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(conf, 4),
                        }
                    )
        except Exception as e:
            print(f"[YOLODetector] Error during YOLO inference ({e}). Falling back to mock detection.")
            return self._run_mock_detection(image, threshold)

        return detections

    def _run_mock_detection(self, image: np.ndarray, conf_threshold: float) -> List[Dict[str, Any]]:
        """Mock detection logic providing deterministic test bounding boxes for local evaluation."""
        h, w = image.shape[:2]

        # Simulate synthetic crack detection box centered on left rail region
        mock_box = [int(w * 0.28), int(h * 0.35), int(w * 0.35), int(h * 0.52)]
        mock_conf = 0.895

        if mock_conf >= conf_threshold:
            return [
                {
                    "bbox": mock_box,
                    "class_id": 0,
                    "class_name": "crack",
                    "confidence": mock_conf,
                }
            ]
        return []

    def draw_annotations(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draws bounding boxes, labels, and confidence percentage overlays onto an image copy.

        Args:
            image (np.ndarray): Original BGR image.
            detections (List[Dict[str, Any]]): Detection list returned by detect().

        Returns:
            np.ndarray: BGR image with drawn visual annotations.
        """
        annotated = image.copy()

        # Color scheme per defect class (BGR)
        color_palette = {
            "crack": (0, 0, 255),            # Bright Red
            "missing_fastener": (0, 165, 255), # Bright Orange
            "broken_rail": (0, 0, 139),       # Dark Red/Crimson
            "joint_fault": (255, 191, 0),     # Deep Cyan/Blue
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = color_palette.get(cls_name, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background box
            label_text = f"{cls_name} {conf * 100:.1f}%"
            (font_w, font_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, y1 - font_h - 6),
                (x1 + font_w + 6, y1),
                color,
                -1,
            )
            # Draw text
            cv2.putText(
                annotated,
                label_text,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated


if __name__ == "__main__":
    from src.preprocessing import create_synthetic_rail_sample
    import matplotlib.pyplot as plt

    print("[YOLODetector] Running standalone detector test...")
    sample_img = create_synthetic_rail_sample()

    detector = YOLODetector()
    results = detector.detect(sample_img)
    annotated_img = detector.draw_annotations(sample_img, results)

    print(f"[YOLODetector] Detections found: {results}")

    # Save output demo graphic
    output_demo = Path(__file__).resolve().parent.parent / "reports" / "detection_demo.png"
    cv2.imwrite(str(output_demo), annotated_img)
    print(f"[YOLODetector] Visual output saved to: {output_demo}")
