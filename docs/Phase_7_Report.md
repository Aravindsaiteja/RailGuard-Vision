# RailGuard Vision — Technical Report: Phase 7 (Report Generation & Real-Time UI)

## 1. Executive Summary
Phase 7 implements automated ReportLab PDF inspection report generation (`src/report_generator.py`) and a real-time Streamlit web dashboard (`src/app.py`).

## 2. Component Implementation Details

### 2.1 ReportLab PDF Engine (`src/report_generator.py`)
Generates formal engineering PDF inspection reports featuring:
- Document header, branding banner, and timestamp metadata.
- Embedded high-resolution image with annotated defect bounding boxes.
- Defect breakdown table detailing class name, severity, physical dimensions ($\text{cm}$), and confidence score.
- Track segment structural health score ($0-100$) and historical progression status.
- Automated maintenance action directive box mapped dynamically from health scores.

### 2.2 Streamlit Web Application (`src/app.py`)
Features three execution tabs:
1. **Single Image Upload Tab**: Preprocesses image, executes YOLO detection, extracts sub-pixel measurements, logs inspection entry to SQLite, renders UI metrics, and provides a direct "Download PDF Inspection Report" button.
2. **Video File Upload Tab**: Samples every $N^{\text{th}}$ frame, overlays bounding boxes, displays a live processing FPS counter, and displays summary statistics.
3. **Live Webcam Feed Tab**: Streams camera feed with real-time detection bounding box overlays and FPS performance monitoring.
