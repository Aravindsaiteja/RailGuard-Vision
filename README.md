#  RailGuard Vision — YOLOv8 Railway Track Defect Detection System


**RailGuard Vision** is an end-to-end computer-vision system for real-time railway track defect detection, physical crack dimension measurement (in centimeters), track structural health scoring, time-series growth tracking, automated PDF inspection report generation, and interactive web dashboard monitoring.

---

##  Key Features

1. **YOLOv8 Object Detection**: Fine-tuned on railway defect classes (`crack`, `missing_fastener`, `broken_rail`, `joint_fault`) achieving **88.4% mAP@0.5** at **42.5 FPS**.
2. **OpenCV Image Processing**: CLAHE contrast enhancement for low-light micro-fissure highlight, dual Gaussian/Median edge-preserving denoising, and Hough Line track ROI segmentation.
3. **Sub-Pixel Crack Measurement**: Fits minimum-area oriented bounding rectangles (`cv2.minAreaRect`) and converts pixel measurements into physical centimeters calibrated against standard rail gauge ($143.5\text{ cm}$).
4. **Severity Grading & Track Health Scoring**: Automated threshold rules grading severity (Low, Moderate, High, Critical) and computing a 0–100 track health score with weighted defect deductions.
5. **SQLite Progression Database**: Stores inspection logs in `data/inspections.db` and computes temporal growth rates (New, Stable, Growing, Rapidly Growing).
6. **Automated PDF Report Engine**: ReportLab PDF generator creating engineering inspection reports with annotated imagery, defect tables, health scores, and maintenance action recommendations.
7. **Streamlit Multi-Tab Dashboard**: Renders real-time inspection for Single Images, Video Streams, and Live Webcam feeds with real-time FPS monitoring.

---

## Repository Structure

```
Rail Gaurd Vision/
├── src/
│   ├── config.py             # Central path, model, and threshold configurations
│   ├── preprocessing.py      # CLAHE contrast enhancement & noise filtering
│   ├── segmentation.py       # Canny edge detection & Hough line ROI masking
│   ├── detect.py             # YOLODetector model wrapper & mock fallback mode
│   ├── crack_measurement.py  # MinAreaRect contour fitting & pixel-to-cm calibration
│   ├── severity.py           # Severity rules & track health score deduction algorithm
│   ├── progression.py        # SQLite inspection database & temporal growth tracking
│   ├── report_generator.py   # ReportLab PDF engineering report generator engine
│   ├── evaluate.py           # Test set evaluation metrics & FPS benchmarking
│   └── app.py                # Streamlit web dashboard application
├── models/
│   └── best.pt               # Fine-tuned YOLOv8 model weights
├── data/
│   └── inspections.db        # SQLite inspection database
├── reports/
│   ├── results_summary.md    # Model evaluation metrics & analytical summary
│   └── ...                   # Generated PDF inspection reports
├── tests/
│   ├── test_measurement.py   # Pytest unit tests for crack measurement
│   └── test_severity.py      # Pytest unit tests for severity & health score
├── notebooks/
│   ├── Phase1_Dataset_Preparation.ipynb  # Colab dataset setup & Albumentations
│   └── Phase2_YOLOv8_Training.ipynb      # Colab YOLOv8 GPU training & validation
├── docs/
│   ├── Phase_1_Report.md to Phase_8_Report.md
│   └── Final_Comprehensive_Report.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Installation & Setup

### 1. Clone & Navigate to Repository
```bash
cd "c:/Users/aravi/Downloads/MY_PROJECTS/Rail Gaurd Vision"
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

##  Running the Streamlit App

Launch the interactive dashboard:
```bash
streamlit run src/app.py
```
Open your browser at `http://localhost:8501` to access:
- **Tab 1: Single Image Upload**: Inspect track photos, view measurements, and download PDF reports.
- **Tab 2: Video Stream File**: Process track inspection videos frame-by-frame with live FPS counters.
- **Tab 3: Live Webcam Feed**: Stream live camera feeds for real-time defect localization.

---

##  Running Pytest Unit Tests

Execute the automated unit test suite:
```bash
pytest tests/
```

To run individual module demonstration scripts:
```bash
python src/preprocessing.py
python src/segmentation.py
python src/detect.py
python src/crack_measurement.py
python src/severity.py
python src/progression.py
python src/report_generator.py
python src/evaluate.py
```

---

##  Documentation & Reports

- **Final Capstone Report**: [`docs/Final_Comprehensive_Report.md`](docs/Final_Comprehensive_Report.md)
- **Results Summary & Discussion**: [`reports/results_summary.md`](reports/results_summary.md)
- **Phase Reports**: [`docs/Phase_1_Report.md`](docs/Phase_1_Report.md) through [`docs/Phase_8_Report.md`](docs/Phase_8_Report.md)
- **Colab Notebooks**: [`notebooks/Phase1_Dataset_Preparation.ipynb`](notebooks/Phase1_Dataset_Preparation.ipynb) & [`notebooks/Phase2_YOLOv8_Training.ipynb`](notebooks/Phase2_YOLOv8_Training.ipynb)
