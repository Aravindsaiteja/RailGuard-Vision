# RailGuard Vision — Technical Report: Phase 3 (Local Project Scaffold)

## 1. Executive Summary
Phase 3 establishes the clean, modular Python computer-vision repository structure in VS Code, configuring dependencies, virtual environment configurations, central paths, constants, and `.gitignore` rules.

## 2. Directory Layout
```
Rail Gaurd Vision/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── detect.py
│   ├── crack_measurement.py
│   ├── severity.py
│   ├── progression.py
│   ├── report_generator.py
│   ├── evaluate.py
│   └── app.py
├── models/
│   └── best.pt
├── data/
│   └── inspections.db
├── reports/
│   └── results_summary.md
├── tests/
│   ├── test_measurement.py
│   └── test_severity.py
├── notebooks/
│   ├── Phase1_Dataset_Preparation.ipynb
│   └── Phase2_YOLOv8_Training.ipynb
├── docs/
│   ├── Phase_1_Report.md
│   └── ...
├── requirements.txt
├── .gitignore
└── README.md
```

## 3. Configuration & Environment Setup
- `requirements.txt`: Defines versions for `ultralytics`, `opencv-python-headless`, `reportlab`, `streamlit`, `albumentations`, and `pytest`.
- `src/config.py`: Encapsulates paths (`MODELS_DIR`, `DATA_DIR`, `REPORTS_DIR`, `DB_PATH`), physical calibration constants (standard gauge $143.5\text{ cm}$), severity matrices, and health score deduction weights.
