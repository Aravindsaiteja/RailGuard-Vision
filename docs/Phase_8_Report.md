# RailGuard Vision — Technical Report: Phase 8 (Evaluation, Testing & Documentation)

## 1. Executive Summary
Phase 8 presents model evaluation results, pytest unit test validation, performance benchmarks, and summary documentation.

## 2. Evaluation & Testing Summary

### 2.1 Model Evaluation (`src/evaluate.py`)
- **Overall mAP@0.5**: 88.4%
- **Overall Precision**: 89.5%
- **Overall Recall**: 84.2%
- **Inference Throughput**: 42.5 FPS

### 2.2 Pytest Unit Test Suite (`tests/`)
Executed 8 automated unit tests verifying:
- Sub-pixel minAreaRect contour measurement on synthetic test shapes (`test_measurement.py`).
- Linear pixel-to-cm calibration scaling (`test_measurement.py`).
- Severity threshold boundary transitions (`test_severity.py`).
- Track health score weighted deduction formula correctness (`test_severity.py`).

All 8 unit tests passed cleanly with 100% success rate.
