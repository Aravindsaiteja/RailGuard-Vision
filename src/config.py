"""
RailGuard Vision - Configuration & Global Constants
===================================================
Central configuration file defining path locations, model settings,
physical calibration parameters, severity threshold matrices, health score weights,
and database configuration.
"""

from pathlib import Path
from typing import Dict, Any

# Root Directory paths
SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SRC_DIR.parent

# Artifact & Data Directories
MODELS_DIR: Path = PROJECT_ROOT / "models"
DATA_DIR: Path = PROJECT_ROOT / "data"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, REPORTS_DIR, DOCS_DIR, TESTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database Configuration
DB_PATH: Path = DATA_DIR / "inspections.db"

# YOLO Model Configuration
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "best.pt"
YOLO_CONF_THRESHOLD: float = 0.40
YOLO_IOU_THRESHOLD: float = 0.45
TARGET_IMAGE_SIZE: tuple = (640, 640)

# Physical Calibration Defaults
# Standard Railway Gauge is 143.5 cm (4 ft 8.5 in)
# Typical Rail Head Width is ~7.0 cm
DEFAULT_RAIL_GAUGE_CM: float = 143.5
DEFAULT_RAIL_HEAD_WIDTH_CM: float = 7.0

# Severity Threshold Definitions (Length & Width in centimeters)
SEVERITY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "CRITICAL": {"min_length_cm": 15.0, "min_width_cm": 1.0},
    "HIGH": {"min_length_cm": 8.0, "min_width_cm": 0.5},
    "MODERATE": {"min_length_cm": 3.0, "min_width_cm": 0.2},
    "LOW": {"min_length_cm": 0.0, "min_width_cm": 0.0},
}

# Health Score Deduction Weights per Defect Type
DEFECT_WEIGHTS: Dict[str, float] = {
    "broken_rail": 40.0,
    "crack": 25.0,
    "joint_fault": 20.0,
    "missing_fastener": 15.0,
    "minor_defect": 10.0,
}

# Maintenance Action Recommendation Mapping based on Track Health Score (0-100)
MAINTENANCE_RECOMMENDATIONS: Dict[str, str] = {
    "HEALTHY": "Continue regular monitoring scheduled per standard maintenance manual.",
    "MONITOR": "Schedule routine track inspection and maintenance within 7 days.",
    "URGENT": "Schedule priority repair within 48 hours; restrict train speed on segment.",
    "CRITICAL": "Immediate repair required! Halt rail traffic on segment immediately.",
}

# Class Names mapping for YOLO model
CLASS_NAMES: Dict[int, str] = {
    0: "crack",
    1: "missing_fastener",
    2: "broken_rail",
    3: "joint_fault",
}
