"""
RailGuard Vision - Defect Severity & Track Health Score Module
================================================================
Classifies defect severity levels based on physical dimensions (cm) and
calculates segment track health scores using a weighted deduction model.
"""

from typing import List, Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import SEVERITY_THRESHOLDS, DEFECT_WEIGHTS


def classify_severity(length_cm: float, width_cm: float) -> str:
    """
    Classifies crack defect severity into 'Low', 'Moderate', 'High', or 'Critical'
    based on physical length and width measurements in centimeters.

    Rule Evaluation Order:
    1. Critical: length >= 15.0 cm OR width >= 1.0 cm
    2. High: length >= 8.0 cm OR width >= 0.5 cm
    3. Moderate: length >= 3.0 cm OR width >= 0.2 cm
    4. Low: Below moderate thresholds

    Args:
        length_cm (float): Physical crack length in centimeters.
        width_cm (float): Physical crack width in centimeters.

    Returns:
        str: Severity grade ("Critical", "High", "Moderate", "Low").
    """
    length = max(0.0, float(length_cm))
    width = max(0.0, float(width_cm))

    crit = SEVERITY_THRESHOLDS["CRITICAL"]
    if length >= crit["min_length_cm"] or width >= crit["min_width_cm"]:
        return "Critical"

    high = SEVERITY_THRESHOLDS["HIGH"]
    if length >= high["min_length_cm"] or width >= high["min_width_cm"]:
        return "High"

    mod = SEVERITY_THRESHOLDS["MODERATE"]
    if length >= mod["min_length_cm"] or width >= mod["min_width_cm"]:
        return "Moderate"

    return "Low"


def calculate_health_score(defects: List[Dict[str, Any]]) -> float:
    """
    Computes overall track segment structural health score on a 0 - 100 point scale.

    Deduction Formula:
    Health Score = max(0.0, 100.0 - Sum(Base_Defect_Weight * Severity_Multiplier))

    Severity Multipliers:
    - Critical: 1.50
    - High: 1.20
    - Moderate: 1.00
    - Low: 0.60

    Args:
        defects (List[Dict[str, Any]]): List of defect dictionaries containing 'defect_type'
                                        or 'class_name' and 'severity'.

    Returns:
        float: Computed track health score bounded between 0.0 (Worst) and 100.0 (Perfect).
    """
    if not defects:
        return 100.0

    severity_multipliers = {
        "Critical": 1.50,
        "High": 1.20,
        "Moderate": 1.00,
        "Low": 0.60,
    }

    total_deduction = 0.0

    for defect in defects:
        defect_type = defect.get("class_name", defect.get("defect_type", "minor_defect"))
        severity = defect.get("severity", "Low")

        base_weight = DEFECT_WEIGHTS.get(defect_type, 15.0)
        multiplier = severity_multipliers.get(severity, 1.00)

        total_deduction += base_weight * multiplier

    health_score = max(0.0, 100.0 - total_deduction)
    return round(health_score, 1)


if __name__ == "__main__":
    print("[Severity & Health Score] Running standalone module test...")

    # Test cases for severity classification
    test_cases = [
        (1.5, 0.1, "Low"),
        (4.5, 0.3, "Moderate"),
        (9.0, 0.6, "High"),
        (18.0, 1.2, "Critical"),
    ]

    print("\n--- Severity Classification Test Cases ---")
    for l, w, expected in test_cases:
        res = classify_severity(l, w)
        print(f"Length: {l}cm, Width: {w}cm -> Result: '{res}' | Pass: {res == expected}")

    # Test case for health score calculation
    sample_defects = [
        {"class_name": "crack", "severity": "Moderate"},
        {"class_name": "missing_fastener", "severity": "High"},
    ]
    score = calculate_health_score(sample_defects)
    print(f"\nSample Track Segment Defects: {sample_defects}")
    print(f"Computed Health Score: {score} / 100.0")
