"""
Pytest Unit Tests - Severity & Health Score Module
===================================================
Tests defect severity boundary cases and track health score deduction formulas.
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.severity import classify_severity, calculate_health_score


def test_classify_severity_boundary_cases():
    """Tests exact threshold boundary transitions for severity grading."""
    # Low Severity (< 3.0 cm length and < 0.2 cm width)
    assert classify_severity(2.9, 0.19) == "Low"

    # Moderate Severity (>= 3.0 cm length OR >= 0.2 cm width)
    assert classify_severity(3.0, 0.1) == "Moderate"
    assert classify_severity(1.0, 0.2) == "Moderate"

    # High Severity (>= 8.0 cm length OR >= 0.5 cm width)
    assert classify_severity(8.0, 0.4) == "High"
    assert classify_severity(4.0, 0.5) == "High"

    # Critical Severity (>= 15.0 cm length OR >= 1.0 cm width)
    assert classify_severity(15.0, 0.2) == "Critical"
    assert classify_severity(5.0, 1.0) == "Critical"
    assert classify_severity(22.0, 1.5) == "Critical"


def test_calculate_health_score_no_defects():
    """Ensures track segment with no defects yields perfect 100.0 health score."""
    assert calculate_health_score([]) == 100.0


def test_calculate_health_score_weighted_deductions():
    """Tests weighted deduction arithmetic for single and multiple defects."""
    # Crack base weight = 25.0, Moderate multiplier = 1.0 -> deduction = 25.0 -> Score = 75.0
    defects = [{"class_name": "crack", "severity": "Moderate"}]
    assert calculate_health_score(defects) == 75.0

    # Multiple defects:
    # 1. Broken rail base weight 40.0 * Critical multiplier 1.50 = 60.0 deduction
    # 2. Missing fastener base weight 15.0 * High multiplier 1.20 = 18.0 deduction
    # Total deduction = 78.0 -> Score = 22.0
    defects_multiple = [
        {"class_name": "broken_rail", "severity": "Critical"},
        {"class_name": "missing_fastener", "severity": "High"},
    ]
    assert calculate_health_score(defects_multiple) == 22.0


def test_calculate_health_score_lower_bound_zero():
    """Ensures health score does not drop below 0.0 despite extensive severe defects."""
    severe_defects = [{"class_name": "broken_rail", "severity": "Critical"}] * 5
    score = calculate_health_score(severe_defects)
    assert score == 0.0
