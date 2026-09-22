# RailGuard Vision — Technical Report: Phase 6 (Severity, Health Score & Progression DB)

## 1. Executive Summary
Phase 6 establishes rule-based defect severity classification, track segment structural health scoring, SQLite inspection database storage, and temporal crack growth progression tracking.

## 2. Decision Logic & Formulas

### 2.1 Severity Classification Matrix (`src/severity.py`)
- **Critical**: $\text{Length} \ge 15.0\text{ cm}$ OR $\text{Width} \ge 1.0\text{ cm}$
- **High**: $\text{Length} \ge 8.0\text{ cm}$ OR $\text{Width} \ge 0.5\text{ cm}$
- **Moderate**: $\text{Length} \ge 3.0\text{ cm}$ OR $\text{Width} \ge 0.2\text{ cm}$
- **Low**: Below moderate thresholds

### 2.2 Weighted Track Health Score Deduction
$$\text{Health Score} = \max\left(0.0, 100.0 - \sum \left(\text{Base Weight} \times \text{Severity Multiplier}\right)\right)$$
- Base Weights: `broken_rail` (40), `crack` (25), `joint_fault` (20), `missing_fastener` (15).
- Severity Multipliers: Critical (1.50), High (1.20), Moderate (1.00), Low (0.60).

### 2.3 SQLite Inspection Database Schema & Progression
Stored in `data/inspections.db`:
- Table `inspections`: `id`, `track_segment_id`, `timestamp`, `defect_type`, `severity`, `length_cm`, `width_cm`, `health_score`.
- `compare_with_previous(track_segment_id)` computes percentage size change over sequential inspections to assign status: `New`, `Stable`, `Growing` ($+5\%\text{ to }+35\%$), or `Rapidly Growing` ($>+35\%$).
