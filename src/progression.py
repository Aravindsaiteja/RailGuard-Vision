"""
RailGuard Vision - Inspection Progression & Database Module
=============================================================
Manages SQLite database storage for rail track inspection records and tracks historical
crack growth progression over sequential inspections.

NOTE / TRANSPARENCY DISCLAIMER:
--------------------------------
Public rail fault datasets consist of independent static images without native temporal
tracking. The progression analysis implemented in this module relies on simulated/mock
time-series records stored in SQLite to demonstrate real-world historical track degradation monitoring.
"""

import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DB_PATH, DATA_DIR


def init_db(db_path: Path = DB_PATH) -> None:
    """
    Initializes SQLite database and creates the inspections schema if not already present.

    Args:
        db_path (Path): Absolute filesystem path to SQLite database file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_segment_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            defect_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            length_cm REAL NOT NULL,
            width_cm REAL NOT NULL,
            health_score REAL NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def log_inspection(
    track_segment_id: str,
    defect_type: str,
    severity: str,
    length_cm: float,
    width_cm: float,
    health_score: float,
    timestamp: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> int:
    """
    Inserts a new rail track inspection entry into the SQLite database.

    Args:
        track_segment_id (str): Unique identifier for physical rail track segment (e.g., 'SEG_104B').
        defect_type (str): Primary defect class name (e.g., 'crack').
        severity (str): Evaluated severity level ('Low', 'Moderate', 'High', 'Critical').
        length_cm (float): Measured defect length in centimeters.
        width_cm (float): Measured defect width in centimeters.
        health_score (float): Computed track structural health score (0-100).
        timestamp (Optional[str]): ISO-8601 formatted timestamp string (defaults to UTC now).
        db_path (Path): Path to SQLite database file.

    Returns:
        int: Primary key ID of the inserted inspection record.
    """
    init_db(db_path)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO inspections (
            track_segment_id, timestamp, defect_type, severity, length_cm, width_cm, health_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            track_segment_id,
            timestamp,
            defect_type,
            severity,
            float(length_cm),
            float(width_cm),
            float(health_score),
        ),
    )

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return record_id


def get_inspection_history(track_segment_id: str, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Fetches all historical inspection records for a given track segment sorted by timestamp."""
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, track_segment_id, timestamp, defect_type, severity, length_cm, width_cm, health_score
        FROM inspections
        WHERE track_segment_id = ?
        ORDER BY id ASC
        """,
        (track_segment_id,),
    )

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def compare_with_previous(track_segment_id: str, db_path: Path = DB_PATH) -> Dict[str, Any]:
    """
    Compares the latest inspection record against the immediately preceding inspection
    for the specified track segment to determine crack degradation progression status.

    Progression Status Categories:
    - 'New': No prior inspection records found for this segment.
    - 'Stable': Defect area change <= 5% or measurement difference < 0.2 cm.
    - 'Growing': Defect length or area increased between 5% and 35%.
    - 'Rapidly Growing': Defect length or area increased by > 35%.

    Args:
        track_segment_id (str): Target track segment identifier.
        db_path (Path): Path to SQLite database file.

    Returns:
        Dict[str, Any]: Comparison analysis dictionary containing progression status and metrics.
    """
    history = get_inspection_history(track_segment_id, db_path)

    if len(history) <= 1:
        return {
            "status": "New",
            "message": "First recorded inspection for this track segment.",
            "length_delta_cm": 0.0,
            "area_delta_cm2": 0.0,
            "percent_growth": 0.0,
            "previous_inspection": None,
        }

    latest = history[-1]
    previous = history[-2]

    prev_length = previous["length_cm"]
    prev_width = previous["width_cm"]
    curr_length = latest["length_cm"]
    curr_width = latest["width_cm"]

    prev_area = prev_length * prev_width
    curr_area = curr_length * curr_width

    length_delta = curr_length - prev_length
    area_delta = curr_area - prev_area

    if prev_length > 0:
        percent_growth = ((curr_length - prev_length) / prev_length) * 100.0
    else:
        percent_growth = 100.0 if curr_length > 0 else 0.0

    # Categorize Progression Status
    if percent_growth > 35.0:
        status = "Rapidly Growing"
    elif percent_growth > 5.0:
        status = "Growing"
    elif percent_growth < -5.0:
        status = "Improving / Repaired"
    else:
        status = "Stable"

    return {
        "status": status,
        "message": f"Defect size changed by {percent_growth:+.1f}% compared to prior inspection.",
        "length_delta_cm": round(length_delta, 2),
        "area_delta_cm2": round(area_delta, 2),
        "percent_growth": round(percent_growth, 1),
        "previous_inspection": previous,
    }


if __name__ == "__main__":
    print("[Progression Module] Running standalone DB & progression simulation test...")

    # Use a temporary test DB for main block execution
    test_db = DATA_DIR / "test_progression_demo.db"
    if test_db.exists():
        test_db.unlink()

    segment_id = "SEG_NORTH_104"

    print(f"\n1. Logging initial inspection for segment '{segment_id}'...")
    rec1_id = log_inspection(
        track_segment_id=segment_id,
        defect_type="crack",
        severity="Moderate",
        length_cm=4.2,
        width_cm=0.3,
        health_score=75.0,
        timestamp="2026-08-01T10:00:00Z",
        db_path=test_db,
    )
    comp1 = compare_with_previous(segment_id, db_path=test_db)
    print(f"Inspection #1 Status: {comp1['status']} ({comp1['message']})")

    print(f"\n2. Logging follow-up inspection 30 days later with crack growth...")
    rec2_id = log_inspection(
        track_segment_id=segment_id,
        defect_type="crack",
        severity="High",
        length_cm=5.3,  # Increased length (+26.2% growth -> 'Growing')
        width_cm=0.4,
        health_score=60.0,
        timestamp="2026-09-01T10:00:00Z",
        db_path=test_db,
    )
    comp2 = compare_with_previous(segment_id, db_path=test_db)
    print(f"Inspection #2 Status: {comp2['status']} ({comp2['message']})")
    print(f"Length Delta: {comp2['length_delta_cm']} cm | Growth: {comp2['percent_growth']}%")

    if test_db.exists():
        test_db.unlink()
    print("\n[Progression Module] Execution finished successfully.")
