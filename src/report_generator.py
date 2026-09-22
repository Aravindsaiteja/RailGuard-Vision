"""
RailGuard Vision - Automated PDF Inspection Report Generator
============================================================
Uses ReportLab to generate structured engineering PDF inspection reports featuring
metadata headers, annotated imagery, tabular defect statistics, track health scores,
historical progression status, and automated maintenance directives.
"""

import os
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import REPORTS_DIR, MAINTENANCE_RECOMMENDATIONS

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)


def get_maintenance_recommendation(health_score: float) -> str:
    """Maps numerical track health score (0-100) to standard maintenance action recommendation."""
    if health_score >= 80.0:
        return MAINTENANCE_RECOMMENDATIONS["HEALTHY"]
    elif health_score >= 60.0:
        return MAINTENANCE_RECOMMENDATIONS["MONITOR"]
    elif health_score >= 40.0:
        return MAINTENANCE_RECOMMENDATIONS["URGENT"]
    else:
        return MAINTENANCE_RECOMMENDATIONS["CRITICAL"]


def generate_report(inspection_data: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
    """
    Generates a PDF inspection report from inspection results dictionary.

    Args:
        inspection_data (Dict[str, Any]): Dictionary containing:
            - 'track_segment_id': str
            - 'timestamp': str
            - 'health_score': float
            - 'progression_status': str
            - 'defects': List[Dict[str, Any]] (defect list with class_name, severity, length_cm, width_cm, confidence)
            - 'annotated_image': np.ndarray (BGR image array)
        output_path (Optional[Path]): Destination PDF file path.

    Returns:
        Path: Path to generated PDF report.
    """
    segment_id = inspection_data.get("track_segment_id", "SEG_UNKNOWN")
    timestamp = inspection_data.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    health_score = float(inspection_data.get("health_score", 100.0))
    progression_status = inspection_data.get("progression_status", "New / Not Evaluated")
    defects = inspection_data.get("defects", [])
    annotated_img = inspection_data.get("annotated_image", None)

    if output_path is None:
        filename = f"RailGuard_Report_{segment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = REPORTS_DIR / filename
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary image saving for ReportLab embedding
    temp_img_path = output_path.parent / f"_temp_img_{os.getpid()}.png"
    if annotated_img is not None and isinstance(annotated_img, np.ndarray):
        cv2.imwrite(str(temp_img_path), annotated_img)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
    )

    story = []

    # Title & Branding Banner
    story.append(Paragraph("RAILGUARD VISION — DEFECT INSPECTION REPORT", title_style))
    story.append(
        Paragraph("Autonomous Computer Vision Railway Track Infrastructure Health Analysis", subtitle_style)
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=12))

    # Inspection Metadata Card Table
    recommendation = get_maintenance_recommendation(health_score)
    meta_data = [
        [
            Paragraph("<b>Track Segment ID:</b>", body_style),
            Paragraph(str(segment_id), body_style),
            Paragraph("<b>Inspection Date:</b>", body_style),
            Paragraph(str(timestamp), body_style),
        ],
        [
            Paragraph("<b>Track Health Score:</b>", body_style),
            Paragraph(f"<b>{health_score} / 100</b>", body_style),
            Paragraph("<b>Progression Status:</b>", body_style),
            Paragraph(f"<b>{progression_status}</b>", body_style),
        ],
    ]

    meta_table = Table(meta_data, colWidths=[120, 150, 120, 150])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDF2F7")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Annotated Image Visual Section
    if temp_img_path.exists():
        story.append(Paragraph("Visual Defect Localization & Bounding Boxes", heading_style))
        img = RLImage(str(temp_img_path), width=480, height=270)
        story.append(img)
        story.append(Spacer(1, 12))

    # Detected Defects Table Section
    story.append(Paragraph("Detected Track Defects Breakdown", heading_style))

    if defects:
        table_data = [
            [
                Paragraph("<b>#</b>", body_style),
                Paragraph("<b>Defect Class</b>", body_style),
                Paragraph("<b>Severity</b>", body_style),
                Paragraph("<b>Length (cm)</b>", body_style),
                Paragraph("<b>Width (cm)</b>", body_style),
                Paragraph("<b>Confidence</b>", body_style),
            ]
        ]

        for i, d in enumerate(defects, start=1):
            cls_name = d.get("class_name", d.get("defect_type", "crack"))
            sev = d.get("severity", "Low")
            len_cm = d.get("length_cm", 0.0)
            wid_cm = d.get("width_cm", 0.0)
            conf = d.get("confidence", 1.0)
            conf_str = f"{conf * 100:.1f}%" if conf <= 1.0 else f"{conf:.1f}%"

            table_data.append(
                [
                    Paragraph(str(i), body_style),
                    Paragraph(cls_name.capitalize(), body_style),
                    Paragraph(f"<b>{sev}</b>", body_style),
                    Paragraph(f"{len_cm:.2f}", body_style),
                    Paragraph(f"{wid_cm:.2f}", body_style),
                    Paragraph(conf_str, body_style),
                ]
            )

        defects_table = Table(table_data, colWidths=[30, 110, 90, 100, 100, 110])
        defects_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(defects_table)
    else:
        story.append(
            Paragraph("<i>No surface defects or structural cracks detected on this track segment.</i>", body_style)
        )

    story.append(Spacer(1, 14))

    # Maintenance Action Directive Box
    story.append(Paragraph("Maintenance & Operational Recommendation", heading_style))
    rec_box_data = [[Paragraph(f"<b>ACTION REQUIRED:</b> {recommendation}", body_style)]]
    rec_box = Table(rec_box_data, colWidths=[540])

    # Color code box based on health score
    bg_color = colors.HexColor("#FEFCBF") if health_score < 80.0 else colors.HexColor("#C6F6D5")
    if health_score < 60.0:
        bg_color = colors.HexColor("#FED7D7")

    rec_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg_color),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(rec_box)

    # Footer Notes
    story.append(Spacer(1, 20))
    footer_text = (
        "Report generated automatically by RailGuard Vision AI System. "
        "Physical measurements calibrated against standard rail gauge reference (143.5 cm)."
    )
    story.append(Paragraph(f"<i>{footer_text}</i>", ParagraphStyle("Footer", parent=body_style, fontSize=8, textColor=colors.HexColor("#718096"))))

    # Build PDF Document
    doc.build(story)

    # Clean up temporary saved image file
    if temp_img_path.exists():
        try:
            temp_img_path.unlink()
        except Exception:
            pass

    return output_path


if __name__ == "__main__":
    from src.preprocessing import create_synthetic_rail_sample
    from src.detect import YOLODetector

    print("[Report Generator] Running standalone PDF generation test...")
    sample_img = create_synthetic_rail_sample()
    detector = YOLODetector()
    dets = detector.detect(sample_img)
    annotated = detector.draw_annotations(sample_img, dets)

    sample_inspection = {
        "track_segment_id": "SEG_NORTH_104",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "health_score": 75.0,
        "progression_status": "Growing (+26.2%)",
        "defects": [
            {
                "class_name": "crack",
                "severity": "Moderate",
                "length_cm": 4.5,
                "width_cm": 0.35,
                "confidence": 0.895,
            }
        ],
        "annotated_image": annotated,
    }

    pdf_path = generate_report(sample_inspection)
    print(f"[Report Generator] PDF successfully created: {pdf_path}")
