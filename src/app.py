"""
RailGuard Vision - Streamlit Web Dashboard
===========================================
Interactive real-time computer vision dashboard for railway track defect detection,
crack measurement, severity grading, track health scoring, time-series progression tracking,
and automated PDF inspection report generation.
"""

import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import sys

# Ensure root src directory is in Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_MODEL_PATH, DB_PATH, DEFAULT_RAIL_GAUGE_CM, REPORTS_DIR
from src.preprocessing import enhance_contrast, denoise_image, preprocess_image
from src.detect import YOLODetector
from src.crack_measurement import measure_crack
from src.severity import classify_severity, calculate_health_score
from src.progression import log_inspection, compare_with_previous
from src.report_generator import generate_report, get_maintenance_recommendation


# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="RailGuard Vision — Track Defect AI",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for sleek dark tech aesthetic
st.markdown(
    """
    <style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stMetric {
        background-color: #1E293B;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    .stAlert {
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #38BDF8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector():
    """Caches YOLODetector instance to prevent re-loading weights on UI rerenders."""
    return YOLODetector(model_path=DEFAULT_MODEL_PATH)


def main():
    st.title("🚆 RailGuard Vision")
    st.caption("YOLOv8-based Autonomous Railway Track Defect Detection & Infrastructure Health System")
    st.markdown("---")

    # --- Sidebar Controls ---
    st.sidebar.header("⚙️ System Settings & Controls")
    track_segment_id = st.sidebar.text_input("Track Segment Identifier", value="SEG_NORTH_104")
    conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.10, 0.95, 0.40, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📏 Physical Calibration Reference")
    ref_gauge_cm = st.sidebar.number_input(
        "Standard Rail Gauge Width (cm)", min_value=50.0, max_value=250.0, value=DEFAULT_RAIL_GAUGE_CM
    )
    ref_gauge_px = st.sidebar.number_input("Measured Rail Gauge (pixels)", min_value=50.0, max_value=1000.0, value=240.0)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**RailGuard Vision Capstone Project**\n"
        "• Deep Learning: YOLOv8\n"
        "• Image Processing: OpenCV & CLAHE\n"
        "• Progression: SQLite Log\n"
        "• Reporting: ReportLab PDF"
    )

    detector = load_detector()

    # --- Main Application Tabs ---
    tab1, tab2, tab3 = st.tabs(["📷 Single Image Inspection", "🎥 Video Stream File", "📹 Live Webcam Feed"])

    # ==========================================
    # TAB 1: SINGLE IMAGE INSPECTION
    # ==========================================
    with tab1:
        st.subheader("Single Image Defect Detection & Measurement")
        uploaded_file = st.file_uploader("Choose a railway track image...", type=["jpg", "jpeg", "png", "bmp"])

        if uploaded_file is not None:
            # Read uploaded image file
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            col1, col2 = st.columns(2)

            with col1:
                st.image(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), caption="Uploaded Raw Image", use_column_width=True)

            # Apply Preprocessing Pipeline
            with st.spinner("Applying CLAHE contrast enhancement & YOLO detection..."):
                enhanced = enhance_contrast(raw_image)
                denoised = denoise_image(enhanced)

                # Run YOLO Detection
                detections = detector.detect(denoised, conf_threshold=conf_threshold)

                # Measure Defect Dimensions and Classify Severity
                processed_defects = []
                annotated_img = raw_image.copy()

                for det in detections:
                    bbox = det["bbox"]
                    meas = measure_crack(raw_image, bbox, ref_width_px=ref_gauge_px, ref_width_cm=ref_gauge_cm)
                    sev = classify_severity(meas["length_cm"], meas["width_cm"])

                    defect_info = {
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "bbox": bbox,
                        "length_cm": meas["length_cm"],
                        "width_cm": meas["width_cm"],
                        "severity": sev,
                    }
                    processed_defects.append(defect_info)

                # Draw Visual Annotations
                annotated_img = detector.draw_annotations(raw_image, detections)

                # Compute Track Health Score
                health_score = calculate_health_score(processed_defects)

                # Log to SQLite DB and Compare Progression
                timestamp_str = datetime.now(timezone.utc).isoformat()
                primary_defect = processed_defects[0] if processed_defects else {"class_name": "none", "severity": "Low", "length_cm": 0.0, "width_cm": 0.0}

                log_id = log_inspection(
                    track_segment_id=track_segment_id,
                    defect_type=primary_defect["class_name"],
                    severity=primary_defect["severity"],
                    length_cm=primary_defect["length_cm"],
                    width_cm=primary_defect["width_cm"],
                    health_score=health_score,
                    timestamp=timestamp_str,
                    db_path=DB_PATH,
                )

                progression = compare_with_previous(track_segment_id, db_path=DB_PATH)

            with col2:
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="AI Annotated Detections", use_column_width=True)

            st.markdown("### 📊 Inspection Analysis & Summary")
            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Track Health Score", f"{health_score} / 100")
            m2.metric("Defects Detected", len(processed_defects))
            m3.metric("Progression Status", progression["status"], delta=f"{progression['percent_growth']}% growth")
            m4.metric("Logged Record ID", f"#{log_id}")

            rec_text = get_maintenance_recommendation(health_score)
            if health_score >= 80:
                st.success(f"**Maintenance Directive:** {rec_text}")
            elif health_score >= 60:
                st.warning(f"**Maintenance Directive:** {rec_text}")
            else:
                st.error(f"**Maintenance Directive:** {rec_text}")

            if processed_defects:
                st.subheader("Detailed Defect Measurement Breakdown")
                st.dataframe(
                    [
                        {
                            "Defect Type": d["class_name"].capitalize(),
                            "Severity": d["severity"],
                            "Length (cm)": d["length_cm"],
                            "Width (cm)": d["width_cm"],
                            "Confidence": f"{d['confidence']*100:.1f}%",
                            "Bounding Box [x1,y1,x2,y2]": str(d["bbox"]),
                        }
                        for d in processed_defects
                    ],
                    use_container_width=True,
                )

            # --- PDF Report Download Button ---
            st.markdown("### 📄 Export Inspection Report")
            inspection_payload = {
                "track_segment_id": track_segment_id,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "health_score": health_score,
                "progression_status": f"{progression['status']} ({progression['percent_growth']:+.1f}%)",
                "defects": processed_defects,
                "annotated_image": annotated_img,
            }

            pdf_out_path = generate_report(inspection_payload)
            with open(pdf_out_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📥 Download PDF Inspection Report",
                data=pdf_bytes,
                file_name=f"RailGuard_Report_{track_segment_id}.pdf",
                mime="application/pdf",
            )

    # ==========================================
    # TAB 2: VIDEO FILE STREAMING
    # ==========================================
    with tab2:
        st.subheader("Real-Time Video Defect Inspection")
        video_file = st.file_uploader("Upload track video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
        frame_skip = st.slider("Frame Process Interval (Process 1 frame every N frames)", 1, 10, 3)

        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)

            st_frame = st.empty()
            fps_metric = st.empty()

            frame_count = 0
            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip == 0:
                    dets = detector.detect(frame, conf_threshold=conf_threshold)
                    annotated_frame = detector.draw_annotations(frame, dets)

                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0.0

                    # Draw FPS on frame
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {fps:.1f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )

                    st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                    fps_metric.metric("Real-Time Processing FPS", f"{fps:.1f} FPS")

            cap.release()
            st.success("Video processing complete!")

    # ==========================================
    # TAB 3: LIVE WEBCAM FEED
    # ==========================================
    with tab3:
        st.subheader("Live Webcam Feed Inspection")
        run_webcam = st.checkbox("Start Live Camera Feed")

        if run_webcam:
            cap = cv2.VideoCapture(0)
            st_cam = st.empty()
            st_fps = st.empty()

            frame_cnt = 0
            t0 = time.time()

            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Unable to access live webcam feed.")
                    break

                frame_cnt += 1
                dets = detector.detect(frame, conf_threshold=conf_threshold)
                annotated = detector.draw_annotations(frame, dets)

                dt = time.time() - t0
                fps = frame_cnt / dt if dt > 0 else 0.0

                cv2.putText(
                    annotated,
                    f"LIVE FPS: {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

                st_cam.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                st_fps.metric("Live Feed Speed", f"{fps:.1f} FPS")

            cap.release()


if __name__ == "__main__":
    main()
