
import streamlit as st
from PIL import Image, ExifTags
from ultralytics import YOLO
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from fpdf import FPDF

st.set_page_config(page_title="Road Defect Detection", layout="centered")

# --- Constants ---
LOG_FILE = "logs/detection_logs.csv"
os.makedirs("logs", exist_ok=True)

# --- Load model ---
@st.cache_resource
def load_model():
    return YOLO("best (1).torchscript")
    
model = load_model()

# --- EXIF GPS Extraction ---
def get_exif_location(img_pil):
    try:
        exif_data = img_pil._getexif()
        if not exif_data:
            return None
        gps_info = {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = ExifTags.GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
        if not gps_info:
            return None

        def convert_to_degrees(value):
            d = value[0][0] / value[0][1]
            m = value[1][0] / value[1][1]
            s = value[2][0] / value[2][1]
            return d + (m / 60.0) + (s / 3600.0)

        lat = convert_to_degrees(gps_info["GPSLatitude"])
        lon = convert_to_degrees(gps_info["GPSLongitude"])
        if gps_info.get("GPSLatitudeRef") != "N":
            lat = -lat
        if gps_info.get("GPSLongitudeRef") != "E":
            lon = -lon
        return lat, lon
    except Exception:
        return None

def gps_to_google_maps_url(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

# --- Logging functions ---
def log_results(results, location_url=None):
    logs = []
    for res in results:
        logs.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Road name": res["name"],
            "Confidence": round(res["confidence"], 2),
            "Location": location_url or ""
        })
    if logs:
        df = pd.DataFrame(logs)
        df.to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
    return logs

def load_all_logs():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["Timestamp", "Road name", "Confidence", "Location"])

def generate_pdf_report(logs_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Road Defect Scan Report", 0, 1, "C")
    pdf.set_font("Arial", size=12)
    for _, row in logs_df.iterrows():
        line = f"{row['Timestamp']} - {row['Road name']} - Confidence: {row['Confidence']}%"
        if row["Location"]:
            line += f" - Location: {row['Location']}"
        pdf.cell(0, 10, line, 0, 1)
    pdf_path = "logs/scan_result.pdf"
    pdf.output(pdf_path)
    return pdf_path

def make_clickable(val):
    if val:
        return f'[Map]({val})'
    return ""

# --- Main app ---

st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["Upload & Detect", "Detection Logs"])

if page == "Upload & Detect":
    st.title("🚧 Road Defect Detection System")
    st.markdown("---")

    input_mode = st.radio("Select Input Mode", ["Upload Image", "Use Webcam"])

    img_cv = None
    location_url = None
    results_to_log = None

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("🖼️ Upload Image (jpg, jpeg, png)", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file:
            img_pil = Image.open(uploaded_file)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            gps = get_exif_location(img_pil)
            if gps:
                location_url = gps_to_google_maps_url(*gps)
    else:
        capture = st.camera_input("📷 Capture Image from Webcam")
        if capture:
            img_pil = Image.open(capture)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            # No GPS data from webcam images
            location_url = None

    if img_cv is not None:
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Input Image", use_column_width=True)

        with st.spinner("⏳ Detecting road defects..."):
            detection_results = model(img_cv)

            # Parse detections into list of dicts for logging
            results_to_log = []
            for det in detection_results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = det
                name = model.names[int(cls)]
                confidence = float(conf * 100)
                results_to_log.append({"name": name, "confidence": confidence})

            # Show annotated image
            annotated_img = detection_results[0].plot()
            st.image(annotated_img, caption="Detection Output", use_column_width=True)

            # Log detection
            log_results(results_to_log, location_url)

            # Show detected location link if available
            if location_url:
                st.markdown(f"**Detected Location:** [{location_url}]({location_url})")
                lat, lon = gps
                st.map(data={"lat": [lat], "lon": [lon]})

elif page == "Detection Logs":
    st.title("🧾 Detection Logs")
    st.markdown("---")

    logs_df = load_all_logs()

    all_names = sorted(logs_df["Road name"].unique())
    filter_name = st.multiselect("Filter by Road name(s)", options=all_names, default=all_names)

    if not logs_df.empty:
        min_date = pd.to_datetime(logs_df["Timestamp"]).min()
        max_date = pd.to_datetime(logs_df["Timestamp"]).max()
    else:
        min_date = max_date = None

    date_range = st.date_input(
        "Filter by Date Range",
        value=(min_date, max_date) if min_date and max_date else None,
        min_value=min_date,
        max_value=max_date,
    )

    filtered_df = logs_df.copy()
    if filter_name:
        filtered_df = filtered_df[filtered_df["Road name"].isin(filter_name)]

    if date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (pd.to_datetime(filtered_df["Timestamp"]) >= start_date)
            & (pd.to_datetime(filtered_df["Timestamp"]) <= end_date + pd.Timedelta(days=1))
        ]

    filtered_df["Map Link"] = filtered_df["Location"].apply(make_clickable)

    st.dataframe(
        filtered_df.drop(columns=["Location"]),
        use_container_width=True,
    )

    st.markdown("### Location Links:")
    for idx, row in filtered_df.iterrows():
        if row["Location"]:
            st.markdown(f"- **{row['Road name']}** at [{row['Location']}]({row['Location']})")

    if st.button("📄 Generate PDF Report"):
        if filtered_df.empty:
            st.warning("No logs to generate report.")
        else:
            pdf_path = generate_pdf_report(filtered_df)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=f,
                    file_name="detection_report.pdf",
                    mime="application/pdf",
                )
