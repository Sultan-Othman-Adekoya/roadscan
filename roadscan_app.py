import streamlit as st
from PIL import Image, ExifTags
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Road Defect Detection", layout="centered")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("model_- 17 may 2025 7_46.pt") 

model = load_model()

# Extract GPS coordinates from EXIF data
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

st.title("🚧 Road Defect Detection System")

option = st.radio("Select input method", ["Upload Image", "Use Webcam"])
img = None

def run_webcam():
    stframe = st.empty()
    camera = cv2.VideoCapture(0)
    take_photo = st.button("Capture Photo")
    captured_image = None

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to read from webcam.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")
        if take_photo:
            captured_image = frame
            break

    camera.release()
    stframe.empty()
    return captured_image

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image of a road", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        img = Image.open(uploaded_file)
elif option == "Use Webcam":
    st.write("Click the button to take a snapshot from your webcam.")
    captured_img = run_webcam()
    if captured_img is not None:
        img = Image.fromarray(captured_img)

if img is not None:
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting road defects..."):
        img_array = np.array(img)
        results = model(img_array)
        result_img = results[0].plot()

        st.image(result_img, caption="Detection Result", use_column_width=True)

        # Display GPS data if available
        location = get_exif_location(img)
        if location:
            lat, lon = location
            map_url = gps_to_google_maps_url(lat, lon)
            st.markdown(f"**Detected Location:** [{lat:.6f}, {lon:.6f}]({map_url})")
            st.map(data={"lat": [lat], "lon": [lon]})
        else:
            st.warning("No GPS location data found in image metadata.")
