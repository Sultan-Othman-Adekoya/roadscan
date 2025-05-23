
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# Load the model once with caching
@st.cache_resource
def load_model():
    model = YOLO('model_- 17 may 2025 7_46.pt')  # Replace with your model if needed
    return model

st.set_page_config(page_title="RoadScan AI", layout="centered")
st.title("🛣️ RoadScan AI - Road Defect Detection App")

uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running detection..."):
        model = load_model()
        results = model.predict(image, conf=0.5)

    # Display annotated image
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        annotated_frame = results[0].plot()
        st.image(annotated_frame, caption="Detected Defects", use_column_width=True)
        st.success("Detection complete ✅")
    else:
        st.image(image, caption="No Defects Detected ✅", use_column_width=True)
        st.info("No road defects were detected.")  
        
def get_exif_location(img_pil):
    """Extract GPS coordinates from image EXIF, return (lat, lon) or None."""
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
 
