
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# Load the model once with caching
@st.cache_resource
def load_model():
    model = YOLO('https://hub.ultralytics.com/models/9ExafekBT5o47OLbCIgq')  # Replace with your model if needed
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
 
