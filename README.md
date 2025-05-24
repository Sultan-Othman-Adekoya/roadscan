# RoadScan 

A computer vision-powered web application that detects road defects (e.g., potholes, cracks) from images or webcam input. The system uses a YOLOv8 model to identify road issues and log detections with optional GPS data from image EXIF metadata. Built using Streamlit, OpenCV, and PyTorch.


# Features

-  Upload or capture images using webcam
-  Detect road defects using a YOLOv8 model
-  Extract GPS coordinates from image EXIF metadata
-  Generate clickable Google Maps links for GPS-tagged images
-  View detection logs with filters for date and defect type
-  Generate downloadable PDF reports from logs
-  Lightweight, runs on local machine or cloud


# Requirements
- Python 3.10+
- pip


# Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/Sultan-Othman-Adekoya/roadscan
   cd roadscan
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run roadscan_app.py

# Folder Structure

roadscan/
├── models/
│   └── roadscan_model.pt
├── images/
├── outputs/
├── roadscan_app.py
├── requirements.txt
├── runtime.txt
├── .streamlit/
│   └── config.toml
└── README.md
