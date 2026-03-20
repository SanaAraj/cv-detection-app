# CV Detection App

Real-time object detection web application using YOLO11.

## Overview

Upload an image through the web interface and get back an annotated version with bounding boxes, class labels, and confidence scores. The app uses YOLO11n (nano) for fast inference without requiring a GPU.

## How It Works

1. Upload an image via the web interface
2. YOLO11n runs detection on the image
3. The app returns the annotated image with detected objects highlighted

## Setup

```bash
git clone https://github.com/SanaAraj/cv-detection-app.git
cd cv-detection-app
pip install -r requirements.txt
```

## Usage

Start the server:

```bash
uvicorn main:app --reload
```

Open http://localhost:8000 in your browser. Upload an image and click "Detect Objects" to see the results.

## Model

The app uses YOLO11n (nano), the smallest and fastest variant of YOLO11. The model is pretrained on COCO dataset and can detect 80 different object classes including people, vehicles, animals, and common household items.

Model weights are downloaded automatically on first run.

## Tech Stack

- YOLO11 (Ultralytics)
- FastAPI
- OpenCV
- Python 3.10+
