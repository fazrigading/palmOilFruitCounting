# Gemini Context: Palm Oil Fruit Counting (FFB)

## Project Overview
**Objective:** Accurately count individual fruits on Fresh Fruit Bunches (FFBs) of palm oil using computer vision (images/videos).
**Goal:** Assist in yield estimation and bunch weight prediction.
**Author:** Fazri Gading (FGDX)

## Current Architecture & Tools

### 1. Data Preparation Tools
*   **Main Launcher (`main_tool.py`):**
    *   Unified GUI launcher bringing together the Cropper and Annotator tools.
    *   **Usage:** `python3 main_tool.py`
*   **Gading's Batch Image Cropper (`gui_tools/cropper.py`):**
    *   A custom Tkinter GUI for efficient image cropping.
    *   **Key Features:** Batch processing, aspect ratio cycling (scroll wheel), "Hide Cropped" filter, saves at original resolution.
*   **Image Annotator (`gui_tools/annotator.py`):**
    *   A custom Tkinter GUI for reviewing, modifying, and manually filtering polygon annotations.
*   **SAM Annotator (`sam_annotator.py`):**
    *   Uses Segment Anything Model 2 (SAM2) for robust automatic zero-shot annotation.
    *   **Key Features:** Incorporates an automatic filtering step (based on size, aspect ratio, circularity, and RGB color) to specifically isolate black-maroonish and red-orangeish palm oil fruitlets, rejecting background elements like sky and leaves.
*   **Data Annotation (`dataAnnotation.py`):**
    *   Alternative/Legacy script automating fruit segmentation using HSV color thresholding.

### 2. Model Strategy
*   **Target Architecture:** YOLOv8 or YOLO11 (Ultralytics) for real-time detection of small objects (fruits) and instance segmentation.
*   **Alternatives:** EfficientDet-Lite, MobileNetV2+SSD (for legacy edge devices).
*   **Counting Logic:**
    *   **Image:** Count bounding boxes or instance masks.
    *   **Video:** Potential use of MOT (ByteTrack/StrongSORT) to track and count fruits across frames (preventing double counting).

### 3. Data Pipeline
*   **Input:** Images/Videos collected via mobile phones (varying angles, lighting, distances).
*   **Preprocessing:** Cropping (via GUI tool), Resize (e.g., 640x640), Augmentation (Brightness, Hue, Mosaic, Rotation).
*   **Output:** Fruit count per bunch.

## Roadmap & Status

### Phase 1: Setup & Data (Current Focus)
*   [x] Repository initialization.
*   [x] Basic tooling (`main_tool.py`, `cropper.py`, `annotator.py`).
*   [x] Automatic Annotation Pipeline (SAM2 with fruitlet filtering).
*   [ ] Complete dataset collection (angles: 360, lighting: various).
*   [ ] Final manual review of SAM annotations.

### Phase 2: Training
*   [ ] Baseline training with YOLOv8n / YOLO11n (Detection/Segmentation).
*   [ ] Hyperparameter tuning & Augmentation experiments.
*   [ ] Validation (mAP metrics).

### Phase 3: Implementation
*   [ ] Image-to-count script.
*   [ ] Video tracking pipeline (optional/advanced).
*   [ ] Mobile export (TFLite/ONNX).

## Technical Context
*   **Language:** Python 3.8+
*   **Key Libraries:** `opencv-python`, `numpy`, `Pillow`, `ultralytics`, `torch`, `sam2`.
*   **Directory Structure:**
    *   `dataset/`: Contains images, augmented datasets, and SAM annotations (`labels_bbox`, `labels_seg`).
    *   `gui_tools/`: Contains GUI modules for cropping and annotating.
    *   `dist/` & `build/`: Build artifacts for the bundled tools.
