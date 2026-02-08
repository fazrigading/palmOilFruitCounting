# Gemini Context: Palm Oil Fruit Counting (FFB)

## Project Overview
**Objective:** Accurately count individual fruits on Fresh Fruit Bunches (FFBs) of palm oil using computer vision (images/videos).
**Goal:** Assist in yield estimation and bunch weight prediction.
**Author:** Fazri Gading (FGDX)

## Current Architecture & Tools

### 1. Data Preparation Tools
*   **Gading's Batch Image Cropper (`cropper.py`):**
    *   A custom Tkinter GUI for efficient image cropping.
    *   **Key Features:** Batch processing, aspect ratio cycling (scroll wheel), "Hide Cropped" filter, saves at original resolution.
    *   **Usage:** `python3 cropper.py`
*   **Data Annotation (`dataAnnotation.py`):**
    *   Automates fruit segmentation using HSV color thresholding (targets ripe/orange fruits).
    *   Generates YOLO format labels (`.txt`) for segmentation.
    *   Saves visualization masks to `dataset/visualized`.
*   **SAM Annotator (`sam_annotator.py`):**
    *   Mentions usage of Segment Anything Model (SAM) for robust annotation (as per `plan.md`).

### 2. Model Strategy
*   **Target Architecture:** YOLOv8 or YOLO11 (Ultralytics) for real-time detection of small objects (fruits).
*   **Alternatives:** EfficientDet-Lite, MobileNetV2+SSD (for legacy edge devices).
*   **Counting Logic:**
    *   **Image:** Count bounding boxes.
    *   **Video:** potential use of MOT (ByteTrack/StrongSORT) to track and count fruits across frames (preventing double counting).

### 3. Data Pipeline
*   **Input:** Images/Videos collected via mobile phones (varying angles, lighting, distances).
*   **Preprocessing:** Resize (e.g., 640x640), Augmentation (Brightness, Hue, Mosaic, Rotation).
*   **Output:** Fruit count per bunch.

## Roadmap & Status

### Phase 1: Setup & Data (Current Focus)
*   [x] Repository initialization.
*   [x] Basic tooling (`cropper.py`, `dataAnnotation.py`).
*   [ ] Complete dataset collection (angles: 360, lighting: various).
*   [ ] Finalize annotation (SAM/Manual/HSV).

### Phase 2: Training
*   [ ] Baseline training with YOLOv8n.
*   [ ] Hyperparameter tuning & Augmentation experiments.
*   [ ] Validation (mAP metrics).

### Phase 3: Implementation
*   [ ] Image-to-count script.
*   [ ] Video tracking pipeline (optional/advanced).
*   [ ] Mobile export (TFLite/ONNX).

## Technical Context
*   **Language:** Python 3.8+
*   **Key Libraries:** `opencv-python`, `numpy`, `Pillow`, `ultralytics` (implied for YOLO).
*   **Build:** PyInstaller used for packaging the cropper tool.
*   **Directory Structure:**
    *   `dataset/`: Contains cropped images, labels, and failed crops.
    *   `dist/` & `build/`: Build artifacts for the cropper tool.

## Commands
*   **Run Cropper:** `python3 cropper.py`
*   **Build Cropper:** `pyinstaller --onefile --noconsole --name "gadings-batch-image-cropper" --version-file=version_info.txt cropper.py`
