# Plan: Fruit Counting on Palm Oil Bunches (FFB)

## 1. Project Overview
**Objective:** Accurate counting of individual fruits on Fresh Fruit Bunches (FFBs) using images/videos collected via mobile phones.

**Goal:** Develop a computer vision model to detect and count individual fruits to assist in yield estimation and bunch weight prediction.

## 2. Data Acquisition & Preparation
Since data is collected via mobile phones, variability in quality, lighting, and angles is expected.

### 2.1 Data Collection Guidelines
- **Angles:** Capture images from multiple sides of the bunch (360-degree view if possible) to handle occlusion of individual fruits.
- **Lighting:** Collect data at different times of day (morning, noon, evening) to account for shadows and backlight within the bunch structure.
- **Conditions:** Include various weather conditions (sunny, cloudy).
- **Distances:** Capture from varying close-up distances (0.5m - 1.5m) to ensure fruits are clearly visible.

### 2.2 Data Annotation Pipeline
We have developed a robust hybrid pipeline for annotating fruitlets:
1. **Cropping:** Using `gui_tools/cropper.py` to batch crop images to focal areas.
2. **Auto-Annotation (SAM2):** Using `sam_annotator.py` to automatically generate highly accurate polygon masks. This script includes custom filtering logic (color and Shape metrics) to isolate only the target fruitlets (red-orangeish and black-maroonish) and discard background elements (sky, leaves).
3. **Manual Review/Correction:** Using `gui_tools/annotator.py` to load images and their generated YOLO `.txt` labels, filtering out any remaining noisy annotations and adjusting boundaries if necessary.

**Classes:**
- `0: Fruit` (Individual palm oil fruit).

### 2.3 Preprocessing & Augmentation
- **Resizing:** Resize to model input standard (e.g., 640x640).
- **Augmentation:**
    - Random brightness/contrast (crucial for outdoor settings).
    - Hue/Saturation changes (to handle different camera color profiles and ripeness variations).
    - Mosaic augmentation (popular with YOLO to detect small objects like fruits).

## 3. Model Selection
Prioritize models that balance accuracy with inference speed, considering potential mobile deployment for real-time counting.

### 3.1 Recommended Architectures
- **YOLOv8 / YOLO11 (Ultralytics):** State-of-the-art for real-time detection and instance segmentation. Good balance of speed and accuracy for small object detection. Since we have high-quality SAM segmentations, training a YOLO instance segmentation model is highly viable.
- **EfficientDet-Lite / MobileNetV2 + SSD:** Lightweight alternatives for legacy edge devices.

### 3.2 Counting Logic
- **Simple Detection:** Count number of bounding boxes or masks per image.
- **Tracking (Video):** If input is video, use MOT (Multi-Object Tracking) like ByteTrack/StrongSORT to count unique fruits as the camera moves.

## 4. Development Workflow

### Phase 1: Setup & Data (In Progress)
1.  [x] Initialize repository and tools (`main_tool.py`, cropper, annotator).
2.  [x] Implement automatic annotation using SAM2 with targeted fruitlet filtering.
3.  [ ] Process full dataset through the SAM pipeline and review with the GUI annotator.

### Phase 2: Training (Upcoming)
1.  **Baseline:** Train a nano/small version of YOLO (e.g., `yolov8n-seg`) on the segmented dataset.
2.  **Tuning:** Experiment with hyperparameters and augmentation intensity.
3.  **Validation:** Monitor mAP@0.5 and mAP@0.5:0.95.

### Phase 3: Counting Logic Implementation & Deployment
1.  Develop an inference script to take an image of a bunch, run the trained YOLO model, and print the fruit count.
2.  Evaluate and integrate video tracking (ByteTrack) if moving to video-based counting.
3.  Export model to ONNX or TFLite for mobile/edge use cases.

## 5. Evaluation Metrics
- **Detection Performance:** Precision, Recall, mAP (Mean Average Precision) for both bounding boxes and masks.
- **Counting Performance:**
    - **MAE (Mean Absolute Error):** Average difference between predicted fruit count and ground truth.
    - **RMSE (Root Mean Square Error):** Penalizes larger errors in fruit counting more heavily.

## 6. Deployment Strategy
- **Format:** Export model to ONNX and TFLite.
- **Interface:**
    - *Option A (Edge):* Android/iOS app using TFLite interpreter. Runs offline for field workers.
    - *Option B (Cloud):* Fastapi/Flask backend receiving images from phone, returning count.

## 7. Timeline (Estimate)
- **Week 1:** Data collection and initial annotation of individual fruits.
- **Week 2:** Model selection, environment setup, and baseline training.
- **Week 3:** Hyperparameter tuning and implementing counting/tracking logic.
- **Week 4:** Evaluation, testing on field data, and mobile export.