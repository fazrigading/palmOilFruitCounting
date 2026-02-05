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
- **Metadata:** Record bunch ID and timestamp for each capture to prevent double-counting in large datasets.

### 2.2 Data Annotation
- **Tools:** Use tools like CVAT, LabelImg, or Roboflow.
- **Labeling Method:**
    - *Bounding Boxes:* Standard for object detection (YOLO, etc.) of individual fruits.
    - *Instance Segmentation:* (Optional) Useful if fruits are highly clustered and need precise separation.
- **Classes:**
    - `Fruit` (Individual palm oil fruit).
    - `Occluded_Fruit` (Optional: to train model to recognize partially hidden fruits within the bunch).

### 2.3 Preprocessing & Augmentation
- **Resizing:** Resize to model input standard (e.g., 640x640).
- **Augmentation:**
    - Random brightness/contrast (crucial for outdoor settings).
    - Hue/Saturation changes (to handle different camera color profiles and ripeness variations).
    - Mosaic augmentation (popular with YOLO to detect small objects like fruits).
    - Random rotation/flip.

## 3. Model Selection
Prioritize models that balance accuracy with inference speed, considering potential mobile deployment for real-time counting.

### 3.1 Recommended Architectures
- **YOLOv8 / YOLO11 (Ultralytics):** State-of-the-art for real-time detection. Good balance of speed and accuracy for small object detection (fruits).
- **EfficientDet-Lite:** Optimized for mobile and edge devices.
- **MobileNetV2 + SSD:** Lightweight, older but very fast on legacy devices.

### 3.2 Counting Logic
- **Simple Detection:** Count number of bounding boxes per image.
- **Tracking (Video):** If input is video, use MOT (Multi-Object Tracking) like ByteTrack or StrongSORT to count unique fruits as the camera moves around the bunch, preventing double counting of the same fruit.

## 4. Development Workflow

### Phase 1: Setup & Data
1.  Initialize repository.
2.  Set up Python environment (PyTorch/TensorFlow).
3.  Organize dataset structure (Train/Val/Test split).

### Phase 2: Training
1.  **Baseline:** Train a nano/small version of YOLO (e.g., `yolov8n`) to establish a baseline for fruit detection.
2.  **Tuning:** Experiment with hyperparameters (learning rate, momentum) and augmentation intensity.
3.  **Validation:** Monitor mAP@0.5 and mAP@0.5:0.95.

### Phase 3: Counting Logic Implementation
1.  Develop a script to take an image of a bunch and return the fruit count.
2.  (Advanced) Implement a video tracking pipeline to count fruits on the *entire* bunch surface as the user pans around it.

## 5. Evaluation Metrics
- **Detection Performance:** Precision, Recall, mAP (Mean Average Precision).
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