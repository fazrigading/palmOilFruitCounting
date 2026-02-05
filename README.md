# Palm Oil Fruit Counting & Analysis

A computer vision project designed for the accurate detection, counting, and segmentation of individual fruits on Fresh Fruit Bunches (FFB) of Palm Oil. This repository contains data annotation tools and processing scripts to assist in yield estimation.

## 🚀 Featured Tool: Gading's Batch Image Cropper

Included in this repository is **Gading's Batch Image Cropper**, a specialized GUI tool built with Tkinter for preparing image datasets.

### Key Features
- **Batch Processing:** Easily navigate through large folders of images.
- **Aspect Ratio Locking:** Choose from standard ratios (1:1, 16:9, etc.) or use "Free" mode.
- **Smart Resizing:** Automatically resize cropped images to target dimensions (default 640x640).
- **Progress Tracking:** High-visibility "Cropped ✅" indicators show which images are already processed.
- **Cross-Platform:** Runs on Windows, macOS, and Linux.

### Usage (Developer)
1. Install dependencies:
   ```bash
   pip install Pillow
   ```
2. Run the application:
   ```bash
   python3 cropper.py
   ```

---

## 📊 Research Data Annotation
The project also includes `dataAnnotation.py` for automatic fruit segmentation using color-based HSV thresholding.

### Automatic Labeling
- **HSV Segmentation:** Targets ripe/orange palm oil fruits.
- **YOLO Format:** Generates standard YOLOv8/v11 segmentation labels (.txt).
- **Visual Verification:** Saves processed masks in `dataset/visualized` for quality control.

---

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8+
- OpenCV
- NumPy
- Pillow (PIL)

```bash
git clone https://github.com/yourusername/palmOilFruitCounting.git
cd palmOilFruitCounting
pip install opencv-python numpy Pillow
```

## 📦 Distribution
To build a standalone executable of the Cropper tool:
```bash
pip install pyinstaller
pyinstaller --onefile --noconsole --name "gadings-batch-image-cropper" --version-file=version_info.txt cropper.py
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author
**Fazri Gading** (FGDX)  
Contact: [fazrigading@gmail.com](mailto:fazrigading@gmail.com)
