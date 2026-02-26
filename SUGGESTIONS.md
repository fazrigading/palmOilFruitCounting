# Project Suggestions & Improvements

This document outlines suggested improvements for the Palm Oil Fruit Counting project, organized by category and priority.

---

## 1. Code Quality & Best Practices

### 1.1 Type Hints

All Python files lack type annotations. Adding them improves IDE support and code maintainability.

**Example:**

```python
# Before
def filter_fruitlet_masks(masks, image):
    ...

# After
def filter_fruitlet_masks(masks: list[dict], image: np.ndarray) -> list[dict]:
    ...
```

### 1.2 Docstrings

Functions like `filter_fruitlet_masks()` in `sam_annotator.py` have docstrings, but many others don't (e.g., `save_yolo_bbox`, `process_images`).

### 1.3 Bug Fix: Parameter Order in `sam_annotator.py:217`

```python
# Current (buggy):
process_images(args.input, args.output, args.config, args.model_type, args.checkpoint, args.device)

# Function signature: 
# process_images(image_dir, output_dir, config=None, model_type="tiny", checkpoint=None, device="cuda")
```

The order of `config` and `model_type` is swapped. This needs to be fixed.

### 1.4 Logging

Add logging instead of `print()` statements throughout the codebase for better debugging.

**Example:**

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Processing {len(images)} images...")
```

### 1.5 Error Handling

`sam_annotator.py` lacks try-except blocks for GPU operations and file I/O.

---

## 2. Features & Functionality

### 2.1 High Priority

#### Dataset Splitting

Create a `split_dataset.py` script to split into train/val/test sets for YOLO training.

```python
# Example structure
def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    ...
```

#### Label Validation

Script to validate YOLO format labels:

- Check for out-of-bounds coordinates (0-1)
- Detect empty label files
- Verify image-label pairing

#### Actual Augmentations

`augment.py` only resizes. The plan.md mentions:

- Random brightness/contrast
- Hue/saturation changes
- Mosaic augmentation

These should be implemented.

#### Multi-Class Support

Currently only class `0` (Fruit) exists. Consider adding:

- `0`: Unripe fruit (green/black)
- `1`: Ripe fruit (orange/red)
- `2`: Overripe fruit (dark maroon)

### 2.2 Medium Priority

#### Annotation Statistics

Script to analyze dataset:

- Class distribution
- Object size distribution
- Image count per split
- Average objects per image

#### Inference Script

Phase 3 in plan.md mentions this. Create:

- `predict.py` - Run model on single image
- `count_fruits.py` - Batch counting with output CSV

#### Training Config

Create YAML config for YOLO training:

```yaml
# configs/train.yaml
model: yolov8n-seg.pt
epochs: 100
batch_size: 16
imgsz: 640
augment:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  mosaic: 1.0
```

#### Video Support

Add video frame extraction for field video data.

---

## 3. Project Structure

### 3.1 Recommended Package Structure

```tree
palmOilFruitCounting/
├── src/
│   └── palm_oil_counting/
│       ├── __init__.py
│       ├── annotation/        # sam_annotator.py, dataAnnotation.py
│       │   ├── __init__.py
│       │   ├── sam_annotator.py
│       │   └── hsv_annotator.py
│       ├── gui/               # cropper.py, annotator.py
│       │   ├── __init__.py
│       │   ├── cropper.py
│       │   └── annotator.py
│       ├── preprocessing/     # augment.py
│       │   ├── __init__.py
│       │   └── augment.py
│       └── utils/             # Shared utilities
│           ├── __init__.py
│           ├── yolo_format.py
│           └── visualization.py
├── configs/                   # YAML configs
│   ├── annotator_config.yaml  # Annotator GUI session settings
│   ├── filter_config.yaml     # Annotation filter criteria
│   └── train.yaml             # YOLO training configuration
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_filter.py
│   └── test_yolo_format.py
├── scripts/                   # CLI entry points
│   ├── annotate.py
│   ├── crop.py
│   └── train.py
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── requirements.txt
```

### 3.2 Add Tests Directory

No tests exist. Add pytest tests for:

- Filter logic in `filter_fruitlet_masks()`
- YOLO format conversion functions
- Annotation loading/saving

### 3.3 Modern Python Packaging

Add `pyproject.toml` instead of just `requirements.txt`.

---

## 4. Documentation

### 4.1 Update README.md

Currently missing:

- SAM annotator usage instructions
- Project structure explanation
- Data pipeline diagram
- Installation with pyproject.toml

### 4.2 API Documentation

Document the SAM filtering parameters:

- `area < 50` - minimum mask area
- `aspect_ratio 0.3-3.3` - shape constraint
- `circularity < 0.4` - roundness filter
- Color thresholds for fruit detection

### 4.3 CONTRIBUTING.md

Add guidelines for:

- Code style (PEP 8, type hints)
- Pull request process
- Testing requirements

---

## 5. GUI Tools Improvements

### 5.1 Keyboard Shortcuts Display

Add a shortcuts menu to `annotator.py` (similar to `cropper.py`'s shortcut menu).

### 5.2 Zoom/Pan

Add zoom and pan functionality to both GUI tools for detailed inspection:

- Mouse wheel for zoom
- Middle-click drag for pan

### 5.3 Batch Operations

Add batch operations to annotator:

- Select multiple annotations (Ctrl+click)
- Delete all visible annotations
- Batch filter by criteria

### 5.4 Annotation Drawing

Currently `annotator.py` only reviews annotations. Add ability to:

- Draw new polygon annotations
- Edit existing polygon vertices
- Merge/split annotations

---

## 6. Next Steps (from plan.md)

The plan is solid. Immediate priorities:

1. **Complete dataset annotation** - Process full dataset through SAM pipeline
2. **Create train/val split script** - Prepare for YOLO training
3. **Set up YOLO training pipeline** - YOLOv8-seg or YOLO11-seg
4. **Implement counting metrics** - MAE/RMSE evaluation

---

## 7. Additional Recommendations

### 7.1 CI/CD

Add GitHub Actions for:

- Automated testing on PR
- Linting (ruff/black)
- Type checking (mypy)

### 7.2 Pre-commit Hooks

Add `.pre-commit-config.yaml` for:

- Black formatting
- Ruff linting
- Type checking

### 7.3 Requirements Versioning

Pin package versions in `requirements.txt`:

```text
torch==2.1.0
opencv-python-headless==4.8.1.78
...
```

### 7.4 Configuration Management

Move JSON configs to YAML for better readability and comments support.

---

## Priority Summary

| Priority | Task |
| -------- | ---- |
| P0 | Fix parameter order bug in sam_annotator.py |
| P0 | Add train/val split script |
| P1 | Add type hints |
| P1 | Add docstrings |
| P1 | Implement actual augmentations |
| P1 | Add inference script |
| P2 | Add tests |
| P2 | Add logging |
| P2 | Add zoom/pan to GUI |
| P3 | Add CI/CD |
| P3 | Add pre-commit hooks |
