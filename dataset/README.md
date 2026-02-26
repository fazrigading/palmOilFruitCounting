# Dataset Preparation Guide

This guide explains how to prepare your palm oil fruit dataset for training YOLO models.

## Dataset Structure

Preferred files structure:

```tree
dataset/
├── images/             # Input images (640x640 with white padding)
└── labels/             # Converted labels (optional)
    ├── bbox/           # Bounding boxes in YOLO format
    └── mask/           # Segmentation masks in YOLO format
```

## Image Preparation

### 1. Collect Raw Images

Capture or gather images of Fresh Fruit Bunches (FFB) of palm oil. Ensure:

- Good lighting conditions
- Clear view of individual fruitlets
- Various ripeness stages (optional for diversity)

### 2. Resize with White Padding

YOLO requires consistent input size (640x640). To preserve original aspect ratio:

1. Use the batch image cropper tool:

   ```bash
   palm-crop
   ```

2. Or resize manually with white padding (letterbox):

   ```bash
   python scripts/crop.py --input raw_images/ --output dataset/images/ --size 640
   ```

This adds white padding to maintain aspect ratio while fitting 640x640.

## Generating Annotations with SAM2

### 1. Install SAM2

```bash
pip install -e ".[sam]"
```

### 2. Run SAM2 Annotation

```bash
python scripts/annotate.py sam \
    --input dataset/images/ \
    --output dataset/labels/mask/ \
    --model-type base_plus \
    --device cuda
```

Available model types: `tiny`, `small`, `base_plus`, `large`

### 3. Filter False Positives

SAM2 may detect unwanted objects from background and foreground. Use the filter configuration in `configs/filter_config.yaml`:

```yaml
filter:
  max_ratio: 1.2        # Elongation ratio (width/height)
  min_area: 0.04        # Minimum area (4% of image)
  max_area: 0.6         # Maximum area (60% of image)
  min_width: 1.2        # Minimum width (percent of image width)
  max_width: 5.0        # Maximum width
  min_height: 1.0       # Minimum height
  max_height: 12.0      # Maximum height
```

Edit `configs/filter_config.yaml` to adjust thresholds, then re-run filtering.

## Label Format

### Segmentation Format (`labels/mask/`)

YOLO segmentation format - one `.txt` file per image:

```yaml
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

- `class_id`: 0 for palm oil fruitlet
- Coordinates: Normalized to [0, 1] (x_center, y_center relative to image size)

### Detection Format (`labels/bbox/`)

YOLO bounding box format:

```yaml
class_id x_center y_center width height
```

All values normalized to [0, 1].

## Converting Between Formats

### Segmentation to Detection

```bash
# Convert masks to bounding boxes
python -c "
from palm_oil_counting.utils import save_yolo_bbox, save_yolo_segmentation
import os

# Load segmentation and convert to bbox
masks = 'dataset/labels/mask'
output = 'dataset/labels'

for f in os.listdir(masks):
    if f.endswith('.txt'):
        # Convert here
        pass
"
```

## Train/Val/Test Split

### Automatic Split

```bash
python scripts/dataset.py --input dataset/ --split 0.8 0.1 0.1
```

This creates:

```tree
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Manual Split

Alternatively, organize manually:

```bash
mkdir -p dataset/images/train dataset/images/val dataset/images/test
mkdir -p dataset/labels/train dataset/labels/val dataset/labels/test
```

## Training Configuration

Create `configs/train.yaml`:

```yaml
path: dataset
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: fruitlet
```

## Verification

### Visual Check

Use the annotation reviewer to verify labels:

```bash
palm-annotate
```

### Validate Labels

```bash
python -c "
from palm_oil_counting.utils import validate_yolo_label
import os

label_dir = 'dataset/labels/train'
for f in os.listdir(label_dir):
    is_valid, errors = validate_yolo_label(os.path.join(label_dir, f), check_bounds=True)
    if not is_valid:
        print(f'{f}: {errors}')
"
```

## Tips

1. **Quality over quantity**: Well-annotated 100 images > poorly annotated 1000
2. **Diverse samples**: Include different lighting, angles, and ripeness stages
3. **Edge cases**: Add images with overlapping fruits, shadows, or partial FFB
4. **Verify**: Always visually check a few annotations before training
5. **Filter aggressively**: Start with strict filters, relax if too many valid fruits are removed
6. **Remove redundant labels**: Remove unused masks and bboxes from the labels to help model understand the main object.

## Troubleshooting

**SAM2 not found**: Install with `pip install -e ".[sam]"`

**CUDA out of memory**: Use smaller model (`--model-type tiny`) or process in batches

**Too many false positives**: Increase `min_area` in filter config

**Valid fruits filtered out**: Decrease `min_area` or adjust other thresholds
