# Palm Oil Fruit Counting

A computer vision project for detecting, counting, and segmenting individual fruits on Fresh Fruit Bunches (FFB) of Palm Oil. Uses YOLO format for annotations and supports SAM2-based segmentation.

## Features

- **SAM2-based Segmentation**: Automatic fruit detection using Meta's Segment Anything Model 2
- **HSV Color-based Annotation**: Alternative annotation method targeting ripe/orange palm oil fruits
- **YOLO Format Support**: Generates standard YOLOv8/v11 segmentation and detection labels
- **Batch Image Cropper**: GUI tool for preparing image datasets
- **Annotation Reviewer**: GUI tool for reviewing and correcting annotations
- **Data Augmentation**: Image augmentation utilities for training data

## Installation

### Basic Installation

```bash
git clone https://github.com/fazrigading/palmOilFruitCounting.git
cd palmOilFruitCounting
pip install -e "."
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### With SAM2 Support

```bash
pip install -e ".[all]"
```

## Requirements

- Python >= 3.8
- torch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python-headless >= 4.8.0
- numpy >= 1.24.0
- albumentations >= 1.3.0
- Pillow >= 10.0.0
- tqdm >= 4.65.0
- matplotlib >= 3.7.0
- pyyaml >= 6.0

## Usage

### GUI Applications

```bash
# Run the batch image cropper
palm-crop

# Run the annotation reviewer
palm-annotate
```

### Command Line Scripts

```bash
# Automatic SAM2-based annotation
python scripts/annotate.py --input-dir images/ --output-dir labels/

# HSV color-based annotation
python scripts/review.py

# Dataset preparation and cropping
python scripts/crop.py

# Dataset splitting and augmentation
python scripts/dataset.py
```

## Project Structure

```tree
palmOilFruitCounting/
├── src/palm_oil_counting/
│   ├── annotation/           # Annotation tools
│   │   ├── sam_annotator.py  # SAM2-based annotation
│   │   └── hsv_annotator.py  # HSV color-based annotation
│   ├── gui/                  # GUI applications
│   │   ├── cropper.py        # Batch image cropper
│   │   └── annotator.py      # Annotation reviewer
│   ├── preprocessing/        # Data preprocessing
│   │   └── augment.py        # Image augmentation
│   └── utils/                # Utilities
│       ├── yolo_format.py    # YOLO format I/O
│       └── visualization.py  # Visualization tools
├── tests/                    # Test suite
├── scripts/                   # Command line scripts
├── configs/                   # Configuration files
└── pyproject.toml            # Project configuration
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_filter.py -v

# Run with coverage
pytest tests/ --cov=src/palm_oil_counting --cov-report=html
```

## Code Style

This project follows PEP 8 conventions with Black formatting (line-length 100).

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/ --fix

# Type checking
mypy src/
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Fazri Gading - [fazrigading@gmail.com](mailto:fazrigading@gmail.com)
