# AGENTS.md - Development Guidelines for AI Agents

This document provides guidelines for AI agents operating in this repository.

## Project Overview

Palm Oil Fruit Counting is a computer vision project for detecting, counting, and segmenting individual fruits on Fresh Fruit Bunches (FFB) of Palm Oil. It uses YOLO format for annotations and supports SAM2-based segmentation.

## Build, Lint, and Test Commands

### Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install all optional dependencies (including SAM2)
pip install -e ".[all]"
```

### Code Formatting and Linting

```bash
# Format code with Black (line-length: 100)
black src/ tests/

# Lint with Ruff (fixes automatically where possible)
ruff check src/ tests/ --fix

# Type checking with mypy
mypy src/
```

### Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run a single test file
pytest tests/test_filter.py -v

# Run a specific test class
pytest tests/test_filter.py::TestFilterFruitletMasks -v

# Run a specific test function
pytest tests/test_filter.py::TestFilterFruitletMasks::test_filter_small_masks -v

# Run tests with coverage report
pytest tests/ --cov=src/palm_oil_counting --cov-report=html

# Run tests matching a pattern
pytest -k "test_filter" -v
```

### Building the Package

```bash
# Build source distribution and wheel
python -m build

# Install in editable mode
pip install -e .
```

## Code Style Guidelines

### General Principles

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Write clean, readable, and maintainable code
- Add type hints to all public functions
- Use descriptive variable and function names

### Formatting

- **Line length**: Maximum 100 characters (configured in Black)
- **Indentation**: 4 spaces (no tabs)
- **Trailing commas**: Use where appropriate for better diffs
- **Quotes**: Use double quotes for strings, single quotes only when containing double quotes

### Type Hints

All functions must include type hints:

```python
from typing import List, Dict, Optional, Tuple, Any

def process_images(
    image_dir: str,
    output_dir: str,
    model_type: str = "tiny",
) -> List[Dict[str, Any]]:
    """Process images through the annotation pipeline."""
    pass
```

- Use `Optional[X]` instead of `X | None` for Python < 3.10 compatibility
- Use `List`, `Dict`, `Tuple` from typing module (not built-in types)
- Avoid using `Any` unless absolutely necessary

### Imports

Follow this order with blank lines between groups:

1. Standard library imports
2. Third-party imports
3. Local/application imports

```python
import os
import sys
from typing import List, Dict, Optional

import cv2
import torch
import numpy as np
from tqdm import tqdm

from palm_oil_counting.annotation.sam_annotator import filter_fruitlet_masks
from palm_oil_counting.utils import save_yolo_bbox
```

Use isort (integrated via Ruff) for automatic import sorting.

### Naming Conventions

- **Functions/variables**: `snake_case` (e.g., `filter_fruitlet_masks`, `image_path`)
- **Classes**: `PascalCase` (e.g., `TestFilterFruitletMasks`, `YOLOConverter`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_IMAGE_SIZE`, `DEFAULT_CONFIDENCE`)
- **Private functions/variables**: Prefix with underscore (e.g., `_internal_helper`)

### Docstrings

Use Google-style docstrings:

```python
def filter_fruitlet_masks(masks: List[Dict], image: np.ndarray) -> List[Dict]:
    """
    Filter out background masks to keep only palm oil fruitlets.
    
    Args:
        masks: List of SAM mask dictionaries.
        image: RGB image array (H, W, 3).
        
    Returns:
        Filtered list of mask dictionaries.
        
    Raises:
        ValueError: If image has invalid shape.
    """
```

- First line should be a concise summary
- Leave a blank line between summary and detailed description
- Document all parameters and return values
- Include raises section for exceptions

### Error Handling

- Use specific exception types
- Include meaningful error messages
- Handle exceptions at the appropriate level

```python
def load_image(path: str) -> np.ndarray:
    """Load an image from file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    
    return image
```

### Testing Guidelines

- Place tests in the `tests/` directory
- Name test files: `test_*.py`
- Name test classes: `Test*`
- Name test functions: `test_*`
- Use descriptive test names that explain what is being tested
- Use pytest fixtures for shared setup
- Test both positive and negative cases

```python
class TestFilterFruitletMasks:
    """Tests for filter_fruitlet_masks function."""
    
    def test_filter_small_masks(self):
        """Small masks should be filtered out."""
        masks = [{"segmentation": create_small_mask()}]
        filtered = filter_fruitlet_masks(masks, image)
        assert len(filtered) == 0
```

### Project Structure

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
│   ├── test_filter.py        # Tests for filtering functions
│   └── test_yolo_format.py   # Tests for YOLO utilities
├── scripts/                  # Command line scripts for entry points
│   ├── annotate.py           # Annotation tool
│   ├── crop.py               # Crop tool
│   ├── dataset.py            # Dataset tool
│   └── review.py             # Review utility
├── configs/                  # Configuration files
│   ├── annotator_config.yaml # Annotator tool config for session
│   ├── filter_config.yaml    # Filter mask config for review
│   └── train.yaml            # Training model config
└── pyproject.toml            # Project configuration
```

### Configuration Files

- **pyproject.toml**: Project metadata, dependencies, tool configurations
- **configs/**: YAML configuration files for training and annotation
- **.vscode/settings.json**: Editor settings (do not modify unless necessary)

### Git Conventions

Use conventional commit messages:

```text
feat: add dataset splitting functionality
fix: correct parameter order in sam_annotator
docs: update README with SAM2 instructions
test: add tests for yolo_format utilities
refactor: reorganize package structure
```

### Running GUI Applications

```bash
# Run the batch image cropper
palm-crop

# Run the annotation reviewer
palm-annotate
```

## Dependencies

### Required

- torch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python-headless >= 4.8.0
- numpy >= 1.24.0
- albumentations >= 1.3.0
- Pillow >= 10.0.0
- tqdm >= 4.65.0
- matplotlib >= 3.7.0
- pyyaml >= 6.0

### Development (Optional)

- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- black >= 23.0.0
- ruff >= 0.1.0
- mypy >= 1.0.0

### SAM2 (Optional)

- sam2 (for advanced segmentation)

## Key Files

- `pyproject.toml`: Project configuration and dependencies
- `CONTRIBUTING.md`: Detailed contribution guidelines
- `README.md`: Project documentation
- `MASTER-PLAN.md`: Project roadmap
