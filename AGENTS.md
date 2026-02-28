# AGENTS.md - Development Guidelines for AI Agents

Palm Oil Fruit Counting is a computer vision project for detecting, counting, and segmenting individual fruits on Fresh Fruit Bunches (FFB) of Palm Oil. It uses YOLO format for annotations and supports SAM2-based segmentation.

## Build, Lint, and Test Commands

### Installation

```bash
pip install -e ".[dev]"    # With development dependencies
pip install -e ".[all]"   # With all optional dependencies (including SAM2)
```

### Code Formatting and Linting

```bash
black src/ tests/          # Format code (line-length: 100)
ruff check src/ tests/ --fix   # Lint with auto-fix
mypy src/                 # Type checking
```

### Running Tests

```bash
pytest tests/ -v                              # All tests
pytest tests/test_filter.py -v                # Single test file
pytest tests/test_filter.py::TestFilterClass  # Test class
pytest tests/test_filter.py::test_function    # Single test function
pytest tests/ --cov=src/palm_oil_counting      # With coverage
pytest -k "test_filter" -v                    # Pattern matching
```

### Build Package

```bash
python -m build    # Build source distribution and wheel
pip install -e .   # Install in editable mode
```

## Code Style Guidelines

### General

- Follow PEP 8 conventions
- Add type hints to all public functions
- Use descriptive variable and function names

### Formatting

- **Line length**: 100 characters max
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings, single only when containing double quotes
- Use trailing commas where appropriate

### Type Hints

```python
from typing import List, Dict, Optional, Tuple, Any

def process_images(image_dir: str, output_dir: str, model_type: str = "tiny") -> List[Dict[str, Any]]:
    """Process images through the annotation pipeline."""
    pass
```

- Use `Optional[X]` instead of `X | None` for Python < 3.10 compatibility
- Use `List`, `Dict`, `Tuple` from typing module (not built-in types)
- Avoid `Any` unless absolutely necessary

### Imports

Order: 1) Standard library, 2) Third-party, 3) Local imports. Use isort (via Ruff).

```python
import os
from typing import List, Dict, Optional

import cv2
import numpy as np
from tqdm import tqdm

from palm_oil_counting.annotation.sam_annotator import filter_fruitlet_masks
```

### Naming Conventions

- **Functions/variables**: `snake_case` (e.g., `filter_fruitlet_masks`)
- **Classes**: `PascalCase` (e.g., `TestFilterFruitletMasks`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_IMAGE_SIZE`)
- **Private members**: Prefix with underscore (e.g., `_internal_helper`)

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

- First line: concise summary
- Document all parameters, return values, and exceptions

### Error Handling

- Use specific exception types
- Include meaningful error messages
- Handle exceptions at the appropriate level

```python
def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image
```

### Testing Guidelines

- Place tests in `tests/` directory
- Name: `test_*.py` files, `Test*` classes, `test_*` functions
- Use descriptive test names and pytest fixtures

```python
class TestFilterFruitletMasks:
    def test_filter_small_masks(self):
        """Small masks should be filtered out."""
        masks = [{"segmentation": create_small_mask()}]
        filtered = filter_fruitlet_masks(masks, image)
        assert len(filtered) == 0
```

## Project Structure

```tree
palmOilFruitCounting/
├── src/palm_oil_counting/
│   ├── annotation/       # sam_annotator.py, hsv_annotator.py
│   ├── gui/              # cropper.py, annotator.py
│   ├── preprocessing/   # augment.py
│   └── utils/            # yolo_format.py, visualization.py
├── tests/                # test_filter.py, test_yolo_format.py
├── scripts/              # annotate.py, crop.py, dataset.py, review.py
├── configs/              # YAML configuration files
└── pyproject.toml        # Project configuration
```

## Git Conventions

Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

## Running GUI Applications

```bash
palm-crop       # Batch image cropper
palm-annotate   # Annotation reviewer
```

## Dependencies

- **Required**: torch, torchvision, opencv-python-headless, numpy, albumentations, Pillow, tqdm, matplotlib, pyyaml
- **Development**: pytest, pytest-cov, black, ruff, mypy
- **Optional**: sam2 (for advanced segmentation)
