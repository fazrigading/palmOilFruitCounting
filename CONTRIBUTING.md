# Contributing to Palm Oil Fruit Counting

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/fazrigading/palmOilFruitCounting.git
cd palmOilFruitCounting
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.\.venv\Scripts\activate  # Windows
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

## Code Style

### Python Style Guide

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [Ruff](https://docs.astral.sh/ruff/) for linting
- Add type hints to all functions

### Formatting

```bash
black src/ tests/
ruff check src/ tests/ --fix
```

### Type Hints

All new functions should include type hints:

```python
from typing import List, Dict, Optional

def process_images(
    image_dir: str,
    output_dir: str,
    model_type: str = "tiny",
) -> None:
    """Process images through the annotation pipeline."""
    pass
```

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
    pass
```

## Testing

### Running Tests

```bash
pytest tests/ -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test classes `Test*`
- Name test functions `test_*`

Example:

```python
import pytest
import numpy as np

class TestFilterFruitletMasks:
    """Tests for filter_fruitlet_masks function."""
    
    def test_filter_small_masks(self):
        """Small masks should be filtered out."""
        masks = [{'segmentation': create_small_mask()}]
        filtered = filter_fruitlet_masks(masks, image)
        assert len(filtered) == 0
```

### Test Coverage

```bash
pytest tests/ --cov=src/palm_oil_counting --cov-report=html
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, documented code
- Add tests for new functionality
- Ensure all tests pass

### 3. Commit Changes

Use conventional commit messages:

```text
feat: add dataset splitting functionality
fix: correct parameter order in sam_annotator
docs: update README with SAM2 instructions
test: add tests for yolo_format utilities
refactor: reorganize package structure
```

Commit messages without naming convention will be rejected.

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages are descriptive

## Project Structure

```tree
src/palm_oil_counting/
├── annotation/        # Annotation tools
│   ├── sam_annotator.py    # SAM2-based annotation
│   └── hsv_annotator.py    # HSV color-based annotation
├── gui/               # GUI applications
│   ├── cropper.py          # Batch image cropper
│   └── annotator.py        # Annotation reviewer
├── preprocessing/     # Data preprocessing
│   └── augment.py          # Image augmentation
└── utils/             # Utilities
    ├── yolo_format.py      # YOLO format I/O
    └── visualization.py    # Visualization tools
```

## Reporting Issues

When reporting issues, please include:

1. Python version
2. Operating system
3. Steps to reproduce
4. Expected behavior
5. Actual behavior
6. Error messages/traceback

## Feature Requests

Feature requests are welcome! Please:

1. Check if the feature already exists
2. Check if there's an open issue
3. Open a new issue with:
   - Clear description
   - Use case
   - Possible implementation (optional)

## Questions?

For questions, open an issue or contact:

- Email: <fazrigading@gmail.com>
- GitHub: [@fazrigading](https://github.com/fazrigading)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
