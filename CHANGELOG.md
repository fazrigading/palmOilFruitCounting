# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-02

### Added

- SAM2-based segmentation support for advanced fruitlet detection
- HSV color space annotation for palm oil fruit detection
- GUI cropper tool for batch image cropping (`palm-crop`)
- GUI annotation reviewer tool (`palm-annotate`)
- YOLO format utilities for dataset conversion
- Visualization utilities for annotations
- Image augmentation pipeline using albumentations
- Contours to YOLO format conversion

### Fixed

- Fixed mypy dict declaration in dataset.py
- Fixed black objects detection in annotator
- Fixed error with `torch.cuda.is_device_available`  
- Fixed optional declarative on hsv_annotator
- Fixed mypy errors across the codebase

### Refactored

- Cropper GUI refactoring
- Refactored `contours_to_yolo_format` to include `min_area` parameter

### Documentation

- Updated README.md with project structure
- Added dataset preparation guide
- Added AGENTS.md with development guidelines
- Updated project structure tree
- Added Python package CI workflow
- Updated Python version requirement to 3.9+

### Config

- Added personal configuration as an example
- Updated pyproject.toml with comprehensive project metadata
