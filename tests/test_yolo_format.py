"""Tests for YOLO format utilities."""

import pytest
import tempfile
import os
import numpy as np

from palm_oil_counting.utils import (
    save_yolo_bbox,
    save_yolo_segmentation,
    load_yolo_annotations,
    validate_yolo_label,
    contours_to_yolo_format,
)


class TestSaveYoloBbox:
    """Tests for save_yolo_bbox function."""

    def test_save_single_bbox(self, tmp_path):
        masks = [{"bbox": [100, 100, 50, 50]}]
        output_path = tmp_path / "test.txt"

        save_yolo_bbox(masks, 640, 480, str(output_path))

        assert output_path.exists()
        content = output_path.read_text().strip()
        parts = content.split()

        assert parts[0] == "0"
        assert len(parts) == 5

        cx, cy, w, h = (
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
        assert 0 <= cx <= 1
        assert 0 <= cy <= 1
        assert 0 <= w <= 1
        assert 0 <= h <= 1

    def test_save_multiple_bboxes(self, tmp_path):
        masks = [
            {"bbox": [100, 100, 50, 50]},
            {"bbox": [200, 200, 30, 30]},
        ]
        output_path = tmp_path / "test.txt"

        save_yolo_bbox(masks, 640, 480, str(output_path))

        content = output_path.read_text().strip()
        lines = content.split("\n")
        assert len(lines) == 2


class TestSaveYoloSegmentation:
    """Tests for save_yolo_segmentation function."""

    def test_save_single_segmentation(self, tmp_path):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 1

        masks = [{"segmentation": mask}]
        output_path = tmp_path / "test.txt"

        save_yolo_segmentation(masks, 100, 100, str(output_path))

        assert output_path.exists()
        content = output_path.read_text().strip()
        parts = content.split()

        assert parts[0] == "0"
        assert len(parts) >= 7


class TestValidateYoloLabel:
    """Tests for validate_yolo_label function."""

    def test_valid_label(self, tmp_path):
        label_path = tmp_path / "valid.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n")

        is_valid, errors = validate_yolo_label(str(label_path))

        assert is_valid is True
        assert len(errors) == 0

    def test_empty_label(self, tmp_path):
        label_path = tmp_path / "empty.txt"
        label_path.write_text("")

        is_valid, errors = validate_yolo_label(str(label_path), check_empty=True)

        assert is_valid is False
        assert "Empty" in errors[0]

    def test_out_of_bounds(self, tmp_path):
        label_path = tmp_path / "bounds.txt"
        label_path.write_text("0 1.5 0.5 0.2 0.2\n")

        is_valid, errors = validate_yolo_label(str(label_path), check_bounds=True)

        assert is_valid is False
        assert any("out of bounds" in e for e in errors)

    def test_missing_file(self, tmp_path):
        label_path = tmp_path / "missing.txt"

        is_valid, errors = validate_yolo_label(str(label_path))

        assert is_valid is False
        assert "not found" in errors[0].lower()


class TestContoursToYoloFormat:
    """Tests for contours_to_yolo_format function."""

    def test_single_contour(self):
        contour = np.array([[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]])

        labels = contours_to_yolo_format([contour], 100, 100)

        assert len(labels) == 1
        assert labels[0].startswith("0 ")

    def test_small_contour_filtered(self):
        small_contour = np.array([[[10, 10]], [[12, 10]], [[12, 12]]])

        labels = contours_to_yolo_format([small_contour], 100, 100)

        assert len(labels) == 0


class TestLoadYoloAnnotations:
    """Tests for load_yolo_annotations function."""

    def test_load_segmentation(self, tmp_path):
        label_path = tmp_path / "seg.txt"
        label_path.write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")

        annotations = load_yolo_annotations(str(label_path), 100, 100)

        assert len(annotations) == 1
        assert annotations[0]["class_id"] == 0
        assert len(annotations[0]["points"]) == 4

    def test_load_bbox(self, tmp_path):
        label_path = tmp_path / "bbox.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n")

        annotations = load_yolo_annotations(str(label_path), 100, 100)

        assert len(annotations) == 1
        assert annotations[0]["class_id"] == 0
        assert len(annotations[0]["points"]) == 2
