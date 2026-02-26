"""Tests for SAM annotator filter functions."""

import pytest
import numpy as np

from palm_oil_counting.annotation.sam_annotator import filter_fruitlet_masks


class TestFilterFruitletMasks:
    """Tests for filter_fruitlet_masks function."""

    def create_mask(self, shape=(100, 100), region=(25, 25, 75, 75)):
        mask = np.zeros(shape, dtype=bool)
        x1, y1, x2, y2 = region
        mask[y1:y2, x1:x2] = True
        return mask

    def test_filter_small_masks(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        masks = [{"segmentation": self.create_mask(region=(10, 10, 15, 15))}]

        filtered = filter_fruitlet_masks(masks, image)

        assert len(filtered) == 0

    def test_filter_green_objects(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = [50, 200, 50]

        masks = [{"segmentation": self.create_mask()}]

        filtered = filter_fruitlet_masks(masks, image)

        assert len(filtered) == 0

    def test_filter_sky(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = [210, 210, 210]

        masks = [{"segmentation": self.create_mask()}]

        filtered = filter_fruitlet_masks(masks, image)

        assert len(filtered) == 0

    def test_keep_fruitlet_mask(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = [200, 100, 50]

        masks = [{"segmentation": self.create_mask()}]

        filtered = filter_fruitlet_masks(masks, image)

        assert len(filtered) == 1

    def test_filter_elongated_masks(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:] = [200, 100, 50]

        elongated_mask = np.zeros((100, 100), dtype=bool)
        elongated_mask[45:55, 5:95] = True

        masks = [{"segmentation": elongated_mask}]

        filtered = filter_fruitlet_masks(masks, image)

        assert len(filtered) == 0

    def test_empty_masks(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        masks = []

        filtered = filter_fruitlet_masks(masks, image)

        assert len(filtered) == 0
