"""
COCO format utilities for converting and managing annotations.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

COCO_CATEGORIES = [
    {"id": 0, "name": "overripe_fruitlet", "supercategory": "fruitlet"},
    {"id": 1, "name": "ripe_fruitlet", "supercategory": "fruitlet"},
    {"id": 2, "name": "unripe_fruitlet", "supercategory": "fruitlet"},
    {"id": 3, "name": "branch", "supercategory": "object"},
    {"id": 4, "name": "sky", "supercategory": "object"},
    {"id": 5, "name": "background", "supercategory": "object"},
]

COCO_SEGMENTATION_CATEGORIES = [
    {"id": 0, "name": "overripe_fruitlet", "supercategory": "fruitlet"},
    {"id": 1, "name": "ripe_fruitlet", "supercategory": "fruitlet"},
    {"id": 2, "name": "unripe_fruitlet", "supercategory": "fruitlet"},
]


@dataclass
class COCOImageInfo:
    id: int
    file_name: str
    width: int
    height: int
    license: int = 0
    flickr_url: str = ""
    coco_url: str = ""
    date_captured: str = ""


@dataclass
class COCOAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: List[Any]
    area: float
    bbox: List[float]
    iscrowd: int = 0
    ignore: int = 0


class COCOFormatter:
    """COCO format conversion utilities."""

    def __init__(
        self,
        categories: Optional[List[Dict[str, Any]]] = None,
        license_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize COCO formatter.

        Args:
            categories: List of category dictionaries
            license_info: License information dictionary
        """
        self.categories = categories or COCO_CATEGORIES
        self.license_info = license_info or {
            "id": 0,
            "name": "Palm Oil Fruit Dataset",
            "url": "",
        }

    def create_empty_dataset(
        self,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create empty COCO dataset structure.

        Args:
            info: Dataset info dictionary

        Returns:
            Empty COCO dataset dictionary
        """
        if info is None:
            info = {
                "description": "Palm Oil Fruit Dataset - SAM3 Annotations",
                "url": "",
                "version": "1.0",
                "year": 2026,
                "contributor": "",
                "date_created": "2026-01-01 00:00:00",
            }

        return {
            "info": info,
            "licenses": [self.license_info],
            "images": [],
            "annotations": [],
            "categories": self.categories,
        }

    def detections_to_coco_annotations(
        self,
        detections,
        image_id: int,
        image_width: int,
        image_height: int,
        start_annotation_id: int = 0,
        category_mapping: Optional[Dict[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert supervision detections to COCO annotation format.

        Args:
            detections: Supervision Detections object
            image_id: Image ID
            image_width: Image width in pixels
            image_height: Image height in pixels
            start_annotation_id: Starting annotation ID
            category_mapping: Mapping from detection class_id to COCO category_id

        Returns:
            List of COCO annotation dictionaries
        """
        if category_mapping is None:
            category_mapping = {i: i for i in range(len(self.categories))}

        annotations = []

        for det_idx, det in enumerate(detections):
            mask = det.mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                segmentation = [cnt.flatten().tolist()]

                class_id = det.class_id[0] if hasattr(det, "class_id") else 0
                coco_category_id = category_mapping.get(class_id, class_id)

                annotation = {
                    "id": start_annotation_id + len(annotations),
                    "image_id": image_id,
                    "category_id": coco_category_id,
                    "segmentation": segmentation,
                    "area": float(area),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "iscrowd": 0,
                }

                annotations.append(annotation)

        return annotations

    def save_coco_dataset(
        self,
        dataset: Dict[str, Any],
        output_path: str,
    ) -> None:
        """
        Save COCO dataset to JSON file.

        Args:
            dataset: COCO dataset dictionary
            output_path: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)

    def load_coco_dataset(
        self,
        json_path: str,
    ) -> Dict[str, Any]:
        """
        Load COCO dataset from JSON file.

        Args:
            json_path: Path to COCO JSON file

        Returns:
            COCO dataset dictionary
        """
        with open(json_path) as f:
            return json.load(f)

    def split_coco_dataset(
        self,
        dataset: Dict[str, Any],
        train_ratio: float = 0.8,
        seed: Optional[int] = 42,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Split COCO dataset into train and validation sets.

        Args:
            dataset: COCO dataset dictionary
            train_ratio: Ratio of training samples
            seed: Random seed

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if seed is not None:
            np.random.seed(seed)

        images = dataset["images"]
        num_images = len(images)

        indices = np.random.permutation(num_images)
        train_count = int(num_images * train_ratio)

        train_indices = set(indices[:train_count])
        val_indices = set(indices[train_count:])

        def split_dataset(indices_set):
            new_images = [images[i] for i in sorted(indices_set)]
            image_ids = {img["id"] for img in new_images}
            new_annotations = [
                ann for ann in dataset["annotations"] if ann["image_id"] in image_ids
            ]

            new_dataset = dataset.copy()
            new_dataset["images"] = new_images
            new_dataset["annotations"] = new_annotations
            return new_dataset

        train_dataset = split_dataset(train_indices)
        val_dataset = split_dataset(val_indices)

        return train_dataset, val_dataset


def yolo_to_coco(
    yolo_labels_dir: str,
    images_dir: str,
    output_path: str,
    categories: Optional[List[Dict[str, Any]]] = None,
    split_ratio: Optional[float] = None,
) -> None:
    """
    Convert YOLO format annotations to COCO format.

    Args:
        yolo_labels_dir: Directory containing YOLO label files
        images_dir: Directory containing images
        output_path: Path to save COCO JSON file
        categories: List of COCO categories
        split_ratio: Optional split ratio (if provided, creates train.json and val.json)
    """
    formatter = COCOFormatter(categories=categories or COCO_SEGMENTATION_CATEGORIES)
    dataset = formatter.create_empty_dataset()

    image_files = [
        f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    annotation_id = 0

    for img_idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        dataset["images"].append(
            {
                "id": img_idx + 1,
                "file_name": img_file,
                "width": w,
                "height": h,
            }
        )

        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(yolo_labels_dir, label_file)

        if os.path.exists(label_path):
            with open(label_path) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                points = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        px = coords[i] * w
                        py = coords[i + 1] * h
                        points.append([px, py])

                if len(points) < 3:
                    continue

                cnt = np.array(points, dtype=np.int32)
                area = cv2.contourArea(cnt)
                x, y, bw, bh = cv2.boundingRect(cnt)

                dataset["annotations"].append(
                    {
                        "id": annotation_id + 1,
                        "image_id": img_idx + 1,
                        "category_id": class_id,
                        "segmentation": [cnt.flatten().tolist()],
                        "area": float(area),
                        "bbox": [float(x), float(y), float(bw), float(bh)],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    formatter.save_coco_dataset(dataset, output_path)

    if split_ratio is not None:
        train_dataset, val_dataset = formatter.split_coco_dataset(dataset, split_ratio)

        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(output_path)[0]

        formatter.save_coco_dataset(train_dataset, os.path.join(output_dir, "train.json"))
        formatter.save_coco_dataset(val_dataset, os.path.join(output_dir, "val.json"))


def coco_to_yolo_segmentation(
    coco_json_path: str,
    output_dir: str,
    images_dir: str,
) -> None:
    """
    Convert COCO format to YOLO segmentation format.

    Args:
        coco_json_path: Path to COCO JSON file
        output_dir: Directory to save YOLO label files
        images_dir: Directory containing images (for getting dimensions)
    """
    formatter = COCOFormatter()
    dataset = formatter.load_coco_dataset(coco_json_path)

    os.makedirs(output_dir, exist_ok=True)

    image_info = {img["id"]: img for img in dataset["images"]}

    for ann in dataset["annotations"]:
        image_id = ann["image_id"]
        img = image_info.get(image_id)

        if img is None:
            continue

        w = img["width"]
        h = img["height"]

        segmentation = ann["segmentation"]
        if not segmentation:
            continue

        points = segmentation[0]

        normalized_points = []
        for i in range(0, len(points), 2):
            px = points[i] / w if i < len(points) else 0
            py = points[i + 1] / h if i + 1 < len(points) else 0
            normalized_points.append(f"{px:.6f}")
            normalized_points.append(f"{py:.6f}")

        category_id = ann["category_id"]

        label_line = f"{category_id} {' '.join(normalized_points)}"

        output_file = os.path.join(output_dir, f"{img['file_name']}.txt")

        with open(output_file, "a") as f:
            f.write(label_line + "\n")
