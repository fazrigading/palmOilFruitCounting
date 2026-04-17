"""
SAM3-based automatic annotation for palm oil fruit detection.

This module provides automatic segmentation using Meta's Segment Anything Model 3 (SAM3)
with autodistill framework for text-based prompting, optimized for dual GPU batch processing.
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

try:
    from autodistill_sam3 import SegmentAnything3
    from autodistill.detection import CaptionOntology
    from autodistill.helpers import load_image
    import supervision as sv

    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False


ONTOLOGY_CLASSES = {
    "overripe fruitlet": "overripe_fruitlet",
    "ripe fruitlet": "ripe_fruitlet",
    "unripe fruitlet": "unripe_fruitlet",
    "branch": "branch",
    "sky": "sky",
    "background": "background",
}

ONTOLOGY_MINIMAL = {
    "fruitlet": "fruitlet",
}


@dataclass
class SAM3Config:
    model_type: str = "base"
    device: str = "cuda"
    batch_size: int = 4
    points_per_side: int = 32
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.92


def map_class_name_to_id(class_name: str) -> int:
    """Map class name to numeric ID for YOLO format."""
    class_mapping = {
        "overripe_fruitlet": 0,
        "ripe_fruitlet": 1,
        "unripe_fruitlet": 2,
        "branch": 3,
        "sky": 4,
        "background": 5,
    }
    return class_mapping.get(class_name, -1)


def get_ripeness_color_range(ripeness_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get HSV color range for filtering fruitlets by ripeness.

    Args:
        ripeness_type: Type of ripeness - 'overripe', 'ripe', or 'unripe'

    Returns:
        Tuple of (lower_bound, upper_bound) HSV arrays
    """
    color_ranges = {
        "overripe": (
            np.array([0, 50, 20]),
            np.array([20, 255, 100]),
        ),
        "ripe": (
            np.array([10, 100, 100]),
            np.array([25, 255, 255]),
        ),
        "unripe": (
            np.array([100, 50, 0]),
            np.array([140, 255, 80]),
        ),
    }
    return color_ranges.get(ripeness_type, (np.array([0]), np.array([255])))


def filter_by_color_and_size(
    masks: List[Dict[str, Any]],
    image: np.ndarray,
    min_area: int = 100,
    max_area: int = 50000,
    min_aspect_ratio: float = 0.3,
    max_aspect_ratio: float = 3.3,
) -> List[Dict[str, Any]]:
    """
    Filter masks based on color, size, and aspect ratio for palm oil fruitlets.

    Args:
        masks: List of SAM mask dictionaries
        image: RGB image array (H, W, 3)
        min_area: Minimum mask area in pixels
        max_area: Maximum mask area in pixels
        min_aspect_ratio: Minimum aspect ratio
        max_aspect_ratio: Maximum aspect ratio

    Returns:
        Filtered list of mask dictionaries
    """
    filtered_masks = []

    for mask_data in masks:
        mask = mask_data["segmentation"].astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = contours[0]
        area = cv2.contourArea(cnt)

        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        aspect_ratio = float(w) / h
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.2:
            continue

        mean_color = cv2.mean(image, mask=(mask_data["segmentation"].astype(np.uint8)))[:3]
        R, G, B = mean_color

        is_dark_fruitlet = (R + G + B) / 3.0 < 80

        if not is_dark_fruitlet:
            if G > R * 1.2 and G > B:
                continue
            if B > R * 1.2 and B > G:
                continue
            if R > 200 and G > 200 and B > 200:
                continue

        filtered_masks.append(mask_data)

    return filtered_masks


def save_yolo_bbox(
    masks: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    output_path: str,
    class_id: int = 0,
) -> None:
    """
    Save masks as YOLO detection format (class_id center_x center_y width height).

    Args:
        masks: List of mask dictionaries with 'bbox' key
        img_w: Image width in pixels
        img_h: Image height in pixels
        output_path: Path to save the label file
        class_id: Class ID for the objects
    """
    with open(output_path, "w") as f:
        for mask_data in masks:
            x, y, w, h = mask_data["bbox"]

            center_x = (x + w / 2) / img_w
            center_y = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")


def save_yolo_segmentation(
    masks: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    output_path: str,
    class_id: int = 0,
    epsilon_factor: float = 0.002,
) -> None:
    """
    Save masks as YOLO segmentation format (class_id x1 y1 x2 y2 ...).

    Args:
        masks: List of mask dictionaries with 'segmentation' key
        img_w: Image width in pixels
        img_h: Image height in pixels
        output_path: Path to save the label file
        class_id: Class ID for the objects
        epsilon_factor: Factor for contour simplification
    """
    with open(output_path, "w") as f:
        for mask_data in masks:
            mask = mask_data["segmentation"].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                epsilon = epsilon_factor * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) < 3:
                    continue

                points = approx.flatten()
                normalized_points = []
                for i in range(0, len(points), 2):
                    normalized_points.append(f"{points[i] / img_w:.6f}")
                    normalized_points.append(f"{points[i + 1] / img_h:.6f}")

                f.write(f"{class_id} {' '.join(normalized_points)}\n")


def save_coco_annotations(
    results: list[dict[str, Any]],
    images_info: list[dict[str, Any]],
    output_path: str,
    categories: Optional[list[dict[str, Any]]] = None,
) -> None:
    """
    Save annotations in COCO segmentation format.

    Args:
        results: List of detection results with image_id, category_id, segmentation, bbox
        images_info: List of image information dicts
        output_path: Path to save the JSON file
        categories: List of category dicts
    """
    if categories is None:
        categories = [
            {"id": 0, "name": "overripe_fruitlet", "supercategory": "fruitlet"},
            {"id": 1, "name": "ripe_fruitlet", "supercategory": "fruitlet"},
            {"id": 2, "name": "unripe_fruitlet", "supercategory": "fruitlet"},
            {"id": 3, "name": "branch", "supercategory": "object"},
            {"id": 4, "name": "sky", "supercategory": "object"},
            {"id": 5, "name": "background", "supercategory": "object"},
        ]

    coco_dict = {
        "images": images_info,
        "annotations": results,
        "categories": categories,
    }

    import json

    with open(output_path, "w") as f:
        json.dump(coco_dict, f, indent=2)


class SAM3Annotator:
    """
    SAM3-based automatic annotator for palm oil fruit detection.
    """

    def __init__(
        self,
        ontology: Optional[Dict[str, str]] = None,
        device: str = "cuda",
        config: Optional[SAM3Config] = None,
    ):
        """
        Initialize SAM3 annotator.

        Args:
            ontology: Dictionary mapping prompts to class names
            device: Device to run on ('cuda' or 'cpu')
            config: SAM3 configuration
        """
        if not SAM3_AVAILABLE:
            raise ImportError(
                "autodistill-sam3 is not installed. Install with: pip install autodistill-sam3"
            )

        self.device = device
        self.config = config or SAM3Config()

        if ontology is None:
            ontology = ONTOLOGY_CLASSES

        self.ontology = CaptionOntology(ontology)

        self.base_model = SegmentAnything3(ontology=self.ontology)

    def predict(
        self,
        image: Union[str, np.ndarray],
        return_format: str = "cv2",
    ) -> sv.Detections:
        """
        Run inference on a single image.

        Args:
            image: Image path or numpy array
            return_format: Format for loaded image ('cv2' or 'numpy')

        Returns:
            Supervision Detections object
        """
        if isinstance(image, str):
            image = load_image(image, return_format=return_format)

        detections = self.base_model.predict(image)
        return detections

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 4,
    ) -> List[sv.Detections]:
        """
        Run inference on a batch of images.

        Args:
            images: List of image paths or numpy arrays
            batch_size: Batch size for processing

        Returns:
            List of Supervision Detections objects
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch_results = [self.predict(img) for img in batch]
            results.extend(batch_results)

        return results

    def label(
        self,
        input_dir: str,
        output_dir: str,
        output_format: str = "both",
        extension: str = ".jpg",
    ) -> None:
        """
        Label a directory of images.

        Args:
            input_dir: Directory containing images
            output_dir: Directory for output annotations
            output_format: Output format ('yolo', 'coco', or 'both')
            extension: Image file extension
        """
        os.makedirs(output_dir, exist_ok=True)

        image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        images = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]

        print(f"Found {len(images)} images. Starting SAM3 annotation...")

        yolo_bbox_dir = os.path.join(output_dir, "labels_bbox")
        yolo_seg_dir = os.path.join(output_dir, "labels_seg")
        coco_dir = os.path.join(output_dir, "coco_annotations")

        if output_format in ("yolo", "both"):
            os.makedirs(yolo_bbox_dir, exist_ok=True)
            os.makedirs(yolo_seg_dir, exist_ok=True)

        if output_format in ("coco", "both"):
            os.makedirs(coco_dir, exist_ok=True)

        all_annotations = []
        images_info = []

        for idx, img_name in enumerate(tqdm(images)):
            img_path = os.path.join(input_dir, img_name)

            detections = self.predict(img_path)
            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            base_name = os.path.splitext(img_name)[0]

            filtered = filter_by_color_and_size(
                [
                    {
                        "segmentation": det.mask,
                        "bbox": det.bbox,
                    }
                    for det in detections
                ],
                image,
            )

            class_ids = [
                map_class_name_to_id(self.ontology.classes()[cid]) for cid in detections.class_id
            ]

            if output_format in ("yolo", "both"):
                save_yolo_bbox(
                    filtered,
                    w,
                    h,
                    os.path.join(yolo_bbox_dir, f"{base_name}.txt"),
                )
                save_yolo_segmentation(
                    filtered,
                    w,
                    h,
                    os.path.join(yolo_seg_dir, f"{base_name}.txt"),
                )

            if output_format in ("coco", "both"):
                images_info.append(
                    {
                        "id": idx + 1,
                        "file_name": img_name,
                        "width": w,
                        "height": h,
                    }
                )

                for det_idx, det in enumerate(detections):
                    mask = det.mask.astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < 50:
                            continue

                        segmentation = [cnt.flatten().tolist()]

                        all_annotations.append(
                            {
                                "id": len(all_annotations) + 1,
                                "image_id": idx + 1,
                                "category_id": class_ids[det_idx],
                                "segmentation": segmentation,
                                "area": float(area),
                                "bbox": [
                                    float(det.bbox[0]),
                                    float(det.bbox[1]),
                                    float(det.bbox[2]),
                                    float(det.bbox[3]),
                                ],
                                "iscrowd": 0,
                            }
                        )

        if output_format in ("coco", "both"):
            save_coco_annotations(
                all_annotations,
                images_info,
                os.path.join(coco_dir, "annotations.json"),
            )

        print(f"Annotation complete. Results saved to {output_dir}")


def process_images(
    input_dir: str,
    output_dir: str,
    output_format: str = "both",
    use_full_ontology: bool = True,
    device: str = "cuda",
    batch_size: int = 4,
) -> None:
    """
    Process images through SAM3 for automatic annotation.

    Args:
        input_dir: Path to directory containing images
        output_dir: Path to output directory for labels
        output_format: Output format ('yolo', 'coco', or 'both')
        use_full_ontology: Use full class ontology or minimal
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for processing
    """
    if not SAM3_AVAILABLE:
        raise ImportError(
            "autodistill-sam3 is not installed. Install with: pip install autodistill-sam3"
        )

    ontology = ONTOLOGY_CLASSES if use_full_ontology else ONTOLOGY_MINIMAL

    config = SAM3Config(device=device, batch_size=batch_size)

    annotator = SAM3Annotator(ontology=ontology, device=device, config=config)

    annotator.label(
        input_dir=input_dir,
        output_dir=output_dir,
        output_format=output_format,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Automatic Annotator for Palm Oil Fruit")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to images directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/sam3_annotations",
        help="Path to output directory",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="both",
        choices=["yolo", "coco", "both"],
        help="Output format",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="full",
        choices=["full", "minimal"],
        help="Class selection strategy",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for GPU processing",
    )

    args = parser.parse_args()

    process_images(
        input_dir=args.input,
        output_dir=args.output,
        output_format=args.output_format,
        use_full_ontology=(args.classes == "full"),
        device=args.device,
        batch_size=args.batch_size,
    )
