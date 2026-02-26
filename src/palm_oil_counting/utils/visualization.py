"""
Visualization utilities for annotations and masks.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


def draw_annotations(
    image: np.ndarray,
    annotations: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Draws polygon annotations on an image.

    Args:
        image: Input image (BGR format)
        annotations: List of annotation dictionaries with 'points' key
        color: Polygon color (BGR)
        thickness: Line thickness
        show_labels: Whether to show annotation IDs

    Returns:
        Image with annotations drawn
    """
    result = image.copy()

    for idx, ann in enumerate(annotations):
        points = ann.get("points", [])
        if not points:
            continue

        pts = np.array(points, dtype=np.int32)
        cv2.polylines(result, [pts], True, color, thickness)

        if show_labels:
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(
                result, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

    return result


def draw_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    alpha: float = 0.5,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Draws binary masks on an image with transparency.

    Args:
        image: Input image (BGR format)
        masks: List of binary masks
        alpha: Transparency factor (0-1)
        color: Mask color (BGR). If None, generates random colors.

    Returns:
        Image with masks overlaid
    """
    result = image.copy()

    for mask in masks:
        if color is None:
            mask_color = tuple(np.random.randint(0, 255, 3).tolist())
        else:
            mask_color = color

        colored_mask = np.zeros_like(result)
        colored_mask[mask > 0] = mask_color

        result = cv2.addWeighted(result, 1, colored_mask, alpha, 0)

    return result


def draw_bboxes(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draws bounding boxes on an image.

    Args:
        image: Input image (BGR format)
        bboxes: List of (x1, y1, x2, y2) tuples
        color: Box color (BGR)
        thickness: Line thickness

    Returns:
        Image with bounding boxes drawn
    """
    result = image.copy()

    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

    return result


def visualize_yolo_labels(
    image_path: str, label_path: str, output_path: str, is_segmentation: bool = True
) -> None:
    """
    Visualizes YOLO labels on an image and saves the result.

    Args:
        image_path: Path to the image file
        label_path: Path to the YOLO label file
        output_path: Path to save the visualized image
        is_segmentation: Whether labels are segmentation (True) or bbox (False)
    """
    image = cv2.imread(image_path)
    if image is None:
        return

    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 5:
            continue

        class_id = int(parts[0])

        if is_segmentation:
            coords = parts[1:]
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = int(coords[i] * w)
                    y = int(coords[i + 1] * h)
                    points.append([x, y])

            if points:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        else:
            cx, cy, bw, bh = parts[1:5]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
