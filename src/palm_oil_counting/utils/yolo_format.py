"""
YOLO format utilities for reading, writing, and validating annotations.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


def save_yolo_bbox(
    masks: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    output_path: str,
    class_id: int = 0,
) -> None:
    """
    Saves masks as YOLO detection format (class_id center_x center_y width height).

    Args:
        masks: List of mask dictionaries with 'bbox' key containing [x, y, w, h]
        img_w: Image width in pixels
        img_h: Image height in pixels
        output_path: Path to save the label file
        class_id: Class ID for the objects (default: 0)
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
    Saves masks as YOLO segmentation format (class_id x1 y1 x2 y2 ...).

    Args:
        masks: List of mask dictionaries with 'segmentation' key containing binary mask
        img_w: Image width in pixels
        img_h: Image height in pixels
        output_path: Path to save the label file
        class_id: Class ID for the objects (default: 0)
        epsilon_factor: Factor for contour simplification (default: 0.002)
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


def load_yolo_annotations(label_path: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """
    Loads YOLO format annotations from a label file.

    Args:
        label_path: Path to the YOLO label file
        img_w: Image width in pixels
        img_h: Image height in pixels

    Returns:
        List of annotation dictionaries with 'class_id', 'points', and optionally 'bbox'
    """
    annotations = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 3:
            continue

        class_id = int(parts[0])
        coords = parts[1:]

        points = []
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                x = coords[i] * img_w
                y = coords[i + 1] * img_h
                points.append((x, y))

        annotations.append({"type": "polygon", "class_id": class_id, "points": points})

    return annotations


def validate_yolo_label(
    label_path: str, check_empty: bool = True, check_bounds: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validates a YOLO format label file.

    Args:
        label_path: Path to the YOLO label file
        check_empty: Check if the file is empty
        check_bounds: Check if coordinates are within [0, 1]

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return False, [f"File not found: {label_path}"]

    if check_empty and len(lines) == 0:
        errors.append("Empty label file")

    for line_num, line in enumerate(lines, 1):
        parts = line.strip().split()

        if len(parts) < 3:
            errors.append(f"Line {line_num}: Too few values")
            continue

        try:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"Line {line_num}: Invalid number format")
            continue

        if len(coords) % 2 != 0:
            errors.append(f"Line {line_num}: Odd number of coordinates")

        if check_bounds:
            for i, val in enumerate(coords):
                if val < 0 or val > 1:
                    errors.append(f"Line {line_num}: Coordinate {i} out of bounds: {val}")

    return len(errors) == 0, errors


def contours_to_yolo_format(
    contours: List[np.ndarray],
    img_w: int,
    img_h: int,
    class_id: int = 0,
    epsilon_factor: float = 0.002,
    min_area: float = 100.0,
) -> List[str]:
    """
    Converts OpenCV contours to YOLO segmentation format strings.

    Args:
        contours: List of OpenCV contours
        img_w: Image width in pixels
        img_h: Image height in pixels
        class_id: Class ID for the objects
        epsilon_factor: Factor for contour simplification

    Returns:
        List of YOLO format label strings
    """
    yolo_labels = []

    for cnt in contours:
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        area = cv2.contourArea(approx)
        if area < min_area:
            continue

        points = approx.flatten()
        normalized_points = []
        for i in range(len(points)):
            if i % 2 == 0:
                normalized_points.append(float(points[i]) / img_w)
            else:
                normalized_points.append(float(points[i]) / img_h)

        label_str = f"{class_id} " + " ".join([f"{p:.6f}" for p in normalized_points])
        yolo_labels.append(label_str)

    return yolo_labels
