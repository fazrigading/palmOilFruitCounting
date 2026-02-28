"""
HSV-based automatic annotation for palm oil fruit detection.

This module provides automatic segmentation using HSV color thresholding
for ripe/orange palm oil fruits.
"""

import cv2
import numpy as np
import os
import glob
from typing import List, Tuple, Optional


def get_yolo_segmentation_format(
    contours: List[np.ndarray], img_w: int, img_h: int, class_id: int = 0
) -> List[str]:
    """
    Converts OpenCV contours to YOLO segmentation format.

    Args:
        contours: List of OpenCV contours
        img_w: Image width in pixels
        img_h: Image height in pixels
        class_id: Class ID for the objects

    Returns:
        List of YOLO format label strings
    """
    yolo_labels = []
    for cnt in contours:
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
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


def segment_fruits(
    image_path: str, output_dir: str, visualize_dir: Optional[str] = None, min_area: int = 500
) -> None:
    """
    Segments ripe palm oil fruits using HSV color thresholding.

    Args:
        image_path: Path to the input image
        output_dir: Directory to save YOLO labels
        visualize_dir: Directory to save visualization images (optional)
        min_area: Minimum contour area to consider
    """
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_orange1 = np.array([0, 70, 50])
    upper_orange1 = np.array([25, 255, 255])

    lower_orange2 = np.array([160, 70, 50])
    upper_orange2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
    mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(output_dir, f"{base_name}.txt")

    yolo_labels = get_yolo_segmentation_format(filtered_contours, w, h)

    with open(label_path, "w") as f:
        for label in yolo_labels:
            f.write(label + "\n")

    if visualize_dir:
        vis_img = img.copy()
        cv2.drawContours(vis_img, filtered_contours, -1, (0, 255, 0), 2)
        vis_path = os.path.join(visualize_dir, f"{base_name}_vis.jpg")
        cv2.imwrite(vis_path, vis_img)


def process_directory(
    dataset_dir: str, labels_dir: Optional[str] = None, vis_dir: Optional[str] = None, min_area: int = 500
) -> None:
    """
    Process all images in a directory for HSV-based annotation.

    Args:
        dataset_dir: Directory containing images
        labels_dir: Directory to save labels (default: dataset_dir/labels)
        vis_dir: Directory to save visualizations (default: dataset_dir/visualized)
        min_area: Minimum contour area to consider
    """
    if labels_dir is None:
        labels_dir = os.path.join(dataset_dir, "labels")
    if vis_dir is None:
        vis_dir = os.path.join(dataset_dir, "visualized")

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
        image_paths.extend(
            glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True)
        )

    image_paths = [
        p for p in image_paths if "labels" not in p and "visualized" not in p
    ]
    image_paths = sorted(list(set(image_paths)))

    print(f"Found {len(image_paths)} images. Starting annotation...")

    for i, img_path in enumerate(image_paths):
        if i % 50 == 0:
            print(f"Processing image {i}/{len(image_paths)}: {img_path}")
        segment_fruits(img_path, labels_dir, vis_dir, min_area)

    print(
        f"Annotation complete. Labels saved in '{labels_dir}' and visualizations in '{vis_dir}'."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HSV-based automatic annotator for palm oil fruit"
    )
    parser.add_argument(
        "--input", type=str, default="dataset", help="Path to images directory"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to output labels directory"
    )
    parser.add_argument(
        "--visualize", type=str, default=None, help="Path to visualization directory"
    )
    parser.add_argument(
        "--min-area", type=int, default=500, help="Minimum contour area"
    )

    args = parser.parse_args()

    process_directory(
        dataset_dir=args.input,
        labels_dir=args.output,
        vis_dir=args.visualize,
        min_area=args.min_area,
    )
