#!/usr/bin/env python3
"""Dataset utilities - split, validate, and analyze datasets."""

import os
import shutil
import argparse
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Split dataset into train/val/test sets.

    Args:
        images_dir: Path to images directory
        labels_dir: Path to labels directory
        output_dir: Output directory for split dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in os.listdir(images_dir) if Path(f).suffix.lower() in image_extensions]

    random.shuffle(image_files)

    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train : n_train + n_val],
        "test": image_files[n_train + n_val :],
    }

    for split_name in splits:
        for subdir in ["images", "labels"]:
            os.makedirs(os.path.join(output_dir, split_name, subdir), exist_ok=True)

    for split_name, files in splits.items():
        print(f"Processing {split_name}: {len(files)} images")

        for img_file in files:
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_dir, split_name, "images", img_file)
            shutil.copy2(src_img, dst_img)

            label_file = Path(img_file).stem + ".txt"
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(output_dir, split_name, "labels", label_file)

            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f"Warning: Label not found for {img_file}")

    print(f"\nDataset split complete!")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val:   {len(splits['val'])} images")
    print(f"  Test:  {len(splits['test'])} images")


def validate_dataset(
    images_dir: str,
    labels_dir: str,
    check_empty: bool = True,
    check_bounds: bool = True,
) -> Tuple[int, List[str]]:
    """
    Validate dataset for common issues.

    Args:
        images_dir: Path to images directory
        labels_dir: Path to labels directory
        check_empty: Check for empty label files
        check_bounds: Check for out-of-bounds coordinates

    Returns:
        Tuple of (error_count, list of error messages)
    """
    from palm_oil_counting.utils import validate_yolo_label

    errors = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_file in os.listdir(images_dir):
        if Path(img_file).suffix.lower() not in image_extensions:
            continue

        label_file = Path(img_file).stem + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            errors.append(f"Missing label: {label_file}")
            continue

        is_valid, label_errors = validate_yolo_label(
            label_path, check_empty=check_empty, check_bounds=check_bounds
        )

        for err in label_errors:
            errors.append(f"{label_file}: {err}")

    return len(errors), errors


def analyze_dataset(images_dir: str, labels_dir: str) -> Dict[str, Any]:
    """
    Analyze dataset statistics.

    Args:
        images_dir: Path to images directory
        labels_dir: Path to labels directory

    Returns:
        Dictionary with dataset statistics
    """
    import cv2

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    stats: Dict[str, Any] = {
        "total_images": 0,
        "total_labels": 0,
        "total_objects": 0,
        "empty_labels": 0,
        "missing_labels": 0,
        "class_distribution": {},
        "image_sizes": [],
        "objects_per_image": [],
    }

    for img_file in os.listdir(images_dir):
        if Path(img_file).suffix.lower() not in image_extensions:
            continue

        stats["total_images"] += 1

        label_file = Path(img_file).stem + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            stats["missing_labels"] += 1
            continue

        stats["total_labels"] += 1

        with open(label_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            stats["empty_labels"] += 1

        stats["objects_per_image"].append(len(lines))
        stats["total_objects"] += len(lines)

        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                stats["class_distribution"][class_id] = (
                    stats["class_distribution"].get(class_id, 0) + 1
                )

        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            stats["image_sizes"].append((w, h))

    if stats["objects_per_image"]:
        stats["avg_objects_per_image"] = sum(stats["objects_per_image"]) / len(
            stats["objects_per_image"]
        )

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    split_parser = subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("--images", type=str, required=True, help="Images directory")
    split_parser.add_argument("--labels", type=str, required=True, help="Labels directory")
    split_parser.add_argument("--output", type=str, required=True, help="Output directory")
    split_parser.add_argument("--train", type=float, default=0.8, help="Train ratio")
    split_parser.add_argument("--val", type=float, default=0.1, help="Val ratio")
    split_parser.add_argument("--test", type=float, default=0.1, help="Test ratio")

    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("--images", type=str, required=True, help="Images directory")
    validate_parser.add_argument("--labels", type=str, required=True, help="Labels directory")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze dataset statistics")
    analyze_parser.add_argument("--images", type=str, required=True, help="Images directory")
    analyze_parser.add_argument("--labels", type=str, required=True, help="Labels directory")

    args = parser.parse_args()

    if args.command == "split":
        split_dataset(args.images, args.labels, args.output, args.train, args.val, args.test)
    elif args.command == "validate":
        error_count, errors = validate_dataset(args.images, args.labels)
        print(f"Found {error_count} errors:")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    elif args.command == "analyze":
        stats = analyze_dataset(args.images, args.labels)
        print("Dataset Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total labels: {stats['total_labels']}")
        print(f"  Total objects: {stats['total_objects']}")
        print(f"  Empty labels: {stats['empty_labels']}")
        print(f"  Missing labels: {stats['missing_labels']}")
        print(f"  Avg objects/image: {stats.get('avg_objects_per_image', 0):.2f}")
        print(f"  Class distribution: {stats['class_distribution']}")
    else:
        parser.print_help()
