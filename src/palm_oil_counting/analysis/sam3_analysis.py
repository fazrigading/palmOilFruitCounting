"""
Statistical analysis module for comparing SAM2 and SAM3 annotations.
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class AnnotationComparison:
    """Container for annotation comparison results."""

    sam2_count: int
    sam3_count: int
    iou_scores: List[float]
    class_matches: Dict[str, int]
    unmatched_sam2: List[str]
    unmatched_sam3: List[str]


@dataclass
class ClassStatistics:
    """Statistics for a specific class."""

    class_name: str
    count: int
    avg_area: float
    min_area: float
    max_area: float
    std_area: float


class SAMComparator:
    """Compare annotations between SAM2 and SAM3."""

    def __init__(
        self,
        min_iou_threshold: float = 0.5,
        min_area: int = 100,
    ):
        """
        Initialize comparator.

        Args:
            min_iou_threshold: Minimum IoU for considering a match
            min_area: Minimum mask area to consider
        """
        self.min_iou_threshold = min_iou_threshold
        self.min_area = min_area

    def calculate_iou(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray,
    ) -> float:
        """
        Calculate Intersection over Union between two masks.

        Args:
            mask1: First binary mask
            mask2: Second binary mask

        Returns:
            IoU score
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def load_yolo_masks(
        self,
        label_path: str,
        img_w: int,
        img_h: int,
    ) -> List[Dict[str, Any]]:
        """
        Load masks from YOLO label file.

        Args:
            label_path: Path to YOLO label file
            img_w: Image width in pixels
            img_h: Image height in pixels

        Returns:
            List of mask dictionaries
        """
        masks = []

        if not os.path.exists(label_path):
            return masks

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
                    px = int(coords[i] * img_w)
                    py = int(coords[i + 1] * img_h)
                    points.append([px, py])

            if len(points) < 3:
                continue

            cnt = np.array(points, dtype=np.int32)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [cnt], 1)

            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            masks.append(
                {
                    "class_id": class_id,
                    "mask": mask,
                    "area": area,
                    "points": points,
                }
            )

        return masks

    def compare_images(
        self,
        sam2_dir: str,
        sam3_dir: str,
        images_dir: str,
        image_names: Optional[List[str]] = None,
    ) -> List[AnnotationComparison]:
        """
        Compare annotations between SAM2 and SAM3 for multiple images.

        Args:
            sam2_dir: Directory with SAM2 labels
            sam3_dir: Directory with SAM3 labels
            images_dir: Directory with images
            image_names: Optional list of image names to compare

        Returns:
            List of comparison results
        """
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

        if image_names is None:
            image_names = [
                f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)
            ]

        results = []

        for img_name in tqdm(image_names, desc="Comparing annotations"):
            img_path = os.path.join(images_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            h, w = img.shape[:2]

            base_name = os.path.splitext(img_name)[0]

            sam2_path = os.path.join(sam2_dir, f"{base_name}.txt")
            sam3_path = os.path.join(sam3_dir, f"{base_name}.txt")

            sam2_masks = self.load_yolo_masks(sam2_path, w, h)
            sam3_masks = self.load_yolo_masks(sam3_path, w, h)

            iou_scores = []
            class_matches = defaultdict(int)
            unmatched_sam2 = []
            unmatched_sam3 = []

            for sam2_mask in sam2_masks:
                best_iou = 0.0
                best_idx = -1

                for idx, sam3_mask in enumerate(sam3_masks):
                    iou = self.calculate_iou(sam2_mask["mask"], sam3_mask["mask"])

                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_iou >= self.min_iou_threshold:
                    iou_scores.append(best_iou)

                    sam2_class = sam2_mask["class_id"]
                    sam3_class = sam3_masks[best_idx]["class_id"]

                    if sam2_class == sam3_class:
                        class_matches["match"] += 1
                    else:
                        class_matches["mismatch"] += 1
                else:
                    unmatched_sam2.append(base_name)

            for idx, sam3_mask in enumerate(sam3_masks):
                is_matched = False

                for sam2_mask in sam2_masks:
                    iou = self.calculate_iou(sam3_mask["mask"], sam2_mask["mask"])

                    if iou >= self.min_iou_threshold:
                        is_matched = True
                        break

                if not is_matched:
                    unmatched_sam3.append(base_name)

            results.append(
                AnnotationComparison(
                    sam2_count=len(sam2_masks),
                    sam3_count=len(sam3_masks),
                    iou_scores=iou_scores,
                    class_matches=dict(class_matches),
                    unmatched_sam2=unmatched_sam2,
                    unmatched_sam3=unmatched_sam3,
                )
            )

        return results


class StatisticalAnalyzer:
    """Generate statistical analysis of annotations."""

    def __init__(self, classes: Optional[List[str]] = None):
        """
        Initialize analyzer.

        Args:
            classes: List of class names
        """
        self.classes = classes or [
            "overripe_fruitlet",
            "ripe_fruitlet",
            "unripe_fruitlet",
            "branch",
            "sky",
            "background",
        ]

    def analyze_label_directory(
        self,
        label_dir: str,
        images_dir: str,
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze label directory statistics.

        Args:
            label_dir: Directory with YOLO label files
            images_dir: Directory with images
            sample_size: Optional sample size for analysis

        Returns:
            Dictionary with statistics
        """
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        image_names = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]

        if sample_size is not None:
            np.random.seed(42)
            if sample_size < len(image_names):
                image_names = list(np.random.choice(image_names, sample_size, replace=False))

        class_counts = defaultdict(int)
        areas_by_class = defaultdict(list)
        total_objects = 0

        for img_name in tqdm(image_names, desc="Analyzing labels"):
            img_path = os.path.join(images_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            h, w = img.shape[:2]

            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(label_dir, f"{base_name}.txt")

            if not os.path.exists(label_path):
                continue

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
                        px = int(coords[i] * w)
                        py = int(coords[i + 1] * h)
                        points.append([px, py])

                if len(points) < 3:
                    continue

                cnt = np.array(points, dtype=np.int32)
                area = cv2.contourArea(cnt)

                if area < 50:
                    continue

                class_name = self.classes[class_id] if class_id < len(self.classes) else "unknown"

                class_counts[class_name] += 1
                areas_by_class[class_name].append(area)
                total_objects += 1

        stats = {
            "total_images": len(image_names),
            "total_objects": total_objects,
            "avg_objects_per_image": total_objects / len(image_names)
            if len(image_names) > 0
            else 0,
            "class_counts": dict(class_counts),
            "class_statistics": {},
        }

        for class_name, areas in areas_by_class.items():
            if not areas:
                continue

            areas = np.array(areas)

            stats["class_statistics"][class_name] = ClassStatistics(
                class_name=class_name,
                count=int(len(areas)),
                avg_area=float(np.mean(areas)),
                min_area=float(np.min(areas)),
                max_area=float(np.max(areas)),
                std_area=float(np.std(areas)),
            )

        return stats

    def generate_comparison_report(
        self,
        sam2_stats: Dict[str, Any],
        sam3_stats: Dict[str, Any],
        output_path: str,
    ) -> None:
        """
        Generate comparison report between SAM2 and SAM3 statistics.

        Args:
            sam2_stats: SAM2 statistics dictionary
            sam3_stats: SAM3 statistics dictionary
            output_path: Path to save the report
        """
        report = {
            "sam2": sam2_stats,
            "sam3": sam3_stats,
            "comparison": {},
        }

        sam2_counts = sam2_stats.get("class_counts", {})
        sam3_counts = sam3_stats.get("class_counts", {})

        all_classes = set(sam2_counts.keys()) | set(sam3_counts.keys())

        for class_name in all_classes:
            sam2_count = sam2_counts.get(class_name, 0)
            sam3_count = sam3_counts.get(class_name, 0)

            report["comparison"][class_name] = {
                "sam2_count": sam2_count,
                "sam3_count": sam3_count,
                "difference": sam3_count - sam2_count,
                "percent_change": ((sam3_count - sam2_count) / sam2_count * 100)
                if sam2_count > 0
                else 0,
            }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    def print_statistics(
        self,
        stats: Dict[str, Any],
        label: str = "Statistics",
    ) -> None:
        """
        Print statistics to console.

        Args:
            stats: Statistics dictionary
            label: Label for the statistics
        """
        print(f"\n{'=' * 50}")
        print(f"{label}")
        print(f"{'=' * 50}")

        print(f"Total images: {stats.get('total_images', 0)}")
        print(f"Total objects: {stats.get('total_objects', 0)}")
        print(f"Avg objects per image: {stats.get('avg_objects_per_image', 0):.2f}")

        print("\nClass Counts:")
        for class_name, count in stats.get("class_counts", {}).items():
            print(f"  {class_name}: {count}")

        print("\nClass Statistics:")
        for class_name, class_stat in stats.get("class_statistics", {}).items():
            print(f"  {class_name}:")
            print(f"    Count: {class_stat.count}")
            print(f"    Avg Area: {class_stat.avg_area:.2f}")
            print(f"    Min Area: {class_stat.min_area:.2f}")
            print(f"    Max Area: {class_stat.max_area:.2f}")
            print(f"    Std Area: {class_stat.std_area:.2f}")


def compare_directories(
    sam2_dir: str,
    sam3_dir: str,
    images_dir: str,
    output_path: str,
    sample_size: Optional[int] = None,
) -> None:
    """
    Compare SAM2 and SAM3 directories and generate report.

    Args:
        sam2_dir: Directory with SAM2 labels
        sam3_dir: Directory with SAM3 labels
        images_dir: Directory with images
        output_path: Path to save the comparison report
        sample_size: Optional sample size
    """
    analyzer = StatisticalAnalyzer()

    sam2_stats = analyzer.analyze_label_directory(sam2_dir, images_dir, sample_size)
    sam3_stats = analyzer.analyze_label_directory(sam3_dir, images_dir, sample_size)

    analyzer.generate_comparison_report(sam2_stats, sam3_stats, output_path)

    analyzer.print_statistics(sam2_stats, "SAM2 Statistics")
    analyzer.print_statistics(sam3_stats, "SAM3 Statistics")

    print(f"\nComparison report saved to {output_path}")
