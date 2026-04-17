"""
SAM3-based automatic annotation for palm oil fruit detection.

This module provides automatic segmentation using Meta's Segment Anything Model 3 (SAM3)
via Roboflow Inference package with text-based prompting.
"""

import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from typing import Any, Optional
from dataclasses import dataclass

try:
    from inference.models.sam3 import SegmentAnything3 as SAM3Model
    from inference.core.entities.requests.sam3 import Sam3Prompt
    import supervision as sv

    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False


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

CLASS_ID_MAP = {
    "overripe_fruitlet": 0,
    "ripe_fruitlet": 1,
    "unripe_fruitlet": 2,
    "branch": 3,
    "sky": 4,
    "background": 5,
}


@dataclass
class SAM3Config:
    model_id: str = "sam3/sam3_final"
    confidence: float = 0.5
    device: str = "cuda"


def filter_by_color_and_size(
    masks: list[dict[str, Any]],
    image: np.ndarray,
    min_area: int = 100,
    max_area: int = 50000,
    min_aspect_ratio: float = 0.3,
    max_aspect_ratio: float = 3.3,
) -> list[dict[str, Any]]:
    """Filter masks based on color, size, and aspect ratio for palm oil fruitlets."""
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
    masks: list[dict[str, Any]],
    img_w: int,
    img_h: int,
    output_path: str,
    class_id: int = 0,
) -> None:
    """Save masks as YOLO detection format (class_id center_x center_y width height)."""
    with open(output_path, "w") as f:
        for mask_data in masks:
            x, y, w, h = mask_data["bbox"]

            center_x = (x + w / 2) / img_w
            center_y = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")


def save_yolo_segmentation(
    masks: list[dict[str, Any]],
    img_w: int,
    img_h: int,
    output_path: str,
    class_id: int = 0,
    epsilon_factor: float = 0.002,
) -> None:
    """Save masks as YOLO segmentation format (class_id x1 y1 x2 y2 ...)."""
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
    """Save annotations in COCO segmentation format."""
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

    with open(output_path, "w") as f:
        json.dump(coco_dict, f, indent=2)


def parse_inference_results(result, prompts: list[str]) -> list[dict[str, Any]]:
    """Parse inference results into mask dictionaries."""
    detections = []
    prompt_index_map = {}

    for prompt_idx, prompt in enumerate(prompts):
        prompt_index_map[prompt_idx] = prompt

    for item in result.prompt_results:
        prompt_text = prompt_index_map.get(item.prompt_index, "")
        preds = item.predictions

        if len(preds) == 0:
            continue

        all_polygons_coords = []
        all_confidences = []

        for p in preds:
            for polygon_coords in p.masks:
                all_polygons_coords.append(polygon_coords)
                all_confidences.append(p.confidence)

        for poly_coords, conf in zip(all_polygons_coords, all_confidences):
            polygon_np = np.array(poly_coords, dtype=np.int32)

            contours, _ = cv2.findContours(
                sv.polygon_to_mask(polygon_np).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
            else:
                x, y, w, h = 0, 0, 0, 0

            detections.append(
                {
                    "segmentation": sv.polygon_to_mask(polygon_np),
                    "bbox": [x, y, w, h],
                    "confidence": conf,
                    "prompt_index": item.prompt_index,
                    "prompt_text": prompt_text,
                }
            )

    return detections


class SAM3Annotator:
    """SAM3-based automatic annotator for palm oil fruit detection."""

    def __init__(
        self,
        ontology: Optional[dict[str, str]] = None,
        config: Optional[SAM3Config] = None,
    ):
        """Initialize SAM3 annotator."""
        if not INFERENCE_AVAILABLE:
            raise ImportError(
                "inference package is not installed. Install with: pip install inference-gpu[sam3]"
            )

        self.config = config or SAM3Config()
        self.ontology = ontology or ONTOLOGY_CLASSES

        self.prompts = list(self.ontology.keys())
        self.class_names = list(self.ontology.values())

        self.model = SAM3Model(model_id=self.config.model_id)

    def predict(self, image_path: str):
        """Run inference on a single image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        prompts = [Sam3Prompt(type="text", text=prompt) for prompt in self.prompts]

        result = self.model.segment_image(
            image,
            prompts=prompts,
            format="polygon",
        )

        detections = parse_inference_results(result, self.prompts)

        return detections, image

    def label(
        self,
        input_dir: str,
        output_dir: str,
        output_format: str = "both",
    ) -> None:
        """Label a directory of images."""
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

            try:
                detections, image = self.predict(img_path)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue

            h, w = image.shape[:2]
            base_name = os.path.splitext(img_name)[0]

            filtered = filter_by_color_and_size(detections, image)

            class_ids = []
            for det in filtered:
                class_name = self.ontology.get(det["prompt_text"], "unknown")
                class_id = CLASS_ID_MAP.get(class_name, -1)
                class_ids.append(class_id)

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

                for det_idx, det in enumerate(filtered):
                    mask = det["segmentation"].astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < 50:
                            continue

                        segmentation = [cnt.flatten().tolist()]
                        x, y, bw, bh = cv2.boundingRect(cnt)

                        cat_id = class_ids[det_idx] if det_idx < len(class_ids) else 0

                        all_annotations.append(
                            {
                                "id": len(all_annotations) + 1,
                                "image_id": idx + 1,
                                "category_id": cat_id,
                                "segmentation": segmentation,
                                "area": float(area),
                                "bbox": [float(x), float(y), float(bw), float(bh)],
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
    """Process images through SAM3 for automatic annotation."""
    if not INFERENCE_AVAILABLE:
        raise ImportError(
            "inference package is not installed. Install with: pip install inference-gpu[sam3]"
        )

    ontology = ONTOLOGY_CLASSES if use_full_ontology else ONTOLOGY_MINIMAL

    config = SAM3Config(device=device)

    annotator = SAM3Annotator(ontology=ontology, config=config)

    annotator.label(
        input_dir=input_dir,
        output_dir=output_dir,
        output_format=output_format,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Automatic Annotator for Palm Oil Fruit")
    parser.add_argument("--input", type=str, required=True, help="Path to images directory")
    parser.add_argument(
        "--output", type=str, default="dataset/sam3_annotations", help="Path to output directory"
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
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for GPU processing")

    args = parser.parse_args()

    process_images(
        input_dir=args.input,
        output_dir=args.output,
        output_format=args.output_format,
        use_full_ontology=(args.classes == "full"),
        device=args.device,
        batch_size=args.batch_size,
    )
