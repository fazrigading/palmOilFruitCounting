"""
SAM2.1-based automatic annotation for palm oil fruit detection.

This module provides automatic segmentation using Meta's Segment Anything Model 2.1 (SAM2.1)
with custom filtering for palm oil fruitlets.
"""

import os
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


SAM2_CONFIGS = {
    "tiny": (
        "configs/sam2.1_hiera_t.yaml",
        "checkpoints/sam2.1_hiera_tiny.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    ),
    "small": (
        "configs/sam2.1_hiera_s.yaml",
        "checkpoints/sam2.1_hiera_small.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    ),
    "base_plus": (
        "configs/sam2.1_hiera_b+.yaml",
        "checkpoints/sam2.1_hiera_base_plus.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    ),
    "large": (
        "configs/sam2.1_hiera_l.yaml",
        "checkpoints/sam2.1_hiera_large.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    ),
}


def save_yolo_bbox(masks: List[Dict[str, Any]], img_w: int, img_h: int, output_path: str) -> None:
    """Saves masks as YOLO detection format (class_id center_x center_y width height)."""
    with open(output_path, "w") as f:
        for mask_data in masks:
            x, y, w, h = mask_data["bbox"]

            center_x = (x + w / 2) / img_w
            center_y = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")


def save_yolo_segmentation(
    masks: List[Dict[str, Any]], img_w: int, img_h: int, output_path: str
) -> None:
    """Saves masks as YOLO segmentation format (class_id x1 y1 x2 y2 ...)."""
    with open(output_path, "w") as f:
        for mask_data in masks:
            mask = mask_data["segmentation"].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) < 3:
                    continue

                points = approx.flatten()
                normalized_points = []
                for i in range(0, len(points), 2):
                    normalized_points.append(f"{points[i] / img_w:.6f}")
                    normalized_points.append(f"{points[i + 1] / img_h:.6f}")

                f.write(f"0 {' '.join(normalized_points)}\n")


def filter_fruitlet_masks(masks: List[Dict[str, Any]], image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Filters out background/irrelevant SAM masks (leaves, sky, branches)
    to keep specifically black, maroon, red, yellow-orangeish palm oil fruitlets.

    Args:
        masks: List of SAM mask dictionaries
        image: RGB image array

    Returns:
        Filtered list of masks
    """
    filtered_masks = []

    for mask_data in masks:
        mask = mask_data["segmentation"].astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = contours[0]
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        aspect_ratio = float(w) / h
        if aspect_ratio < 0.3 or aspect_ratio > 3.3:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # Slightly relaxed circularity to account for the fibrous spikes/calyxes
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.3:
            continue

        mean_color = cv2.mean(image, mask=mask)[:3]
        R, G, B = mean_color

        # Calculate average intensity to detect black/dark fruitlets
        intensity = (R + G + B) / 3.0

        # If the object is dark, we bypass the strict leaf/sky color checks
        # because low RGB values are highly susceptible to noise and reflection ratios.
        is_dark_fruitlet = intensity < 75

        if not is_dark_fruitlet:
            # Filter out leaves/weeds (increased multiplier slightly for safety)
            if G > R * 1.15 and G > B:
                continue

            # Filter out sky/blue tarps
            if B > R * 1.15 and B > G:
                continue

            # Filter out bright white background/clouds
            if R > 200 and G > 200 and B > 200:
                continue

        filtered_masks.append(mask_data)

    return filtered_masks


def process_images(
    image_dir: str,
    output_dir: str,
    model_type: str = "tiny",
    config: Optional[str] = None,
    checkpoint: Optional[str] = None,
    device: str = "cuda",
) -> None:
    """
    Process images through SAM2.1 for automatic annotation.

    Args:
        image_dir: Path to directory containing images
        output_dir: Path to output directory for labels
        model_type: SAM2.1 model type (tiny, small, base_plus, large)
        config: Path to custom SAM2.1 config file
        checkpoint: Path to SAM2.1 checkpoint file
        device: Device to run on (cuda or cpu)
    """
    if not SAM2_AVAILABLE:
        raise ImportError("SAM2.1 is not installed. Install with: `git clone https://github.com/facebookresearch/sam2.git --depth 1``")

    if model_type not in SAM2_CONFIGS:
        print(
            f"Error: Model type '{model_type}' not supported. Choose from {list(SAM2_CONFIGS.keys())}"
        )
        return

    if config is None:
        config_file, default_ckpt, url = SAM2_CONFIGS[model_type]
        if checkpoint is None:
            checkpoint = default_ckpt
            print(f"Using default checkpoint: {checkpoint}")

        if not os.path.exists(checkpoint):
            print(f"Downloading {model_type} checkpoint to {checkpoint}...")
            torch.hub.download_url_to_file(url, checkpoint)
    else:
        config_file = config

    if device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print(f"Loading SAM2.1 model ({model_type}) from {checkpoint} on {device}...")
    sam2 = build_sam2(config_file, ckpt_path=None, device=device)

    if checkpoint:
        state_dict = torch.load(checkpoint, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        sam2.load_state_dict(state_dict, strict=False)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    bbox_dir = os.path.join(output_dir, "labels_bbox")
    seg_dir = os.path.join(output_dir, "labels_seg")
    os.makedirs(bbox_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]

    print(f"Found {len(images)} images. Starting automatic annotation...")

    for img_name in tqdm(images):
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        masks = mask_generator.generate(image)
        masks = filter_fruitlet_masks(masks, image)

        base_name = os.path.splitext(img_name)[0]

        save_yolo_bbox(masks, w, h, os.path.join(bbox_dir, f"{base_name}.txt"))
        save_yolo_segmentation(masks, w, h, os.path.join(seg_dir, f"{base_name}.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2.1 Automatic Annotator for Palm Oil Fruit")
    parser.add_argument("--input", type=str, default="dataset", help="Path to images directory")
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/sam_annotations",
        help="Path to output directory",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="tiny",
        help="SAM2.1 model type (tiny, small, base_plus, large)",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to SAM2.1 config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SAM2.1 checkpoint")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    process_images(
        image_dir=args.input,
        output_dir=args.output,
        model_type=args.model_type,
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
    )
