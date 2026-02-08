import os
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Instructions for Cloud Notebook (Colab/Kaggle):
# !pip install git+https://github.com/facebookresearch/sam2.git
# !pip install opencv-python pycocotools matplotlib onnxruntime onnx
# !wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

def save_yolo_bbox(masks, img_w, img_h, output_path):
    """Saves masks as YOLO detection format (class_id center_x center_y width height)."""
    with open(output_path, 'w') as f:
        for mask_data in masks:
            # mask_data['bbox'] is [x, y, w, h] in pixels
            x, y, w, h = mask_data['bbox']
            
            # Convert to YOLO format (normalized center_x, center_y, width, height)
            center_x = (x + w / 2) / img_w
            center_y = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            
            # We assume class_id 0 for 'Fruit'
            f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

def save_yolo_segmentation(masks, img_w, img_h, output_path):
    """Saves masks as YOLO segmentation format (class_id x1 y1 x2 y2 ...)."""
    with open(output_path, 'w') as f:
        for mask_data in masks:
            # SAM returns a binary mask in 'segmentation' key
            mask = mask_data['segmentation'].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                # Simplify contour to reduce number of points for YOLO format
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(approx) < 3:
                    continue
                    
                points = approx.flatten()
                normalized_points = []
                for i in range(0, len(points), 2):
                    normalized_points.append(f"{points[i] / img_w:.6f}")
                    normalized_points.append(f"{points[i+1] / img_h:.6f}")
                
                f.write(f"0 {' '.join(normalized_points)}\n")

def process_images(image_dir, output_dir, config, model_type="tiny", checkpoint=None, device="cuda"):
    # SAM2 Model configurations: (config_file, default_checkpoint, download_url)
    sam2_configs = {
        "tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"),
        "small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"),
        "base_plus": ("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"),
        "large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"),
    }

    if model_type not in sam2_configs:
        print(f"Error: Model type '{model_type}' not supported. Choose from {list(sam2_configs.keys())}")
        return
    if config:
        config_file = config
    else:
        config_file, default_ckpt, url = sam2_configs[model_type]
    
    if checkpoint is None:
        checkpoint = default_ckpt

    # Check and download checkpoint if missing
    if not os.path.exists(checkpoint):
        print(f"Downloading {model_type} checkpoint to {checkpoint}...")
        torch.hub.download_url_to_file(url, checkpoint)

    # Initialize SAM2
    print(f"Loading SAM2 model ({model_type}) from {checkpoint} on {device}...")
    sam = build_sam2(config_file, checkpoint, device=device,apply_postprocessing=False)
    
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Filter out very small segments
    )

    # Create output directories
    bbox_dir = os.path.join(output_dir, "labels_bbox")
    seg_dir = os.path.join(output_dir, "labels_seg")
    os.makedirs(bbox_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
    
    print(f"Found {len(images)} images. Starting automatic annotation...")

    for img_name in tqdm(images):
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Generate masks
        masks = mask_generator.generate(image)

        # Save annotations
        base_name = os.path.splitext(img_name)[0]
        
        # Save Bounding Boxes
        save_yolo_bbox(masks, w, h, os.path.join(bbox_dir, f"{base_name}.txt"))
        
        # Save Segmentation Masks
        save_yolo_segmentation(masks, w, h, os.path.join(seg_dir, f"{base_name}.txt"))

if __name__ == "__main__":
    # In a notebook, you can just call process_images(...) directly.
    # Here we provide a CLI interface for flexibility.
    parser = argparse.ArgumentParser(description="SAM Automatic Annotator for Palm Oil Fruit")
    parser.add_argument("--input", type=str, default="dataset", help="Path to images directory")
    parser.add_argument("--output", type=str, default="dataset/sam_annotations", help="Path to output directory")
    parset.add_argument("--config", type=str, default="sam2_hiera_t.yaml", help="Path to SAM2 config file (default: tiny)")
    parser.add_argument("--checkpoint", type=str, default="sam2_hiera_tiny.pt", help="Path to SAM2 checkpoint (default: tiny)")
    parser.add_argument("--model-type", type=str, default="tiny", help="SAM2 model type (tiny, small, base_plus, large)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    
    args = parser.parse_args()
    
    process_images(args.input, args.output, args.config, args.model_type, args.checkpoint, args.device)
