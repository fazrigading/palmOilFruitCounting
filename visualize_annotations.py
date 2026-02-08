import argparse
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

def draw_yolo_bbox(image, label_path):
    h, w = image.shape[:2]
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 5: continue
        class_id = int(parts[0])
        cx, cy, bw, bh = parts[1], parts[2], parts[3], parts[4]
        
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label text
        label_text = f"Class {class_id}"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image

def draw_yolo_seg(image, label_path):
    h, w = image.shape[:2]
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 3: continue 
        
        class_id = int(parts[0])
        coords = parts[1:]
        
        points = []
        for i in range(0, len(coords), 2):
            px = int(coords[i] * w)
            py = int(coords[i+1] * h)
            points.append([px, py])
            
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        
        # Draw polygon outline
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        
        # Fill with semi-transparent color
        overlay = image.copy()
        cv2.fillPoly(overlay, [points], (0, 0, 255))
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        
    return image

def visualize(image_dir, label_dir, output_dir, viz_type):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get image files
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    print(f"Found {len(image_files)} images in {image_dir}")
    print(f"Reading labels from {label_dir}")
    
    count = 0
    for img_path in tqdm(image_files, desc="Visualizing"):
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        label_path = os.path.join(label_dir, name_no_ext + ".txt")
        
        if not os.path.exists(label_path):
            continue
            
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Determine type if auto
        current_type = viz_type
        if current_type == 'auto':
            with open(label_path, 'r') as f:
                line = f.readline()
                if not line:
                    continue
                parts = line.strip().split()
                # YOLO bbox has 5 parts: class x y w h
                # YOLO seg has > 5 parts usually: class x1 y1 x2 y2 ...
                if len(parts) == 5:
                    current_type = 'bbox'
                else:
                    current_type = 'seg'
        
        if current_type == 'bbox':
            image = draw_yolo_bbox(image, label_path)
        elif current_type == 'seg':
            image = draw_yolo_seg(image, label_path)
            
        cv2.imwrite(os.path.join(output_dir, basename), image)
        count += 1
        
    print(f"Visualized {count} images. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO annotations on images.")
    parser.add_argument("--images", type=str, required=True, help="Path to directory containing images")
    parser.add_argument("--labels", type=str, required=True, help="Path to directory containing YOLO txt labels")
    parser.add_argument("--output", type=str, default="dataset/visualized", help="Path to save visualized images")
    parser.add_argument("--type", type=str, choices=['bbox', 'seg', 'auto'], default='auto', help="Type of annotation (bbox, seg, or auto-detect)")
    
    args = parser.parse_args()
    
    visualize(args.images, args.labels, args.output, args.type)