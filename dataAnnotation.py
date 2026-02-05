import cv2
import numpy as np
import os
import glob

def get_yolo_segmentation_format(contours, img_w, img_h):
    yolo_labels = []
    for cnt in contours:
        # Simplify contour to reduce number of points
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) < 3:
            continue
            
        points = approx.flatten()
        normalized_points = []
        for i in range(len(points)):
            if i % 2 == 0:  # x coordinate
                normalized_points.append(float(points[i]) / img_w)
            else:  # y coordinate
                normalized_points.append(float(points[i]) / img_h)
        
        # YOLO format: class_id x1 y1 x2 y2 ...
        label_str = "0 " + " ".join([f"{p:.6f}" for p in normalized_points])
        yolo_labels.append(label_str)
    return yolo_labels

def segment_fruits(image_path, output_dir, visualize_dir=None):
    img = cv2.imread(image_path)
    if img is None:
        return
    
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Range for ripe/orange palm oil fruits
    lower_orange1 = np.array([0, 70, 50])
    upper_orange1 = np.array([25, 255, 255])
    
    lower_orange2 = np.array([160, 70, 50])
    upper_orange2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
    mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (ignore small spots)
    min_area = 500
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Save YOLO labels
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(output_dir, f"{base_name}.txt")
    
    yolo_labels = get_yolo_segmentation_format(filtered_contours, w, h)
    
    with open(label_path, 'w') as f:
        for label in yolo_labels:
            f.write(label + "\n")
            
    # Visualization
    if visualize_dir:
        vis_img = img.copy()
        cv2.drawContours(vis_img, filtered_contours, -1, (0, 255, 0), 2)
        vis_path = os.path.join(visualize_dir, f"{base_name}_vis.jpg")
        cv2.imwrite(vis_path, vis_img)

def main():
    dataset_dir = "dataset"
    labels_dir = os.path.join(dataset_dir, "labels")
    vis_dir = os.path.join(dataset_dir, "visualized")
    
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
        # Also check subdirectories just in case
        image_paths.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
    
    # Filter out files in labels or visualized directories
    image_paths = [p for p in image_paths if "labels" not in p and "visualized" not in p]
    
    # Remove duplicates
    image_paths = sorted(list(set(image_paths)))
    
    print(f"Found {len(image_paths)} images. Starting annotation...")
    
    for i, img_path in enumerate(image_paths):
        if i % 50 == 0:
            print(f"Processing image {i}/{len(image_paths)}: {img_path}")
        segment_fruits(img_path, labels_dir, vis_dir)
    
    print("Annotation complete. Labels saved in 'dataset/labels' and visualizations in 'dataset/visualized'.")

if __name__ == "__main__":
    main()