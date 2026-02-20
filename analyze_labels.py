import cv2
import numpy as np
import os
import glob

image_dir = 'dataset/augmented_dataset'
label_dir = 'dataset/sam_annotations/labels_seg'

images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))[:4]

for img_path in images:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(label_dir, base_name + '.txt')
    
    if not os.path.exists(txt_path):
        print(f"Missing label for {base_name}")
        continue
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    print(f"--- {base_name} ---")
    print(f"Number of labels: {len(lines)}")
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    hsv_means = []
    rgb_means = []
    
    for line in lines:
        parts = list(map(float, line.strip().split()))
        coords = parts[1:]
        points = []
        for i in range(0, len(coords), 2):
            if i+1 < len(coords):
                x = int(coords[i] * w)
                y = int(coords[i+1] * h)
                points.append([x, y])
                
        if len(points) < 3:
            continue
            
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        mean_bgr = cv2.mean(img, mask=mask)[:3]
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]
        
        rgb_means.append((mean_bgr[2], mean_bgr[1], mean_bgr[0])) # RGB
        hsv_means.append(mean_hsv)
        
    if hsv_means:
        hsv_means = np.array(hsv_means)
        rgb_means = np.array(rgb_means)
        print("HSV Min:", np.min(hsv_means, axis=0))
        print("HSV Max:", np.max(hsv_means, axis=0))
        print("HSV Mean:", np.mean(hsv_means, axis=0))
        print("RGB Min:", np.min(rgb_means, axis=0))
        print("RGB Max:", np.max(rgb_means, axis=0))
        print("RGB Mean:", np.mean(rgb_means, axis=0))
        
