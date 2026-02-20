import cv2
import numpy as np
import os
import glob

image_dir = 'dataset/augmented_dataset'
label_dir = 'dataset/sam_annotations/labels_seg'

images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))[:10]

def is_fruitlet_contour(contour, img):
    area = cv2.contourArea(contour)
    # Area filter
    if area < 50:
        return False
        
    # Shape filter
    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return False
    aspect_ratio = float(w)/h
    if aspect_ratio < 0.3 or aspect_ratio > 3.3:
        return False
        
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity < 0.4:
        return False
        
    # Color filter
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    mean_color = cv2.mean(img, mask=mask)[:3] # BGR
    B, G, R = mean_color
    
    # Reject mostly green (leaves)
    if G > R * 1.1 and G > B:
        return False
        
    # Reject sky (blue/white)
    if B > R * 1.1 and B > G:
        return False
    if R > 200 and G > 200 and B > 200: # White sky
        return False
        
    return True

for i, img_path in enumerate(images):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(label_dir, base_name + '.txt')
    
    if not os.path.exists(txt_path):
        continue
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    kept = 0
    total = 0
    
    for line in lines:
        parts = list(map(float, line.strip().split()))
        coords = parts[1:]
        points = []
        for j in range(0, len(coords), 2):
            if j+1 < len(coords):
                px = int(coords[j] * w)
                py = int(coords[j+1] * h)
                points.append([px, py])
                
        if len(points) < 3:
            continue
            
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        total += 1
        
        if is_fruitlet_contour(contour, img):
            kept += 1
            
    # The first 4 are curated, others are not
    print(f"{base_name} ({'Curated' if i < 4 else 'Raw'}): Kept {kept} / Total {total} ({kept/total*100:.1f}%)")
