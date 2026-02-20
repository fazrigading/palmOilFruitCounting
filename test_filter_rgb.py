import cv2
import numpy as np
import os
import glob

image_dir = 'dataset/augmented_dataset'
label_dir = 'dataset/sam_annotations/labels_seg'

images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))[:10]

def filter_fruitlet_masks(masks, image):
    filtered_masks = []
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    
    for mask_data in masks:
        # mask is boolean, so convert to uint8 for cv2
        mask = mask_data['segmentation'].astype(np.uint8) * 255
        
        area = mask_data['area']
        area_pct = (area / img_area) * 100
        if area_pct < 0.05 or area_pct > 60.0:
            continue
            
        x, y, w, h = mask_data['bbox']
        if w == 0 or h == 0:
            continue
        
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 1.5: # User default was 1.2, 1.5 is a safe margin
            continue
            
        w_pct = (w / img_w) * 100
        h_pct = (h / img_h) * 100
        if w_pct < 1.0 or h_pct < 1.0 or w_pct > 90.0 or h_pct > 90.0:
            continue
            
        # Color filtering (image is RGB)
        mean_color = cv2.mean(image, mask=mask)[:3]
        R, G, B = mean_color
        
        # Reject predominantly green or blue
        if G > R and G > B:
            continue
        if B > R and B > G:
            continue
        if R > 200 and G > 200 and B > 200:
            continue
            
        if R < G or R < B:
            if (R + G + B) > 150: 
                continue

        filtered_masks.append(mask_data)
        
    return filtered_masks

for i, img_path in enumerate(images):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(label_dir, base_name + '.txt')
    
    if not os.path.exists(txt_path):
        continue
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    # Read in BGR, convert to RGB as in sam_annotator.py
    img_bgr = cv2.imread(img_path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # create fake masks list from the txt
    fake_masks = []
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
        x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
        area_b = cv2.contourArea(contour)
        
        mask = np.zeros(img.shape[:2], dtype=bool)
        cv_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(cv_mask, [contour], -1, 255, -1)
        mask = cv_mask > 0
        
        fake_masks.append({
            'segmentation': mask,
            'area': area_b,
            'bbox': [x_b, y_b, w_b, h_b]
        })
        
    filtered = filter_fruitlet_masks(fake_masks, img)
    kept = len(filtered)
    total = len(fake_masks)
    
    print(f"{base_name} ({'Curated' if i < 4 else 'Raw'}): Kept {kept} / Total {total} ({kept/total*100:.1f}%)")
