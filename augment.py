import os
import cv2
import albumentations as A
from tqdm import tqdm
import glob

def augment_images(input_dir, output_dir, target_size=640):
    """
    Augments images by resizing them to target_size while maintaining 
    aspect ratio and padding with white background to fit the target_size.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Define the transformation pipeline
    transform = A.Compose([
        A.LongestMaxSize(max_size=target_size),
        A.PadIfNeeded(
            min_height=target_size,
            min_width=target_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    ])

    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        # Also handle uppercase extensions
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    print(f"Found {len(image_paths)} images in {input_dir}")

    for img_path in tqdm(image_paths, desc="Augmenting images"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # OpenCV reads in BGR format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transformation
        augmented = transform(image=img_rgb)["image"]
        
        # Convert back to BGR for saving
        augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, augmented_bgr)

if __name__ == "__main__":
    input_folder = "dataset/cropped/"
    output_folder = "dataset/augmented/"
    augment_images(input_folder, output_folder)
