"""
Image augmentation utilities for dataset preparation.
"""

import os
import cv2
import albumentations as A
from tqdm import tqdm
import glob
import argparse
from typing import Optional, List, Tuple


def augment_images(
    input_dir: str,
    output_dir: str,
    target_size: int = 640,
    maintain_aspect: bool = True,
    padding_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Augments images by resizing while maintaining aspect ratio and padding.

    Args:
        input_dir: Path to input directory containing images
        output_dir: Path to output directory for augmented images
        target_size: Target size for the longest dimension (default: 640)
        maintain_aspect: Whether to maintain aspect ratio (default: True)
        padding_color: RGB color for padding (default: white)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    if maintain_aspect:
        transform = A.Compose(
            [
                A.LongestMaxSize(max_size=target_size),
                A.PadIfNeeded(
                    min_height=target_size,
                    min_width=target_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=padding_color,
                ),
            ]
        )
    else:
        transform = A.Compose([A.Resize(height=target_size, width=target_size)])

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths: List[str] = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    print(f"Found {len(image_paths)} images in {input_dir}")

    for img_path in tqdm(image_paths, desc="Augmenting images"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = transform(image=img_rgb)["image"]

        augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, augmented_bgr)


def augment_with_variations(
    input_dir: str, output_dir: str, target_size: int = 640, num_variations: int = 3
) -> None:
    """
    Augments images with multiple variations (brightness, contrast, hue).

    Args:
        input_dir: Path to input directory containing images
        output_dir: Path to output directory for augmented images
        target_size: Target size for the longest dimension
        num_variations: Number of augmented variations per image
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=target_size),
            A.PadIfNeeded(
                min_height=target_size,
                min_width=target_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(255, 255, 255),
            ),
        ]
    )

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths: List[str] = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    print(
        f"Found {len(image_paths)} images. Generating {num_variations} variations each..."
    )

    for img_path in tqdm(image_paths, desc="Augmenting with variations"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(os.path.basename(img_path))[1]

        augmented_base = base_transform(image=img_rgb)["image"]
        save_path = os.path.join(output_dir, f"{base_name}{ext}")
        cv2.imwrite(save_path, cv2.cvtColor(augmented_base, cv2.COLOR_RGB2BGR))

        for i in range(num_variations):
            variation_transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=target_size),
                    A.PadIfNeeded(
                        min_height=target_size,
                        min_width=target_size,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=(255, 255, 255),
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.8
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=10,
                        p=0.5,
                    ),
                ]
            )

            augmented = variation_transform(image=img_rgb)["image"]
            save_path = os.path.join(output_dir, f"{base_name}_aug{i + 1}{ext}")
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment images for dataset preparation."
    )
    parser.add_argument(
        "--input", type=str, default="dataset/cropped/", help="Path to input directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/augmented/",
        help="Path to output directory",
    )
    parser.add_argument(
        "--size", type=int, default=640, help="Target image size (default: 640)"
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=0,
        help="Number of augmented variations per image (0 = no variations)",
    )
    parser.add_argument(
        "--no-aspect", action="store_true", help="Don't maintain aspect ratio"
    )

    args = parser.parse_args()

    if args.variations > 0:
        augment_with_variations(args.input, args.output, args.size, args.variations)
    else:
        augment_images(
            args.input, args.output, args.size, maintain_aspect=not args.no_aspect
        )
