#!/usr/bin/env python3
"""CLI entry point for annotation tools."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Palm Oil Fruit Counting - Annotation Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    sam_parser = subparsers.add_parser("sam", help="SAM2-based automatic annotation")
    sam_parser.add_argument(
        "--input", type=str, required=True, help="Path to images directory"
    )
    sam_parser.add_argument(
        "--output", type=str, default="dataset/sam_annotations", help="Output directory"
    )
    sam_parser.add_argument(
        "--model-type",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base_plus", "large"],
    )
    sam_parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda/cpu)"
    )

    hsv_parser = subparsers.add_parser("hsv", help="HSV-based automatic annotation")
    hsv_parser.add_argument(
        "--input", type=str, required=True, help="Path to images directory"
    )
    hsv_parser.add_argument("--output", type=str, help="Output labels directory")
    hsv_parser.add_argument(
        "--visualize", type=str, help="Visualization output directory"
    )

    args = parser.parse_args()

    if args.command == "sam":
        from palm_oil_counting.annotation import process_images

        process_images(
            image_dir=args.input,
            output_dir=args.output,
            model_type=args.model_type,
            device=args.device,
        )
    elif args.command == "hsv":
        from palm_oil_counting.annotation import process_directory

        process_directory(
            dataset_dir=args.input, labels_dir=args.output, vis_dir=args.visualize
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
