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
    sam_parser.add_argument("--input", type=str, required=True, help="Path to images directory")
    sam_parser.add_argument(
        "--output", type=str, default="dataset/sam_annotations", help="Output directory"
    )
    sam_parser.add_argument(
        "--model-type",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base_plus", "large"],
    )
    sam_parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    hsv_parser = subparsers.add_parser("hsv", help="HSV-based automatic annotation")
    hsv_parser.add_argument("--input", type=str, required=True, help="Path to images directory")
    hsv_parser.add_argument("--output", type=str, help="Output labels directory")
    hsv_parser.add_argument("--visualize", type=str, help="Visualization output directory")

    sam3_parser = subparsers.add_parser("sam3", help="SAM3-based automatic annotation")
    sam3_parser.add_argument("--input", type=str, required=True, help="Path to images directory")
    sam3_parser.add_argument(
        "--output", type=str, default="dataset/sam3_annotations", help="Output directory"
    )
    sam3_parser.add_argument(
        "--output-format",
        type=str,
        default="both",
        choices=["yolo", "coco", "both"],
        help="Output format",
    )
    sam3_parser.add_argument(
        "--classes",
        type=str,
        default="full",
        choices=["full", "minimal"],
        help="Class selection strategy",
    )
    sam3_parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    sam3_parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for GPU processing"
    )

    analyze_parser = subparsers.add_parser("analyze", help="Analyze and compare annotations")
    analyze_parser.add_argument(
        "--sam2-dir", type=str, required=True, help="Directory with SAM2 labels"
    )
    analyze_parser.add_argument(
        "--sam3-dir", type=str, required=True, help="Directory with SAM3 labels"
    )
    analyze_parser.add_argument(
        "--images-dir", type=str, required=True, help="Directory with images"
    )
    analyze_parser.add_argument(
        "--output", type=str, default="dataset/analysis/comparison.json", help="Output report path"
    )
    analyze_parser.add_argument(
        "--sample-size", type=int, default=None, help="Sample size for analysis"
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

        process_directory(dataset_dir=args.input, labels_dir=args.output, vis_dir=args.visualize)
    elif args.command == "sam3":
        from palm_oil_counting.annotation.sam3_annotator import process_images as sam3_process

        sam3_process(
            input_dir=args.input,
            output_dir=args.output,
            output_format=args.output_format,
            use_full_ontology=(args.classes == "full"),
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "analyze":
        from palm_oil_counting.analysis.sam3_analysis import compare_directories

        compare_directories(
            sam2_dir=args.sam2_dir,
            sam3_dir=args.sam3_dir,
            images_dir=args.images_dir,
            output_path=args.output,
            sample_size=args.sample_size,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
