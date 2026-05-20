import argparse
from collections import defaultdict
from pathlib import Path

from convert_dmd_openlabel_to_yolo_cls import export_dataset_from_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch convert a DMD dataset folder to a YOLO classification dataset."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root folder containing DMD OpenLABEL json files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output dataset root. Result format: split/class_name/*.jpg",
    )
    parser.add_argument(
        "--stream",
        default="body",
        choices=["face", "body", "hands"],
        help="Which RGB camera stream to export.",
    )
    parser.add_argument(
        "--label-prefix",
        default="driver_actions",
        help="Only export action types that start with this prefix.",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=10,
        help="Save one frame every N labeled frames.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional cap for each class across each json. 0 means no cap.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality from 0 to 100.",
    )
    return parser.parse_args()


def find_annotation_files(dataset_root: Path):
    return sorted(dataset_root.rglob("*_ann_*.json"))


def main():
    args = parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1.")

    dataset_root = Path(args.dataset_root)
    annotation_files = find_annotation_files(dataset_root)
    if not annotation_files:
        raise FileNotFoundError(f"No *_ann_*.json found under: {dataset_root}")

    total_sampled = defaultdict(int)
    total_raw = defaultdict(int)
    total_ambiguous = 0

    print(f"Found {len(annotation_files)} annotation file(s)\n")
    for idx, json_path in enumerate(annotation_files, start=1):
        print(f"[{idx}/{len(annotation_files)}] converting {json_path}")
        summary = export_dataset_from_json(
            json_path=json_path,
            output_dir=args.output_dir,
            stream=args.stream,
            label_prefix=args.label_prefix,
            every_n_frames=args.every_n_frames,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            max_per_class=args.max_per_class,
            jpg_quality=args.jpg_quality,
        )
        total_ambiguous += summary["ambiguous_frames"]
        for label, count in summary["raw_counts"].items():
            total_raw[label] += count
        for key, count in summary["sampled_frames"].items():
            total_sampled[key] += count

    print("\nBatch summary")
    print(f"dataset_root: {dataset_root}")
    print(f"output_dir: {Path(args.output_dir)}")
    print(f"stream: {args.stream}")
    print(f"label_prefix: {args.label_prefix}")
    print(f"ambiguous frames skipped: {total_ambiguous}")

    print("\nRaw frame counts")
    for label, count in sorted(total_raw.items()):
        print(f"{label}: {count}")

    print("\nExported images")
    for (split, label), count in sorted(total_sampled.items()):
        print(f"{split}/{label}: {count}")


if __name__ == "__main__":
    main()
