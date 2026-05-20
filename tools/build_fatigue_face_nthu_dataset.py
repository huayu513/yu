import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


CLASS_MAP = {
    "drowsy": "fatigue",
    "notdrowsy": "non_fatigue",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a clean fatigue/non_fatigue classification dataset from archive_1/train_data."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("archive_1/train_data"),
        help="Source directory containing drowsy and notdrowsy image folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_fatigue_face_nthu_v1"),
        help="Output directory for train/val/test classification splits.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Whether to hardlink or copy images into the new dataset.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Expected 0 < train_ratio, 0 <= val_ratio, and train_ratio + val_ratio < 1.")


def collect_images(class_dir: Path):
    return sorted([p for p in class_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def split_paths(paths, train_ratio, val_ratio):
    n = len(paths)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return {
        "train": paths[:train_end],
        "val": paths[train_end:val_end],
        "test": paths[val_end:],
    }


def ensure_dirs(root: Path):
    for split in ("train", "val", "test"):
        for cls in ("fatigue", "non_fatigue"):
            (root / split / cls).mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"
    shutil.copy2(src, dst)
    return "copy"


def main():
    args = parse_args()
    validate_args(args)

    random.seed(args.seed)
    ensure_dirs(args.output_dir)

    manifest_rows = []
    split_counts = defaultdict(Counter)
    method_counts = Counter()

    for source_class, target_class in CLASS_MAP.items():
        class_dir = args.source_dir / source_class
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing source class directory: {class_dir}")

        images = collect_images(class_dir)
        random.shuffle(images)
        split_map = split_paths(images, args.train_ratio, args.val_ratio)

        for split, paths in split_map.items():
            for src_path in paths:
                dst_path = args.output_dir / split / target_class / src_path.name
                method = link_or_copy(src_path, dst_path, args.link_mode)
                method_counts[method] += 1
                split_counts[split][target_class] += 1
                manifest_rows.append(
                    {
                        "split": split,
                        "source_class": source_class,
                        "target_class": target_class,
                        "source_path": str(src_path.resolve()),
                        "output_path": str(dst_path.resolve()),
                    }
                )

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "source_class", "target_class", "source_path", "output_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"source_dir: {args.source_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"manifest: {manifest_path}")
    print(f"seed: {args.seed}")
    print(f"link_mode: {args.link_mode}")
    print()
    print("Counts by split")
    for split in ("train", "val", "test"):
        total = sum(split_counts[split].values())
        print(f"{split}: {total}")
        for cls in ("fatigue", "non_fatigue"):
            print(f"  {cls}: {split_counts[split][cls]}")
    print()
    print("Link/copy stats")
    for key, count in sorted(method_counts.items()):
        print(f"{key}: {count}")


if __name__ == "__main__":
    main()
