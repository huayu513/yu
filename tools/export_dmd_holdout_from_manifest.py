import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a DMD-only holdout set from the mixed dataset manifest for target-domain evaluation."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_mix_ft_clean4/dmd_manifest.csv"),
        help="Path to the DMD manifest created when building the mixed fine-tuning dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/dmd_holdout_eval"),
        help="Directory where the DMD-only holdout images and metadata will be exported.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which manifest split to export. For evaluation use 'test'.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Whether to hardlink or copy images into the holdout directory.",
    )
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, link_mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if link_mode == "hardlink":
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
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = Counter()
    method_counts = Counter()

    with args.manifest.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != args.split:
                continue

            src_path = Path(row["image_path"])
            original_label = row["original_label"]
            output_path = args.output_dir / original_label / src_path.name
            method = link_or_copy(src_path, output_path, args.link_mode)
            method_counts[method] += 1
            counts[original_label] += 1

            rows.append(
                {
                    "image_path": str(output_path.resolve()),
                    "true_label": original_label,
                    "mapped_class": row["mapped_class"],
                    "frame_idx": row["frame_idx"],
                    "video_name": src_path.name,
                    "json_name": "",
                    "stream": "body",
                    "label_prefix": "driver_actions",
                    "frame_shift": "",
                    "source_manifest": str(args.manifest),
                    "source_split": row["split"],
                }
            )

    metadata_path = args.output_dir / "samples_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "true_label",
                "mapped_class",
                "frame_idx",
                "video_name",
                "json_name",
                "stream",
                "label_prefix",
                "frame_shift",
                "source_manifest",
                "source_split",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"manifest: {args.manifest}")
    print(f"split: {args.split}")
    print(f"output_dir: {args.output_dir}")
    print(f"metadata: {metadata_path}")
    print(f"link_mode: {args.link_mode}")
    print(f"images: {len(rows)}")
    print()
    print("Counts by original DMD label")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")
    print()
    print("Link/copy stats")
    for key, count in sorted(method_counts.items()):
        print(f"{key}: {count}")


if __name__ == "__main__":
    main()
