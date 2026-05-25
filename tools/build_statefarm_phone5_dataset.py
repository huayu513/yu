import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

STATEFARM_MAPPING = {
    "c0": "safe_drive",
    "c1": "phone_use",
    "c2": "phone_use",
    "c3": "phone_use",
    "c4": "phone_use",
    "c5": "radio",
    "c6": "drinking",
    "c9": "talking_to_passenger",
}

FINAL_CLASSES = [
    "safe_drive",
    "phone_use",
    "radio",
    "drinking",
    "talking_to_passenger",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a pure State Farm 5-class distraction dataset.")
    parser.add_argument("--statefarm-dir", type=Path, default=Path("dataset_cls_subject"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_statefarm_phone5"))
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args()


def ensure_class_dirs(root: Path):
    for split in ("train", "val", "test"):
        for cls in FINAL_CLASSES:
            (root / split / cls).mkdir(parents=True, exist_ok=True)


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
    src_root = args.statefarm_dir
    dst_root = args.output_dir
    ensure_class_dirs(dst_root)

    stats = defaultdict(Counter)
    manifest_rows = []

    for split in ("train", "val", "test"):
        split_root = src_root / split
        for src_cls, dst_cls in STATEFARM_MAPPING.items():
            src_dir = split_root / src_cls
            if not src_dir.is_dir():
                continue

            for image_path in sorted(src_dir.iterdir()):
                if not image_path.is_file():
                    continue

                out_name = f"sf_{src_cls}_{image_path.name}"
                out_path = dst_root / split / dst_cls / out_name
                method = link_or_copy(image_path, out_path, args.link_mode)
                stats[split][dst_cls] += 1
                stats[f"{split}_method"][method] += 1
                manifest_rows.append(
                    {
                        "source": "statefarm",
                        "split": split,
                        "original_label": src_cls,
                        "mapped_class": dst_cls,
                        "image_path": str(image_path),
                        "output_path": str(out_path),
                    }
                )

    manifest_path = dst_root / "mix_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "split", "original_label", "mapped_class", "image_path", "output_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"statefarm_dir: {src_root}")
    print(f"output_dir: {dst_root}")
    print(f"link_mode: {args.link_mode}")
    print(f"manifest: {manifest_path}")
    print()
    print("Final dataset counts")
    for split in ("train", "val", "test"):
        split_total = sum(stats[split].values())
        print(f"{split}: {split_total}")
        for cls in FINAL_CLASSES:
            print(f"  {cls}: {stats[split][cls]}")


if __name__ == "__main__":
    main()
