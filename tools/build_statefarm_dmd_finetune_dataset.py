import argparse
import csv
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_MAPPING = {
    "safe_drive": "c0",
    "radio": "c5",
    "drinking": "c6",
    "talking_to_passenger": "c9",
}


def parse_mapping(items):
    mapping = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid mapping '{item}', expected src=dst.")
        src, dst = item.split("=", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            raise ValueError(f"Invalid mapping '{item}', expected src=dst.")
        mapping[src] = dst
    return mapping


def ensure_class_dirs(root, classes):
    for split in ("train", "val", "test"):
        for cls in classes:
            (root / split / cls).mkdir(parents=True, exist_ok=True)


def link_or_copy(src, dst, link_mode):
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


def collect_statefarm_classes(dataset_dir):
    train_dir = dataset_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing State Farm train directory: {train_dir}")
    return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])


def add_statefarm_dataset(src_root, dst_root, classes, link_mode):
    counts = defaultdict(Counter)
    for split in ("train", "val", "test"):
        for cls in classes:
            src_dir = src_root / split / cls
            if not src_dir.is_dir():
                continue
            for image_path in sorted(src_dir.iterdir()):
                if not image_path.is_file():
                    continue
                out_path = dst_root / split / cls / image_path.name
                method = link_or_copy(image_path, out_path, link_mode)
                counts[split][cls] += 1
                counts[f"{split}_method"][method] += 1
    return counts


def load_dmd_rows(metadata_csv, mapping):
    rows_by_label = defaultdict(list)
    with metadata_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_label = row["true_label"]
            if src_label not in mapping:
                continue
            row["mapped_class"] = mapping[src_label]
            row["frame_idx"] = int(row["frame_idx"])
            rows_by_label[src_label].append(row)

    for label in rows_by_label:
        rows_by_label[label].sort(key=lambda r: (r["frame_idx"], r["image_path"]))
    return rows_by_label


def split_rows(rows, train_ratio, val_ratio):
    n = len(rows)
    train_end = max(1, int(math.floor(n * train_ratio)))
    val_count = max(1, int(math.floor(n * val_ratio)))
    val_end = min(n - 1, train_end + val_count) if n >= 3 else min(n, train_end + val_count)

    if n == 1:
        return {"train": rows, "val": [], "test": []}
    if n == 2:
        return {"train": rows[:1], "val": [], "test": rows[1:]}

    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]

    if not val_rows and len(test_rows) > 1:
        val_rows = test_rows[:1]
        test_rows = test_rows[1:]
    if not test_rows and len(val_rows) > 1:
        test_rows = val_rows[-1:]
        val_rows = val_rows[:-1]

    return {"train": train_rows, "val": val_rows, "test": test_rows}


def add_dmd_samples(rows_by_label, dst_root, link_mode, train_ratio, val_ratio, prefix):
    stats = defaultdict(Counter)
    manifest_rows = []

    for src_label, rows in rows_by_label.items():
        split_map = split_rows(rows, train_ratio, val_ratio)
        target_class = rows[0]["mapped_class"] if rows else ""
        for split, split_rows_list in split_map.items():
            for row in split_rows_list:
                src_path = Path(row["image_path"])
                filename = f"{prefix}_{src_label}_{src_path.name}"
                dst_path = dst_root / split / target_class / filename
                method = link_or_copy(src_path, dst_path, link_mode)
                stats[split][target_class] += 1
                stats[f"{split}_method"][method] += 1
                manifest_rows.append(
                    {
                        "source": "dmd",
                        "split": split,
                        "original_label": src_label,
                        "mapped_class": target_class,
                        "frame_idx": row["frame_idx"],
                        "image_path": str(src_path),
                        "output_path": str(dst_path),
                    }
                )
    return stats, manifest_rows


def write_manifest(manifest_path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "split",
                "original_label",
                "mapped_class",
                "frame_idx",
                "image_path",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def count_output(root, classes):
    counts = defaultdict(dict)
    for split in ("train", "val", "test"):
        for cls in classes:
            cls_dir = root / split / cls
            counts[split][cls] = len([p for p in cls_dir.iterdir() if p.is_file()]) if cls_dir.exists() else 0
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Build a mixed State Farm + mapped DMD classification dataset for fine-tuning."
    )
    parser.add_argument("--statefarm-dir", type=Path, default=Path("dataset_cls_subject"))
    parser.add_argument("--dmd-samples-dir", type=Path, default=Path("runs/dmd_samples/driver_actions_body_100"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_mix_ft"))
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--prefix", default="dmd")
    parser.add_argument(
        "--map",
        dest="mappings",
        action="append",
        default=[],
        help="Label mapping in src=dst form. Can be passed multiple times.",
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Expected 0 < train_ratio, 0 <= val_ratio, and train_ratio + val_ratio < 1.")

    mapping = DEFAULT_MAPPING.copy()
    if args.mappings:
        mapping.update(parse_mapping(args.mappings))

    metadata_csv = args.dmd_samples_dir / "samples_metadata.csv"
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Missing DMD metadata CSV: {metadata_csv}")

    classes = collect_statefarm_classes(args.statefarm_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_class_dirs(args.output_dir, classes)

    statefarm_stats = add_statefarm_dataset(args.statefarm_dir, args.output_dir, classes, args.link_mode)
    rows_by_label = load_dmd_rows(metadata_csv, mapping)
    dmd_stats, manifest_rows = add_dmd_samples(
        rows_by_label,
        args.output_dir,
        args.link_mode,
        args.train_ratio,
        args.val_ratio,
        args.prefix,
    )

    manifest_path = args.output_dir / "dmd_manifest.csv"
    write_manifest(manifest_path, manifest_rows)

    final_counts = count_output(args.output_dir, classes)

    print(f"statefarm_dir: {args.statefarm_dir}")
    print(f"dmd_samples_dir: {args.dmd_samples_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"link_mode: {args.link_mode}")
    print(f"mapping: {mapping}")
    print(f"dmd_manifest: {manifest_path}")
    print()
    print("Added State Farm images")
    for split in ("train", "val", "test"):
        split_total = sum(statefarm_stats[split].values())
        print(f"{split}: {split_total}")
    print()
    print("Added DMD images")
    for split in ("train", "val", "test"):
        split_total = sum(dmd_stats[split].values())
        by_class = ", ".join(f"{cls}={count}" for cls, count in sorted(dmd_stats[split].items()) if count)
        print(f"{split}: {split_total}" + (f" ({by_class})" if by_class else ""))
    print()
    print("Final dataset counts")
    for split in ("train", "val", "test"):
        split_total = sum(final_counts[split].values())
        print(f"{split}: {split_total}")
        for cls in classes:
            print(f"  {cls}: {final_counts[split][cls]}")


if __name__ == "__main__":
    main()
