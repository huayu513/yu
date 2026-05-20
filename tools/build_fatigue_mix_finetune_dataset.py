import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = ("fatigue", "non_fatigue")


def parse_key_value_items(items):
    mapping = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid repeat mapping '{item}', expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid repeat mapping '{item}', expected key=value.")
        mapping[key] = int(value)
    return mapping


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a mixed NTHU + S5 fatigue classification dataset for target-domain fine-tuning."
    )
    parser.add_argument(
        "--nthu-dir",
        type=Path,
        default=Path("dataset_fatigue_face_nthu_v1"),
        help="Prepared NTHU fatigue dataset with train/val/test splits.",
    )
    parser.add_argument(
        "--s5-dir",
        type=Path,
        default=Path("dataset_fatigue_face_s5_v1"),
        help="Prepared S5 fatigue dataset with train/val/test splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_fatigue_face_mix_ft_v1"),
        help="Output directory for the mixed fine-tuning dataset.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Whether to hardlink or copy images into the output dataset.",
    )
    parser.add_argument(
        "--s5-train-repeat",
        type=int,
        default=20,
        help="Repeat S5 train samples this many times to give target-domain data meaningful weight.",
    )
    parser.add_argument(
        "--s5-val-repeat",
        type=int,
        default=1,
        help="Repeat S5 val samples this many times in the mixed validation split.",
    )
    parser.add_argument(
        "--include-s5-test",
        action="store_true",
        help="Also add S5 test into the mixed test split. Disabled by default to keep S5 test untouched for final evaluation.",
    )
    parser.add_argument(
        "--s5-metadata-csv",
        type=Path,
        default=Path("dataset_fatigue_face_s5_v1/samples_metadata.csv"),
        help="Optional S5 metadata CSV used for source_action-aware weighting.",
    )
    parser.add_argument(
        "--s5-train-action-repeat",
        dest="s5_train_action_repeats",
        action="append",
        default=[],
        help="Per-source_action repeat override for S5 train, e.g. 'yawning/Yawning with hand=80'.",
    )
    return parser.parse_args()


def ensure_dirs(root: Path):
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            (root / split / cls).mkdir(parents=True, exist_ok=True)


def validate_dataset(root: Path, name: str):
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing {name} split directory: {split_dir}")
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Missing {name} class directory: {cls_dir}")


def iter_images(cls_dir: Path):
    return sorted([p for p in cls_dir.iterdir() if p.is_file()])


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


def load_s5_metadata_map(metadata_csv: Path):
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Missing S5 metadata CSV: {metadata_csv}")

    metadata = {}
    with metadata_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[Path(row["image_path"]).name] = row
    return metadata


def add_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    source_name: str,
    link_mode: str,
    repeat: int = 1,
):
    stats = Counter()
    manifest_rows = []

    for cls in CLASSES:
        src_dir = src_root / split / cls
        for image_path in iter_images(src_dir):
            for rep_idx in range(repeat):
                if repeat == 1:
                    filename = f"{source_name}_{image_path.name}"
                else:
                    filename = f"{source_name}_r{rep_idx:02d}_{image_path.name}"
                dst_path = dst_root / split / cls / filename
                method = link_or_copy(image_path, dst_path, link_mode)
                stats[cls] += 1
                stats[f"method_{method}"] += 1
                manifest_rows.append(
                    {
                        "source": source_name,
                        "split": split,
                        "class_name": cls,
                        "repeat_index": rep_idx,
                        "source_path": str(image_path.resolve()),
                        "output_path": str(dst_path.resolve()),
                    }
                )
    return stats, manifest_rows


def add_s5_train_weighted_split(
    s5_root: Path,
    dst_root: Path,
    link_mode: str,
    default_repeat: int,
    action_repeats: dict,
    metadata_map: dict,
):
    stats = Counter()
    manifest_rows = []
    split = "train"

    for cls in CLASSES:
        src_dir = s5_root / split / cls
        for image_path in iter_images(src_dir):
            meta = metadata_map.get(image_path.name, {})
            source_action = meta.get("source_action", "")
            repeat = action_repeats.get(source_action, default_repeat)
            for rep_idx in range(repeat):
                if repeat == 1:
                    filename = f"s5_{image_path.name}"
                else:
                    filename = f"s5_r{rep_idx:02d}_{image_path.name}"
                dst_path = dst_root / split / cls / filename
                method = link_or_copy(image_path, dst_path, link_mode)
                stats[cls] += 1
                stats[f"method_{method}"] += 1
                if source_action:
                    stats[f"source_action::{source_action}"] += 1
                manifest_rows.append(
                    {
                        "source": "s5",
                        "split": split,
                        "class_name": cls,
                        "repeat_index": rep_idx,
                        "source_path": str(image_path.resolve()),
                        "output_path": str(dst_path.resolve()),
                        "source_action": source_action,
                    }
                )
    return stats, manifest_rows


def count_output(root: Path):
    counts = defaultdict(Counter)
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            cls_dir = root / split / cls
            counts[split][cls] = len([p for p in cls_dir.iterdir() if p.is_file()]) if cls_dir.exists() else 0
    return counts


def write_manifest(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "split", "class_name", "repeat_index", "source_path", "output_path", "source_action"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.s5_train_repeat <= 0 or args.s5_val_repeat <= 0:
        raise ValueError("Repeat factors must be positive integers.")
    action_repeats = parse_key_value_items(args.s5_train_action_repeats) if args.s5_train_action_repeats else {}

    validate_dataset(args.nthu_dir, "NTHU")
    validate_dataset(args.s5_dir, "S5")
    ensure_dirs(args.output_dir)

    all_rows = []
    added_stats = defaultdict(Counter)

    for split in ("train", "val", "test"):
        stats, rows = add_split(args.nthu_dir, args.output_dir, split, "nthu", args.link_mode, repeat=1)
        added_stats[f"nthu_{split}"] = stats
        all_rows.extend(rows)

    if action_repeats:
        metadata_map = load_s5_metadata_map(args.s5_metadata_csv)
        stats, rows = add_s5_train_weighted_split(
            args.s5_dir,
            args.output_dir,
            args.link_mode,
            default_repeat=args.s5_train_repeat,
            action_repeats=action_repeats,
            metadata_map=metadata_map,
        )
        added_stats["s5_train"] = stats
        all_rows.extend(rows)
    else:
        stats, rows = add_split(args.s5_dir, args.output_dir, "train", "s5", args.link_mode, repeat=args.s5_train_repeat)
        added_stats["s5_train"] = stats
        all_rows.extend(rows)

    s5_plan = {
        "val": args.s5_val_repeat,
    }
    if args.include_s5_test:
        s5_plan["test"] = 1

    for split, repeat in s5_plan.items():
        stats, rows = add_split(args.s5_dir, args.output_dir, split, "s5", args.link_mode, repeat=repeat)
        added_stats[f"s5_{split}"] = stats
        all_rows.extend(rows)

    manifest_path = args.output_dir / "mix_manifest.csv"
    write_manifest(manifest_path, all_rows)
    final_counts = count_output(args.output_dir)

    print(f"nthu_dir: {args.nthu_dir}")
    print(f"s5_dir: {args.s5_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"manifest: {manifest_path}")
    print(f"link_mode: {args.link_mode}")
    print(f"s5_train_repeat: {args.s5_train_repeat}")
    print(f"s5_val_repeat: {args.s5_val_repeat}")
    print(f"include_s5_test: {args.include_s5_test}")
    if action_repeats:
        print(f"s5_metadata_csv: {args.s5_metadata_csv}")
        print(f"s5_train_action_repeats: {action_repeats}")
    print()
    print("Added images")
    for key in sorted(added_stats):
        total = added_stats[key]["fatigue"] + added_stats[key]["non_fatigue"]
        print(f"{key}: {total}")
        print(f"  fatigue: {added_stats[key]['fatigue']}")
        print(f"  non_fatigue: {added_stats[key]['non_fatigue']}")
        action_keys = [k for k in added_stats[key] if k.startswith("source_action::")]
        for action_key in sorted(action_keys):
            action = action_key.split("::", 1)[1]
            print(f"  {action}: {added_stats[key][action_key]}")
    print()
    print("Final dataset counts")
    for split in ("train", "val", "test"):
        total = final_counts[split]["fatigue"] + final_counts[split]["non_fatigue"]
        print(f"{split}: {total}")
        print(f"  fatigue: {final_counts[split]['fatigue']}")
        print(f"  non_fatigue: {final_counts[split]['non_fatigue']}")


if __name__ == "__main__":
    main()
