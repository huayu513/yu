import argparse
import csv
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from convert_dmd_openlabel_to_yolo_cls import STREAM_ALIAS, load_openlabel, resolve_video_path

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

S1_MAPPING = {
    "safe_drive": "safe_drive",
    "radio": "radio",
    "drinking": "drinking",
    "talking_to_passenger": "talking_to_passenger",
}

FINAL_CLASSES = [
    "safe_drive",
    "phone_use",
    "radio",
    "drinking",
    "talking_to_passenger",
]


def ensure_class_dirs(root: Path, classes):
    for split in ("train", "val", "test"):
        for cls in classes:
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


def evenly_sample(items, max_count):
    if max_count <= 0 or len(items) <= max_count:
        return list(items)
    if max_count == 1:
        return [items[len(items) // 2]]

    sampled = []
    last_idx = len(items) - 1
    for i in range(max_count):
        idx = round(i * last_idx / (max_count - 1))
        sampled.append(items[idx])
    return sampled


def add_statefarm_dataset(src_root: Path, dst_root: Path, link_mode: str):
    stats = defaultdict(Counter)
    manifest_rows = []

    for split in ("train", "val", "test"):
        split_root = src_root / split
        grouped = defaultdict(list)
        for src_cls, dst_cls in STATEFARM_MAPPING.items():
            src_dir = split_root / src_cls
            if not src_dir.is_dir():
                continue
            for image_path in sorted(src_dir.iterdir()):
                if not image_path.is_file():
                    continue
                grouped[dst_cls].append((src_cls, image_path))

        for dst_cls, items in grouped.items():
            for src_cls, image_path in items:
                out_name = f"sf_{src_cls}_{image_path.name}"
                out_path = dst_root / split / dst_cls / out_name
                method = link_or_copy(image_path, out_path, link_mode)
                stats[split][dst_cls] += 1
                stats[f"{split}_method"][method] += 1
                manifest_rows.append(
                    {
                        "source": "statefarm",
                        "split": split,
                        "original_label": src_cls,
                        "mapped_class": dst_cls,
                        "frame_idx": "",
                        "image_path": str(image_path),
                        "output_path": str(out_path),
                    }
                )

    return stats, manifest_rows


def add_statefarm_dataset_balanced(
    src_root: Path,
    dst_root: Path,
    link_mode: str,
    phone_train_cap: int,
    phone_val_cap: int,
    phone_test_cap: int,
):
    stats = defaultdict(Counter)
    manifest_rows = []
    phone_caps = {
        "train": phone_train_cap,
        "val": phone_val_cap,
        "test": phone_test_cap,
    }

    for split in ("train", "val", "test"):
        split_root = src_root / split
        grouped = defaultdict(list)
        for src_cls, dst_cls in STATEFARM_MAPPING.items():
            src_dir = split_root / src_cls
            if not src_dir.is_dir():
                continue
            for image_path in sorted(src_dir.iterdir()):
                if not image_path.is_file():
                    continue
                grouped[dst_cls].append((src_cls, image_path))

        for dst_cls, items in grouped.items():
            if dst_cls == "phone_use":
                items = evenly_sample(items, phone_caps[split])
            for src_cls, image_path in items:
                out_name = f"sf_{src_cls}_{image_path.name}"
                out_path = dst_root / split / dst_cls / out_name
                method = link_or_copy(image_path, out_path, link_mode)
                stats[split][dst_cls] += 1
                stats[f"{split}_method"][method] += 1
                manifest_rows.append(
                    {
                        "source": "statefarm",
                        "split": split,
                        "original_label": src_cls,
                        "mapped_class": dst_cls,
                        "frame_idx": "",
                        "image_path": str(image_path),
                        "output_path": str(out_path),
                    }
                )

    return stats, manifest_rows


def load_s1_rows(metadata_csv: Path):
    rows_by_label = defaultdict(list)
    with metadata_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_label = row["true_label"]
            if src_label not in S1_MAPPING:
                continue
            row["mapped_class"] = S1_MAPPING[src_label]
            row["frame_idx"] = int(row["frame_idx"])
            rows_by_label[src_label].append(row)

    for label in rows_by_label:
        rows_by_label[label].sort(key=lambda r: (r["frame_idx"], r["image_path"]))
    return rows_by_label


def split_rows(rows, train_ratio, val_ratio):
    n = len(rows)
    train_end = max(1, math.floor(n * train_ratio))
    val_count = max(1, math.floor(n * val_ratio))
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


def add_s1_samples(rows_by_label, dst_root: Path, link_mode: str, train_ratio: float, val_ratio: float):
    stats = defaultdict(Counter)
    manifest_rows = []

    for src_label, rows in rows_by_label.items():
        split_map = split_rows(rows, train_ratio, val_ratio)
        target_class = rows[0]["mapped_class"] if rows else ""
        for split, split_rows_list in split_map.items():
            for row in split_rows_list:
                src_path = Path(row["image_path"])
                filename = f"s1_{src_label}_{src_path.name}"
                dst_path = dst_root / split / target_class / filename
                method = link_or_copy(src_path, dst_path, link_mode)
                stats[split][target_class] += 1
                stats[f"{split}_method"][method] += 1
                manifest_rows.append(
                    {
                        "source": "dmd_s1",
                        "split": split,
                        "original_label": src_label,
                        "mapped_class": target_class,
                        "frame_idx": row["frame_idx"],
                        "image_path": str(src_path),
                        "output_path": str(dst_path),
                    }
                )

    return stats, manifest_rows


def collect_cellphone_frames(openlabel, stream_key: str):
    stream_info = openlabel["streams"][stream_key]
    frame_shift = int(stream_info.get("stream_properties", {}).get("sync", {}).get("frame_shift", 0))
    cellphone_frames = []

    for obj in openlabel.get("objects", {}).values():
        if obj.get("type") != "cellphone":
            continue
        for interval in obj.get("frame_intervals", []):
            start = int(interval["frame_start"]) - frame_shift
            end = int(interval["frame_end"]) - frame_shift
            if end < 0:
                continue
            start = max(0, start)
            end = max(start, end)
            cellphone_frames.extend(range(start, end + 1))

    return sorted(set(cellphone_frames)), frame_shift


def collect_non_cellphone_frames(openlabel, stream_key: str, margin: int):
    stream_info = openlabel["streams"][stream_key]
    frame_shift = int(stream_info.get("stream_properties", {}).get("sync", {}).get("frame_shift", 0))
    frame_intervals = openlabel.get("frame_intervals", [])
    if not frame_intervals:
        return [], frame_shift

    max_frame = max(int(interval["frame_end"]) for interval in frame_intervals) - frame_shift
    max_frame = max(0, max_frame)
    blocked = set()

    for obj in openlabel.get("objects", {}).values():
        if obj.get("type") != "cellphone":
            continue
        for interval in obj.get("frame_intervals", []):
            start = int(interval["frame_start"]) - frame_shift - margin
            end = int(interval["frame_end"]) - frame_shift + margin
            if end < 0:
                continue
            start = max(0, start)
            end = max(start, min(max_frame, end))
            blocked.update(range(start, end + 1))

    safe_frames = [frame_idx for frame_idx in range(max_frame + 1) if frame_idx not in blocked]
    return safe_frames, frame_shift


def export_s2_phone_samples(
    json_path: Path,
    dst_root: Path,
    stream: str,
    every_n_frames: int,
    max_samples: int,
    train_ratio: float,
    val_ratio: float,
    jpg_quality: int,
):
    stats = defaultdict(Counter)
    manifest_rows = []
    openlabel = load_openlabel(json_path)
    stream_key = STREAM_ALIAS[stream]
    video_path = resolve_video_path(json_path, openlabel, stream_key)
    frames, frame_shift = collect_cellphone_frames(openlabel, stream_key)

    sampled_frames = [f for f in frames if f % every_n_frames == 0]
    sampled_frames = evenly_sample(sampled_frames, max_samples)
    split_map = split_rows(
        [{"frame_idx": frame_idx} for frame_idx in sampled_frames],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_stem = video_path.stem
    for split, items in split_map.items():
        for item in items:
            frame_idx = int(item["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            out_name = f"s2_phone_use_{video_stem}_f{frame_idx:06d}.jpg"
            out_path = dst_root / split / "phone_use" / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])

            stats[split]["phone_use"] += 1
            manifest_rows.append(
                {
                    "source": "dmd_s2",
                    "split": split,
                    "original_label": "cellphone",
                    "mapped_class": "phone_use",
                    "frame_idx": frame_idx,
                    "image_path": str(video_path),
                    "output_path": str(out_path),
                    "frame_shift": frame_shift,
                }
            )

    cap.release()
    return stats, manifest_rows, video_path.name, frame_shift, len(frames), len(sampled_frames)


def export_s2_safe_samples(
    json_path: Path,
    dst_root: Path,
    stream: str,
    every_n_frames: int,
    max_samples: int,
    train_ratio: float,
    val_ratio: float,
    jpg_quality: int,
    margin: int,
):
    stats = defaultdict(Counter)
    manifest_rows = []
    openlabel = load_openlabel(json_path)
    stream_key = STREAM_ALIAS[stream]
    video_path = resolve_video_path(json_path, openlabel, stream_key)
    frames, frame_shift = collect_non_cellphone_frames(openlabel, stream_key, margin)

    sampled_frames = [f for f in frames if f % every_n_frames == 0]
    sampled_frames = evenly_sample(sampled_frames, max_samples)
    split_map = split_rows(
        [{"frame_idx": frame_idx} for frame_idx in sampled_frames],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_stem = video_path.stem
    for split, items in split_map.items():
        for item in items:
            frame_idx = int(item["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            out_name = f"s2_safe_drive_{video_stem}_f{frame_idx:06d}.jpg"
            out_path = dst_root / split / "safe_drive" / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])

            stats[split]["safe_drive"] += 1
            manifest_rows.append(
                {
                    "source": "dmd_s2",
                    "split": split,
                    "original_label": "non_cellphone",
                    "mapped_class": "safe_drive",
                    "frame_idx": frame_idx,
                    "image_path": str(video_path),
                    "output_path": str(out_path),
                    "frame_shift": frame_shift,
                }
            )

    cap.release()
    return stats, manifest_rows, video_path.name, frame_shift, len(frames), len(sampled_frames)


def write_manifest(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
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
                "frame_shift",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def count_output(root: Path):
    counts = defaultdict(dict)
    for split in ("train", "val", "test"):
        for cls in FINAL_CLASSES:
            cls_dir = root / split / cls
            counts[split][cls] = len([p for p in cls_dir.iterdir() if p.is_file()]) if cls_dir.exists() else 0
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Build a 5-class distraction dataset with phone_use from State Farm + DMD S1/S2."
    )
    parser.add_argument("--statefarm-dir", type=Path, default=Path("dataset_cls_subject"))
    parser.add_argument(
        "--s1-metadata-csv", type=Path, default=Path("runs/dmd_samples/driver_actions_body_100/samples_metadata.csv")
    )
    parser.add_argument(
        "--s2-json",
        type=Path,
        default=Path(
            "dmd-dataset-mini-sample-gB-10-s2/dmd/gB/10/s2/gB_10_s2_2019-03-11T15;15;21+01;00_rgb_ann_distraction.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_mix_ft_phone5_v1"))
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--phone-train-cap", type=int, default=0)
    parser.add_argument("--phone-val-cap", type=int, default=0)
    parser.add_argument("--phone-test-cap", type=int, default=0)
    parser.add_argument("--s2-stream", choices=("body", "face", "hands"), default="body")
    parser.add_argument("--s2-every-n-frames", type=int, default=15)
    parser.add_argument("--s2-max-samples", type=int, default=300)
    parser.add_argument("--s2-safe-max-samples", type=int, default=0)
    parser.add_argument("--s2-safe-margin", type=int, default=45)
    parser.add_argument("--jpg-quality", type=int, default=95)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Expected 0 < train_ratio, 0 <= val_ratio, and train_ratio + val_ratio < 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_class_dirs(args.output_dir, FINAL_CLASSES)

    if any(v > 0 for v in (args.phone_train_cap, args.phone_val_cap, args.phone_test_cap)):
        statefarm_stats, statefarm_rows = add_statefarm_dataset_balanced(
            args.statefarm_dir,
            args.output_dir,
            args.link_mode,
            args.phone_train_cap,
            args.phone_val_cap,
            args.phone_test_cap,
        )
    else:
        statefarm_stats, statefarm_rows = add_statefarm_dataset(args.statefarm_dir, args.output_dir, args.link_mode)
    s1_rows = load_s1_rows(args.s1_metadata_csv)
    s1_stats, s1_manifest = add_s1_samples(s1_rows, args.output_dir, args.link_mode, args.train_ratio, args.val_ratio)
    s2_stats, s2_manifest, s2_video_name, s2_frame_shift, s2_all_frames, s2_export_candidates = export_s2_phone_samples(
        json_path=args.s2_json,
        dst_root=args.output_dir,
        stream=args.s2_stream,
        every_n_frames=args.s2_every_n_frames,
        max_samples=args.s2_max_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        jpg_quality=args.jpg_quality,
    )
    s2_safe_stats = defaultdict(Counter)
    s2_safe_manifest = []
    s2_safe_video_name = ""
    s2_safe_frame_shift = 0
    s2_safe_all_frames = 0
    s2_safe_export_candidates = 0
    if args.s2_safe_max_samples > 0:
        (
            s2_safe_stats,
            s2_safe_manifest,
            s2_safe_video_name,
            s2_safe_frame_shift,
            s2_safe_all_frames,
            s2_safe_export_candidates,
        ) = export_s2_safe_samples(
            json_path=args.s2_json,
            dst_root=args.output_dir,
            stream=args.s2_stream,
            every_n_frames=args.s2_every_n_frames,
            max_samples=args.s2_safe_max_samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            jpg_quality=args.jpg_quality,
            margin=args.s2_safe_margin,
        )

    manifest_rows = statefarm_rows + s1_manifest + s2_manifest + s2_safe_manifest
    manifest_path = args.output_dir / "mix_manifest.csv"
    write_manifest(manifest_path, manifest_rows)

    final_counts = count_output(args.output_dir)

    print(f"statefarm_dir: {args.statefarm_dir}")
    print(f"s1_metadata_csv: {args.s1_metadata_csv}")
    print(f"s2_json: {args.s2_json}")
    print(f"output_dir: {args.output_dir}")
    print(f"link_mode: {args.link_mode}")
    print(f"final_classes: {FINAL_CLASSES}")
    print(f"manifest: {manifest_path}")
    print()
    print("Added State Farm images")
    for split in ("train", "val", "test"):
        print(f"{split}: {sum(statefarm_stats[split].values())}")
    print()
    print("Added DMD S1 images")
    for split in ("train", "val", "test"):
        split_total = sum(s1_stats[split].values())
        by_class = ", ".join(f"{cls}={count}" for cls, count in sorted(s1_stats[split].items()) if count)
        print(f"{split}: {split_total}" + (f" ({by_class})" if by_class else ""))
    print()
    print("Added DMD S2 phone_use images")
    print(f"s2_video: {s2_video_name}")
    print(f"s2_stream: {args.s2_stream}")
    print(f"s2_frame_shift: {s2_frame_shift}")
    print(f"s2_cellphone_frames_total: {s2_all_frames}")
    print(f"s2_sampled_frames_before_split: {s2_export_candidates}")
    for split in ("train", "val", "test"):
        split_total = sum(s2_stats[split].values())
        by_class = ", ".join(f"{cls}={count}" for cls, count in sorted(s2_stats[split].items()) if count)
        print(f"{split}: {split_total}" + (f" ({by_class})" if by_class else ""))
    if args.s2_safe_max_samples > 0:
        print()
        print("Added DMD S2 safe_drive images")
        print(f"s2_safe_video: {s2_safe_video_name}")
        print(f"s2_safe_stream: {args.s2_stream}")
        print(f"s2_safe_frame_shift: {s2_safe_frame_shift}")
        print(f"s2_non_cellphone_frames_total: {s2_safe_all_frames}")
        print(f"s2_safe_sampled_frames_before_split: {s2_safe_export_candidates}")
        print(f"s2_safe_margin: {args.s2_safe_margin}")
        for split in ("train", "val", "test"):
            split_total = sum(s2_safe_stats[split].values())
            by_class = ", ".join(f"{cls}={count}" for cls, count in sorted(s2_safe_stats[split].items()) if count)
            print(f"{split}: {split_total}" + (f" ({by_class})" if by_class else ""))
    print()
    print("Final dataset counts")
    for split in ("train", "val", "test"):
        split_total = sum(final_counts[split].values())
        print(f"{split}: {split_total}")
        for cls in FINAL_CLASSES:
            print(f"  {cls}: {final_counts[split][cls]}")


if __name__ == "__main__":
    main()
