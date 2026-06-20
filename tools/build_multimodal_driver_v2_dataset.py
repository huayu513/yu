import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2

DISTRACTION_LABELS = ("safe_drive", "phone_use", "radio", "drinking", "talking_to_passenger")
FATIGUE_LABELS = ("non_fatigue", "fatigue")
S1_KEEP = {"safe_drive", "radio", "drinking", "talking_to_passenger"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build multimodal driver dataset v2 with S1 + S2 distraction and S5 fatigue."
    )
    parser.add_argument(
        "--s1-metadata",
        type=Path,
        default=Path("runs/dmd_samples/driver_actions_body_100/samples_metadata.csv"),
    )
    parser.add_argument(
        "--s1-root",
        type=Path,
        default=Path("dmd-dataset-mini-sample-gA-3-s1/dmd/gA/3/s1"),
    )
    parser.add_argument(
        "--s2-manifest",
        type=Path,
        default=Path("dataset_mix_ft_phone5_v2/mix_manifest.csv"),
        help="Manifest containing dmd_s2 phone_use rows.",
    )
    parser.add_argument(
        "--s2-root",
        type=Path,
        default=Path("dmd-dataset-mini-sample-gB-10-s2/dmd/gB/10/s2"),
    )
    parser.add_argument(
        "--s2-phone-use-count",
        type=int,
        default=100,
        help="How many S2 phone_use samples to keep for balancing.",
    )
    parser.add_argument(
        "--s5-metadata",
        type=Path,
        default=Path("dataset_fatigue_face_s5_v1/samples_metadata.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_multimodal_driver_v2"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--jpg-quality", type=int, default=95)
    return parser.parse_args()


def choose_split(key: str, val_ratio: float, test_ratio: float):
    score = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 10000 / 10000.0
    if score < test_ratio:
        return "test"
    if score < test_ratio + val_ratio:
        return "val"
    return "train"


def split_rows_per_class(rows, label_key: str, val_ratio: float, test_ratio: float):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[label_key], []).append(row)

    result = {"train": [], "val": [], "test": []}
    for label, label_rows in grouped.items():
        label_rows = sorted(label_rows, key=lambda r: hashlib.md5(r["sample_id"].encode("utf-8")).hexdigest())
        total = len(label_rows)
        test_n = round(total * test_ratio)
        val_n = round(total * val_ratio)
        train_n = total - test_n - val_n
        result["train"].extend(label_rows[:train_n])
        result["val"].extend(label_rows[train_n : train_n + val_n])
        result["test"].extend(label_rows[train_n + val_n :])
    return result


def ensure_dirs(root: Path):
    for split in ("train", "val", "test"):
        (root / "body" / split).mkdir(parents=True, exist_ok=True)
        (root / "face" / split).mkdir(parents=True, exist_ok=True)


def load_stream_shifts(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        openlabel = json.load(f)["openlabel"]
    streams = openlabel["streams"]
    return {
        "face": int(streams["face_camera"].get("stream_properties", {}).get("sync", {}).get("frame_shift", 0)),
        "body": int(streams["body_camera"].get("stream_properties", {}).get("sync", {}).get("frame_shift", 0)),
    }


def safe_imwrite(path: Path, image, jpg_quality: int):
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def export_frame(cap_cache, video_path: Path, frame_idx: int, out_path: Path, jpg_quality: int):
    key = str(video_path)
    if key not in cap_cache:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        cap_cache[key] = cap
    cap = cap_cache[key]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_imwrite(out_path, frame, jpg_quality)


def close_caps(cap_cache):
    for cap in cap_cache.values():
        cap.release()


def write_rows(csv_path: Path, rows):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "split",
                "source",
                "body_path",
                "face_path",
                "distraction_label",
                "fatigue_label",
                "body_frame_idx",
                "face_frame_idx",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def count_labels(rows, key):
    counts = {}
    for row in rows:
        label = row[key]
        counts[label] = counts.get(label, 0) + 1
    return counts


def load_s1_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row["true_label"] in S1_KEEP]


def load_s2_rows(path: Path, keep_count: int):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if row["source"] == "dmd_s2" and row["mapped_class"] == "phone_use"]
    rows.sort(key=lambda r: int(r["frame_idx"]))
    if keep_count >= len(rows):
        return rows
    step = len(rows) / keep_count
    selected = []
    for i in range(keep_count):
        selected.append(rows[int(i * step)])
    return selected


def main():
    args = parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Expected non-negative val/test ratios with sum < 1.")

    ensure_dirs(args.output_dir)
    cap_cache = {}
    split_rows = {"train": [], "val": [], "test": []}
    distraction_pool = []

    # S1 distraction rows: keep four classes, re-split deterministically.
    s1_rows = load_s1_rows(args.s1_metadata)
    if not s1_rows:
        raise RuntimeError("No usable S1 rows found.")
    s1_json_path = args.s1_root / s1_rows[0]["json_name"]
    s1_shifts = load_stream_shifts(s1_json_path)
    for row in s1_rows:
        label = row["true_label"]
        body_frame_idx = int(row["frame_idx"])
        body_shift = int(row["frame_shift"])
        original_frame_idx = body_frame_idx + body_shift
        face_frame_idx = original_frame_idx - s1_shifts["face"]
        if face_frame_idx < 0:
            continue
        sample_id = f"s1_{label}_{body_frame_idx:06d}"
        split = choose_split(sample_id, args.val_ratio, args.test_ratio)
        body_video = args.s1_root / row["video_name"]
        face_video = args.s1_root / row["video_name"].replace("_rgb_body.mp4", "_rgb_face.mp4")
        body_out = args.output_dir / "body" / split / f"{sample_id}.jpg"
        face_out = args.output_dir / "face" / split / f"{sample_id}.jpg"
        export_frame(cap_cache, body_video, body_frame_idx, body_out, args.jpg_quality)
        export_frame(cap_cache, face_video, face_frame_idx, face_out, args.jpg_quality)
        distraction_pool.append(
            {
                "sample_id": sample_id,
                "split": split,
                "source": "s1_distraction",
                "body_path": str(body_out.resolve()),
                "face_path": str(face_out.resolve()),
                "distraction_label": label,
                "fatigue_label": "-1",
                "body_frame_idx": body_frame_idx,
                "face_frame_idx": face_frame_idx,
                "notes": "",
            }
        )

    # S2 distraction rows: add balanced phone_use.
    s2_rows = load_s2_rows(args.s2_manifest, args.s2_phone_use_count)
    if not s2_rows:
        raise RuntimeError("No usable S2 phone_use rows found.")
    s2_json_path = args.s2_root / "gB_10_s2_2019-03-11T15;15;21+01;00_rgb_ann_distraction.json"
    s2_shifts = load_stream_shifts(s2_json_path)
    s2_body_video = args.s2_root / "gB_10_s2_2019-03-11T15;15;21+01;00_rgb_body.mp4"
    s2_face_video = args.s2_root / "gB_10_s2_2019-03-11T15;15;21+01;00_rgb_face.mp4"
    for row in s2_rows:
        label = "phone_use"
        body_frame_idx = int(row["frame_idx"])
        body_shift = int(row.get("frame_shift") or 0)
        original_frame_idx = body_frame_idx + body_shift
        face_frame_idx = original_frame_idx - s2_shifts["face"]
        if face_frame_idx < 0:
            continue
        sample_id = f"s2_{label}_{body_frame_idx:06d}"
        split = choose_split(sample_id, args.val_ratio, args.test_ratio)
        body_out = args.output_dir / "body" / split / f"{sample_id}.jpg"
        face_out = args.output_dir / "face" / split / f"{sample_id}.jpg"
        export_frame(cap_cache, s2_body_video, body_frame_idx, body_out, args.jpg_quality)
        export_frame(cap_cache, s2_face_video, face_frame_idx, face_out, args.jpg_quality)
        distraction_pool.append(
            {
                "sample_id": sample_id,
                "split": split,
                "source": "s2_distraction",
                "body_path": str(body_out.resolve()),
                "face_path": str(face_out.resolve()),
                "distraction_label": label,
                "fatigue_label": "-1",
                "body_frame_idx": body_frame_idx,
                "face_frame_idx": face_frame_idx,
                "notes": row.get("original_label", ""),
            }
        )

    # Re-split distraction rows per class to keep the five classes balanced.
    distraction_split_rows = split_rows_per_class(
        distraction_pool, "distraction_label", args.val_ratio, args.test_ratio
    )
    for split in ("train", "val", "test"):
        for row in distraction_split_rows[split]:
            row["split"] = split
        split_rows[split].extend(distraction_split_rows[split])

    # S5 fatigue rows: keep original split to preserve face-fatigue construction.
    with args.s5_metadata.open("r", encoding="utf-8-sig", newline="") as f:
        s5_rows = list(csv.DictReader(f))
    if not s5_rows:
        raise RuntimeError("No usable S5 rows found.")
    s5_json_path = Path(s5_rows[0]["json_path"])
    s5_shifts = load_stream_shifts(s5_json_path)
    for row in s5_rows:
        split = row["split"]
        label = row["label"]
        face_frame_idx = int(row["frame_idx"])
        original_frame_idx = face_frame_idx + s5_shifts["face"]
        body_frame_idx = original_frame_idx - s5_shifts["body"]
        if body_frame_idx < 0:
            continue
        face_video = Path(row["face_video"])
        body_video = face_video.with_name(face_video.name.replace("_rgb_face.mp4", "_rgb_body.mp4"))
        sample_id = f"s5_{label}_{face_frame_idx:06d}"
        body_out = args.output_dir / "body" / split / f"{sample_id}.jpg"
        face_out = args.output_dir / "face" / split / f"{sample_id}.jpg"
        export_frame(cap_cache, face_video, face_frame_idx, face_out, args.jpg_quality)
        export_frame(cap_cache, body_video, body_frame_idx, body_out, args.jpg_quality)
        split_rows[split].append(
            {
                "sample_id": sample_id,
                "split": split,
                "source": "s5_fatigue",
                "body_path": str(body_out.resolve()),
                "face_path": str(face_out.resolve()),
                "distraction_label": "-1",
                "fatigue_label": label,
                "body_frame_idx": body_frame_idx,
                "face_frame_idx": face_frame_idx,
                "notes": row.get("source_action", ""),
            }
        )

    close_caps(cap_cache)

    for split in ("train", "val", "test"):
        write_rows(args.output_dir / f"{split}.csv", split_rows[split])

    meta = {
        "distraction_classes": list(DISTRACTION_LABELS),
        "fatigue_classes": list(FATIGUE_LABELS),
        "format": "paired_body_face_dual_task",
        "missing_label_token": "-1",
        "train_count": len(split_rows["train"]),
        "val_count": len(split_rows["val"]),
        "test_count": len(split_rows["test"]),
        "train_distraction_counts": count_labels(
            [r for r in split_rows["train"] if r["distraction_label"] != "-1"], "distraction_label"
        ),
        "val_distraction_counts": count_labels(
            [r for r in split_rows["val"] if r["distraction_label"] != "-1"], "distraction_label"
        ),
        "test_distraction_counts": count_labels(
            [r for r in split_rows["test"] if r["distraction_label"] != "-1"], "distraction_label"
        ),
        "train_fatigue_counts": count_labels(
            [r for r in split_rows["train"] if r["fatigue_label"] != "-1"], "fatigue_label"
        ),
        "val_fatigue_counts": count_labels(
            [r for r in split_rows["val"] if r["fatigue_label"] != "-1"], "fatigue_label"
        ),
        "test_fatigue_counts": count_labels(
            [r for r in split_rows["test"] if r["fatigue_label"] != "-1"], "fatigue_label"
        ),
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"output_dir: {args.output_dir}")
    for split in ("train", "val", "test"):
        print(f"{split}: {len(split_rows[split])}")
    print(f"metadata: {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
