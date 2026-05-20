import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2


DISTRACTION_LABELS = ("safe_drive", "radio", "drinking", "talking_to_passenger")
FATIGUE_LABELS = ("non_fatigue", "fatigue")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build multimodal driver dataset v1 with paired body/face images and dual-task labels."
    )
    parser.add_argument(
        "--s1-metadata",
        type=Path,
        default=Path("runs/dmd_samples/driver_actions_body_100/samples_metadata.csv"),
        help="Metadata CSV from sampled DMD S1 distraction body images.",
    )
    parser.add_argument(
        "--s1-root",
        type=Path,
        default=Path("dmd-dataset-mini-sample-gA-3-s1/dmd/gA/3/s1"),
        help="Raw DMD S1 directory containing rgb_body/rgb_face videos and annotation JSON.",
    )
    parser.add_argument(
        "--s5-metadata",
        type=Path,
        default=Path("dataset_fatigue_face_s5_v1/samples_metadata.csv"),
        help="Metadata CSV from prepared DMD S5 fatigue face dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_multimodal_driver_v1"),
        help="Output multimodal dataset root.",
    )
    parser.add_argument("--s1-val-ratio", type=float, default=0.15)
    parser.add_argument("--s1-test-ratio", type=float, default=0.15)
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to place already-exported modality images into the output dataset.",
    )
    return parser.parse_args()


def choose_split(key: str, val_ratio: float, test_ratio: float):
    score = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 10000 / 10000.0
    if score < test_ratio:
        return "test"
    if score < test_ratio + val_ratio:
        return "val"
    return "train"


def ensure_dirs(root: Path):
    for split in ("train", "val", "test"):
        (root / "body" / split).mkdir(parents=True, exist_ok=True)
        (root / "face" / split).mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    shutil.copy2(src, dst)


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
    if str(video_path) not in cap_cache:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        cap_cache[str(video_path)] = cap
    cap = cap_cache[str(video_path)]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
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


def main():
    args = parse_args()
    if args.s1_val_ratio < 0 or args.s1_test_ratio < 0 or args.s1_val_ratio + args.s1_test_ratio >= 1:
        raise ValueError("Expected non-negative s1 val/test ratios with sum < 1.")

    ensure_dirs(args.output_dir)
    cap_cache = {}
    split_rows = {"train": [], "val": [], "test": []}

    # S1: distraction samples with body images already exported. Build paired face frames and distraction labels.
    with args.s1_metadata.open("r", encoding="utf-8-sig") as f:
        s1_rows = list(csv.DictReader(f))

    s1_rows = [row for row in s1_rows if row["true_label"] in DISTRACTION_LABELS]
    if not s1_rows:
        raise RuntimeError("No usable S1 distraction rows found.")

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
        split = choose_split(sample_id, args.s1_val_ratio, args.s1_test_ratio)

        body_video = args.s1_root / row["video_name"]
        face_video = args.s1_root / row["video_name"].replace("_rgb_body.mp4", "_rgb_face.mp4")

        body_out = args.output_dir / "body" / split / f"{sample_id}.jpg"
        face_out = args.output_dir / "face" / split / f"{sample_id}.jpg"
        export_frame(cap_cache, body_video, body_frame_idx, body_out, args.jpg_quality)
        export_frame(cap_cache, face_video, face_frame_idx, face_out, args.jpg_quality)

        split_rows[split].append(
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

    # S5: fatigue samples with face images already exported. Build paired body frames and fatigue labels.
    with args.s5_metadata.open("r", encoding="utf-8-sig") as f:
        s5_rows = list(csv.DictReader(f))
    if not s5_rows:
        raise RuntimeError("No S5 fatigue rows found.")

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

        sample_id = f"s5_{label}_{face_frame_idx:06d}"
        face_video = Path(row["face_video"])
        body_video = face_video.with_name(face_video.name.replace("_rgb_face.mp4", "_rgb_body.mp4"))

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
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"output_dir: {args.output_dir}")
    for split in ("train", "val", "test"):
        print(f"{split}: {len(split_rows[split])}")
    print(f"metadata: {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
