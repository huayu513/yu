import argparse
import csv
import json
from pathlib import Path

import cv2
from build_statefarm_dmd_phone5_dataset import collect_cellphone_frames
from convert_dmd_openlabel_to_yolo_cls import collect_frame_labels, load_openlabel
from convert_dmd_s5_to_fatigue_face_cls import build_samples, split_rows

DISTRACTION_LABELS = ("safe_drive", "phone_use", "radio", "drinking", "talking_to_passenger")
S1_LABELS = ("safe_drive", "radio", "drinking", "talking_to_passenger")
FATIGUE_LABELS = ("non_fatigue", "fatigue")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build multimodal driver dataset v3 with expanded S1/S2/S5 paired samples."
    )
    parser.add_argument(
        "--s1-json",
        type=Path,
        default=Path(
            "dmd-dataset-mini-sample-gA-3-s1/dmd/gA/3/s1/gA_3_s1_2019-03-08T10;27;38+01;00_rgb_ann_distraction.json"
        ),
    )
    parser.add_argument(
        "--s2-json",
        type=Path,
        default=Path(
            "dmd-dataset-mini-sample-gB-10-s2/dmd/gB/10/s2/gB_10_s2_2019-03-11T15;15;21+01;00_rgb_ann_distraction.json"
        ),
    )
    parser.add_argument(
        "--s5-json",
        type=Path,
        default=Path(
            "dmd-dataset-mini-sample-gB-10-s5/dmd/gB/10/s5/gB_10_s5_2019-03-12T10;35;20+01;00_rgb_ann_drowsiness.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_multimodal_driver_v3"))
    parser.add_argument("--s1-per-class", type=int, default=200)
    parser.add_argument("--s2-phone-count", type=int, default=200)
    parser.add_argument("--s1-val-ratio", type=float, default=0.15)
    parser.add_argument("--s1-test-ratio", type=float, default=0.15)
    parser.add_argument("--s5-fatigue-every-n", type=int, default=4)
    parser.add_argument("--s5-nonfatigue-every-n", type=int, default=8)
    parser.add_argument("--s5-close-threshold", type=int, default=20)
    parser.add_argument("--s5-train-ratio", type=float, default=0.6)
    parser.add_argument("--s5-val-ratio", type=float, default=0.2)
    parser.add_argument("--jpg-quality", type=int, default=95)
    return parser.parse_args()


def ensure_dirs(root: Path):
    for split in ("train", "val", "test"):
        (root / "body" / split).mkdir(parents=True, exist_ok=True)
        (root / "face" / split).mkdir(parents=True, exist_ok=True)


def load_stream_shifts(openlabel: dict):
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
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_imwrite(out_path, frame, jpg_quality)
    return True


def close_caps(cap_cache):
    for cap in cap_cache.values():
        cap.release()


def evenly_sample(items, target_count: int):
    items = sorted(items)
    if target_count <= 0 or len(items) <= target_count:
        return items
    if target_count == 1:
        return [items[len(items) // 2]]
    last_idx = len(items) - 1
    return [items[round(i * last_idx / (target_count - 1))] for i in range(target_count)]


def split_even(rows, val_ratio: float, test_ratio: float):
    rows = list(rows)
    n = len(rows)
    test_n = round(n * test_ratio)
    val_n = round(n * val_ratio)
    train_n = n - test_n - val_n
    return {
        "train": rows[:train_n],
        "val": rows[train_n : train_n + val_n],
        "test": rows[train_n + val_n :],
    }


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


def build_s1_distraction(args, all_split_rows, cap_cache):
    openlabel = load_openlabel(args.s1_json)
    shifts = load_stream_shifts(openlabel)
    frame_to_labels, _, _ = collect_frame_labels(openlabel, "driver_actions", "body_camera")
    by_label = {label: [] for label in S1_LABELS}
    for frame_idx, labels in frame_to_labels.items():
        if len(labels) != 1:
            continue
        label = next(iter(labels))
        if label in by_label:
            by_label[label].append(frame_idx)

    body_video = args.s1_json.parent / args.s1_json.name.replace("_rgb_ann_distraction.json", "_rgb_body.mp4")
    face_video = args.s1_json.parent / args.s1_json.name.replace("_rgb_ann_distraction.json", "_rgb_face.mp4")

    for label in S1_LABELS:
        selected = evenly_sample(by_label[label], args.s1_per_class)
        rows = []
        for frame_idx in selected:
            face_frame_idx = frame_idx - shifts["face"]
            if face_frame_idx < 0:
                continue
            sample_id = f"s1_{label}_{frame_idx:06d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "source": "s1_distraction",
                    "distraction_label": label,
                    "fatigue_label": "-1",
                    "body_frame_idx": frame_idx,
                    "face_frame_idx": face_frame_idx,
                    "notes": "",
                }
            )
        split_map = split_even(rows, args.s1_val_ratio, args.s1_test_ratio)
        for split, split_rows_list in split_map.items():
            for row in split_rows_list:
                row["split"] = split
                body_out = args.output_dir / "body" / split / f"{row['sample_id']}.jpg"
                face_out = args.output_dir / "face" / split / f"{row['sample_id']}.jpg"
                ok_body = export_frame(cap_cache, body_video, row["body_frame_idx"], body_out, args.jpg_quality)
                ok_face = export_frame(cap_cache, face_video, row["face_frame_idx"], face_out, args.jpg_quality)
                if not (ok_body and ok_face):
                    continue
                row["body_path"] = str(body_out.resolve())
                row["face_path"] = str(face_out.resolve())
                all_split_rows[split].append(row)


def build_s2_phone(args, all_split_rows, cap_cache):
    openlabel = json.loads(args.s2_json.read_text(encoding="utf-8"))["openlabel"]
    shifts = load_stream_shifts(openlabel)
    phone_frames, _ = collect_cellphone_frames(openlabel, "body_camera")
    selected = evenly_sample(phone_frames, args.s2_phone_count)

    body_video = args.s2_json.parent / args.s2_json.name.replace("_rgb_ann_distraction.json", "_rgb_body.mp4")
    face_video = args.s2_json.parent / args.s2_json.name.replace("_rgb_ann_distraction.json", "_rgb_face.mp4")

    rows = []
    for frame_idx in selected:
        face_frame_idx = frame_idx - shifts["face"]
        if face_frame_idx < 0:
            continue
        sample_id = f"s2_phone_use_{frame_idx:06d}"
        rows.append(
            {
                "sample_id": sample_id,
                "source": "s2_distraction",
                "distraction_label": "phone_use",
                "fatigue_label": "-1",
                "body_frame_idx": frame_idx,
                "face_frame_idx": face_frame_idx,
                "notes": "cellphone",
            }
        )

    split_map = split_even(rows, args.s1_val_ratio, args.s1_test_ratio)
    for split, split_rows_list in split_map.items():
        for row in split_rows_list:
            row["split"] = split
            body_out = args.output_dir / "body" / split / f"{row['sample_id']}.jpg"
            face_out = args.output_dir / "face" / split / f"{row['sample_id']}.jpg"
            ok_body = export_frame(cap_cache, body_video, row["body_frame_idx"], body_out, args.jpg_quality)
            ok_face = export_frame(cap_cache, face_video, row["face_frame_idx"], face_out, args.jpg_quality)
            if not (ok_body and ok_face):
                continue
            row["body_path"] = str(body_out.resolve())
            row["face_path"] = str(face_out.resolve())
            all_split_rows[split].append(row)


def build_s5_fatigue(args, all_split_rows, cap_cache):
    openlabel = json.loads(args.s5_json.read_text(encoding="utf-8"))["openlabel"]
    shifts = load_stream_shifts(openlabel)
    samples = build_samples(
        openlabel,
        args.s5_close_threshold,
        args.s5_fatigue_every_n,
        args.s5_nonfatigue_every_n,
    )
    split_rows_by_label = {
        label: split_rows(rows, args.s5_train_ratio, args.s5_val_ratio) for label, rows in samples.items()
    }

    face_video = args.s5_json.parent / args.s5_json.name.replace("_rgb_ann_drowsiness.json", "_rgb_face.mp4")
    body_video = args.s5_json.parent / args.s5_json.name.replace("_rgb_ann_drowsiness.json", "_rgb_body.mp4")

    for label, split_map in split_rows_by_label.items():
        for split, rows in split_map.items():
            for row_meta in rows:
                face_frame_idx = int(row_meta["frame_idx"])
                original_frame_idx = face_frame_idx + shifts["face"]
                body_frame_idx = original_frame_idx - shifts["body"]
                if body_frame_idx < 0:
                    continue
                sample_id = f"s5_{label}_{face_frame_idx:06d}"
                row = {
                    "sample_id": sample_id,
                    "split": split,
                    "source": "s5_fatigue",
                    "distraction_label": "-1",
                    "fatigue_label": label,
                    "body_frame_idx": body_frame_idx,
                    "face_frame_idx": face_frame_idx,
                    "notes": row_meta.get("source_action", ""),
                }
                body_out = args.output_dir / "body" / split / f"{sample_id}.jpg"
                face_out = args.output_dir / "face" / split / f"{sample_id}.jpg"
                ok_body = export_frame(cap_cache, body_video, body_frame_idx, body_out, args.jpg_quality)
                ok_face = export_frame(cap_cache, face_video, face_frame_idx, face_out, args.jpg_quality)
                if not (ok_body and ok_face):
                    continue
                row["body_path"] = str(body_out.resolve())
                row["face_path"] = str(face_out.resolve())
                all_split_rows[split].append(row)


def main():
    args = parse_args()
    ensure_dirs(args.output_dir)
    cap_cache = {}
    split_rows = {"train": [], "val": [], "test": []}

    build_s1_distraction(args, split_rows, cap_cache)
    build_s2_phone(args, split_rows, cap_cache)
    build_s5_fatigue(args, split_rows, cap_cache)
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
