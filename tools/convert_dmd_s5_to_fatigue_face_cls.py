import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DMD S5 drowsiness annotations into a face-based fatigue/non_fatigue classification dataset."
    )
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to the DMD S5 drowsiness OpenLABEL JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_fatigue_face_s5_v1"),
        help="Output classification dataset directory.",
    )
    parser.add_argument(
        "--close-threshold",
        type=int,
        default=20,
        help="Only eyes_state/close intervals at least this many frames long are treated as fatigue.",
    )
    parser.add_argument(
        "--fatigue-every-n",
        type=int,
        default=5,
        help="Sample every Nth frame from fatigue intervals.",
    )
    parser.add_argument(
        "--nonfatigue-every-n",
        type=int,
        default=10,
        help="Sample every Nth frame from open-eye intervals.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality for exported images.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.fatigue_every_n <= 0 or args.nonfatigue_every_n <= 0:
        raise ValueError("Sampling steps must be positive.")
    if args.close_threshold <= 0:
        raise ValueError("close-threshold must be positive.")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Expected 0 < train_ratio, 0 <= val_ratio, and train_ratio + val_ratio < 1.")


def load_openlabel(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)["openlabel"]


def resolve_face_video(json_path: Path, openlabel: dict) -> Path:
    streams = openlabel.get("streams", {})
    face_stream = streams.get("face_camera")
    if not face_stream:
        raise KeyError("Could not find face_camera stream in OpenLABEL JSON.")
    uri = face_stream["uri"]
    face_path = json_path.parent / Path(uri).name
    if not face_path.is_file():
        raise FileNotFoundError(f"Face video not found: {face_path}")
    return face_path


def sample_interval(interval, every_n, label, source_action):
    start = int(interval["frame_start"])
    end = int(interval["frame_end"])
    rows = []
    for frame_idx in range(start, end + 1, every_n):
        rows.append(
            {
                "frame_idx": frame_idx,
                "label": label,
                "source_action": source_action,
                "interval_start": start,
                "interval_end": end,
                "interval_length": end - start + 1,
            }
        )
    return rows


def build_samples(openlabel: dict, close_threshold: int, fatigue_every_n: int, nonfatigue_every_n: int):
    actions = openlabel.get("actions", {})

    fatigue_rows = []
    nonfatigue_rows = []

    for action in actions.values():
        action_type = action.get("type", "")
        intervals = action.get("frame_intervals", [])

        if action_type == "eyes_state/open":
            for interval in intervals:
                nonfatigue_rows.extend(sample_interval(interval, nonfatigue_every_n, "non_fatigue", action_type))
        elif action_type == "eyes_state/close":
            for interval in intervals:
                length = int(interval["frame_end"]) - int(interval["frame_start"]) + 1
                if length >= close_threshold:
                    fatigue_rows.extend(sample_interval(interval, fatigue_every_n, "fatigue", action_type))
        elif action_type.startswith("yawning/"):
            for interval in intervals:
                fatigue_rows.extend(sample_interval(interval, fatigue_every_n, "fatigue", action_type))

    fatigue_rows.sort(key=lambda x: x["frame_idx"])
    nonfatigue_rows.sort(key=lambda x: x["frame_idx"])
    return {"fatigue": fatigue_rows, "non_fatigue": nonfatigue_rows}


def split_rows(rows, train_ratio: float, val_ratio: float):
    n = len(rows)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    train_end = max(1, int(n * train_ratio))
    val_count = max(1, int(n * val_ratio)) if n >= 3 else 0
    val_end = min(n, train_end + val_count)

    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]

    if n >= 3 and not val_rows:
        val_rows = test_rows[:1]
        test_rows = test_rows[1:]
    if n >= 3 and not test_rows:
        test_rows = val_rows[-1:]
        val_rows = val_rows[:-1]

    return {"train": train_rows, "val": val_rows, "test": test_rows}


def export_dataset(
    face_video: Path, split_rows_by_label: dict, output_dir: Path, jpg_quality: int, metadata_extra: dict
):
    all_rows = []
    frame_requests = {}

    for label, split_map in split_rows_by_label.items():
        for split, rows in split_map.items():
            class_dir = output_dir / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            for row in rows:
                filename = f"{face_video.stem}_f{row['frame_idx']:06d}.jpg"
                out_path = class_dir / filename
                export_row = {
                    **row,
                    **metadata_extra,
                    "split": split,
                    "image_path": str(out_path.resolve()),
                }
                all_rows.append(export_row)
                frame_requests[row["frame_idx"]] = out_path

    cap = cv2.VideoCapture(str(face_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open face video: {face_video}")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    for frame_idx in sorted(frame_requests):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {face_video}")
        out_path = frame_requests[frame_idx]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), frame, encode_params):
            raise RuntimeError(f"Failed to write image: {out_path}")
    cap.release()

    metadata_path = output_dir / "samples_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "label",
                "split",
                "frame_idx",
                "source_action",
                "interval_start",
                "interval_end",
                "interval_length",
                "json_path",
                "face_video",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "label": row["label"],
                    "split": row["split"],
                    "frame_idx": row["frame_idx"],
                    "source_action": row["source_action"],
                    "interval_start": row["interval_start"],
                    "interval_end": row["interval_end"],
                    "interval_length": row["interval_length"],
                    "json_path": row["json_path"],
                    "face_video": row["face_video"],
                }
            )

    return metadata_path, all_rows


def main():
    args = parse_args()
    validate_args(args)
    openlabel = load_openlabel(args.json)
    face_video = resolve_face_video(args.json, openlabel)
    samples = build_samples(openlabel, args.close_threshold, args.fatigue_every_n, args.nonfatigue_every_n)

    split_rows_by_label = {label: split_rows(rows, args.train_ratio, args.val_ratio) for label, rows in samples.items()}

    metadata_path, all_rows = export_dataset(
        face_video=face_video,
        split_rows_by_label=split_rows_by_label,
        output_dir=args.output_dir,
        jpg_quality=args.jpg_quality,
        metadata_extra={
            "json_path": str(args.json.resolve()),
            "face_video": str(face_video.resolve()),
        },
    )

    split_counts = defaultdict(Counter)
    source_counts = Counter()
    for row in all_rows:
        split_counts[row["split"]][row["label"]] += 1
        source_counts[(row["label"], row["source_action"])] += 1

    print(f"json: {args.json}")
    print(f"face_video: {face_video}")
    print(f"output_dir: {args.output_dir}")
    print(f"metadata: {metadata_path}")
    print(f"close_threshold: {args.close_threshold}")
    print(f"fatigue_every_n: {args.fatigue_every_n}")
    print(f"nonfatigue_every_n: {args.nonfatigue_every_n}")
    print()
    print("Counts by split")
    for split in ("train", "val", "test"):
        total = sum(split_counts[split].values())
        print(f"{split}: {total}")
        for label in ("fatigue", "non_fatigue"):
            print(f"  {label}: {split_counts[split][label]}")
    print()
    print("Counts by source action")
    for (label, action), count in sorted(source_counts.items()):
        print(f"{label} <- {action}: {count}")


if __name__ == "__main__":
    main()
