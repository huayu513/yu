import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2

STREAM_ALIAS = {
    "face": "face_camera",
    "body": "body_camera",
    "hands": "hands_camera",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DMD OpenLABEL annotations to a YOLO classification dataset.")
    parser.add_argument("--json", required=True, help="Path to *_ann_distraction.json.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output dataset root. Result format: split/class_name/*.jpg",
    )
    parser.add_argument(
        "--stream",
        default="body",
        choices=["face", "body", "hands"],
        help="Which RGB camera stream to export.",
    )
    parser.add_argument(
        "--label-prefix",
        default="driver_actions",
        help="Only export action types that start with this prefix.",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=10,
        help="Save one frame every N labeled frames.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional cap for each class. 0 means no cap.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality from 0 to 100.",
    )
    return parser.parse_args()


def load_openlabel(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["openlabel"]


def choose_split(stem: str, frame_idx: int, val_ratio: float, test_ratio: float):
    key = f"{stem}_{frame_idx}".encode()
    score = int(hashlib.md5(key).hexdigest(), 16) % 10000 / 10000.0
    if score < test_ratio:
        return "test"
    if score < test_ratio + val_ratio:
        return "val"
    return "train"


def sanitize_label(action_type: str):
    return action_type.split("/")[-1].strip().replace(" ", "_")


def collect_frame_labels(openlabel, label_prefix: str, stream_key: str):
    stream_info = openlabel["streams"][stream_key]
    frame_shift = stream_info.get("stream_properties", {}).get("sync", {}).get("frame_shift", 0)

    frame_to_labels = defaultdict(set)
    class_counts = defaultdict(int)

    for action in openlabel.get("actions", {}).values():
        action_type = action.get("type", "")
        if not action_type.startswith(label_prefix + "/"):
            continue

        label = sanitize_label(action_type)
        for interval in action.get("frame_intervals", []):
            start = int(interval["frame_start"]) - int(frame_shift)
            end = int(interval["frame_end"]) - int(frame_shift)
            if end < 0:
                continue
            start = max(0, start)
            end = max(start, end)
            for frame_idx in range(start, end + 1):
                frame_to_labels[frame_idx].add(label)
                class_counts[label] += 1

    return frame_to_labels, dict(class_counts), int(frame_shift)


def resolve_video_path(json_path: Path, openlabel, stream_key: str):
    stream_uri = openlabel["streams"][stream_key]["uri"]
    candidate = json_path.parent / Path(stream_uri).name
    if candidate.exists():
        return candidate

    candidate = json_path.parent / Path(stream_uri)
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Video not found for stream {stream_key}: {stream_uri}")


def export_dataset_from_json(
    json_path,
    output_dir,
    stream,
    label_prefix,
    every_n_frames,
    val_ratio,
    test_ratio,
    max_per_class,
    jpg_quality,
):
    json_path = Path(json_path)
    openlabel = load_openlabel(json_path)
    stream_key = STREAM_ALIAS[stream]
    video_path = resolve_video_path(json_path, openlabel, stream_key)
    frame_to_labels, raw_counts, frame_shift = collect_frame_labels(openlabel, label_prefix, stream_key)

    unique_frames = {idx: next(iter(labels)) for idx, labels in frame_to_labels.items() if len(labels) == 1}
    ambiguous_frames = {idx: labels for idx, labels in frame_to_labels.items() if len(labels) > 1}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_stem = video_path.stem
    output_root = Path(output_dir)
    exported_counts = defaultdict(int)
    sampled_frames = defaultdict(int)

    for frame_idx in sorted(unique_frames):
        label = unique_frames[frame_idx]
        if frame_idx % every_n_frames != 0:
            continue
        if max_per_class and exported_counts[label] >= max_per_class:
            continue

        split = choose_split(video_stem, frame_idx, val_ratio, test_ratio)
        dest_dir = output_root / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        out_path = dest_dir / f"{video_stem}_f{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        exported_counts[label] += 1
        sampled_frames[(split, label)] += 1

    cap.release()

    return {
        "json_name": json_path.name,
        "video_name": video_path.name,
        "stream_key": stream_key,
        "label_prefix": label_prefix,
        "frame_shift": frame_shift,
        "all_labeled_frames": len(frame_to_labels),
        "unique_frames": len(unique_frames),
        "ambiguous_frames": len(ambiguous_frames),
        "raw_counts": dict(raw_counts),
        "sampled_frames": dict(sampled_frames),
    }


def print_summary(summary):
    print(f"json: {summary['json_name']}")
    print(f"video: {summary['video_name']}")
    print(f"stream: {summary['stream_key']}")
    print(f"label_prefix: {summary['label_prefix']}")
    print(f"frame_shift: {summary['frame_shift']}")
    print(f"all labeled frames: {summary['all_labeled_frames']}")
    print(f"unique-label frames: {summary['unique_frames']}")
    print(f"ambiguous frames skipped: {summary['ambiguous_frames']}")

    print("\nClasses found")
    for label in sorted(summary["raw_counts"]):
        print(label)

    print("\nRaw frame counts")
    for label, count in sorted(summary["raw_counts"].items()):
        print(f"{label}: {count}")

    print("\nExported images")
    for (split, label), count in sorted(summary["sampled_frames"].items()):
        print(f"{split}/{label}: {count}")


def export_dataset(args):
    summary = export_dataset_from_json(
        json_path=args.json,
        output_dir=args.output_dir,
        stream=args.stream,
        label_prefix=args.label_prefix,
        every_n_frames=args.every_n_frames,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_per_class=args.max_per_class,
        jpg_quality=args.jpg_quality,
    )
    print_summary(summary)


def main():
    args = parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1.")
    export_dataset(args)


if __name__ == "__main__":
    main()
