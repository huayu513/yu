import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
from convert_dmd_openlabel_to_yolo_cls import (
    STREAM_ALIAS,
    collect_frame_labels,
    load_openlabel,
    resolve_video_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample single frames from DMD action segments for manual inspection or single-frame testing."
    )
    parser.add_argument("--json", required=True, help="Path to *_ann_*.json.")
    parser.add_argument("--output-dir", required=True, help="Output folder for sampled images.")
    parser.add_argument(
        "--stream",
        default="body",
        choices=["face", "body", "hands"],
        help="Which DMD RGB stream to sample from.",
    )
    parser.add_argument(
        "--label-prefix",
        default="driver_actions",
        help="Only sample labels whose OpenLABEL action type starts with this prefix.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Maximum number of sampled images to save for each class.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality from 0 to 100.",
    )
    return parser.parse_args()


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


def main():
    args = parse_args()
    json_path = Path(args.json)
    openlabel = load_openlabel(json_path)
    stream_key = STREAM_ALIAS[args.stream]
    video_path = resolve_video_path(json_path, openlabel, stream_key)
    frame_to_labels, raw_counts, frame_shift = collect_frame_labels(openlabel, args.label_prefix, stream_key)

    # 只保留单标签帧，避免一张图对应多个动作标签。
    label_to_frames = defaultdict(list)
    ambiguous_count = 0
    for frame_idx, labels in frame_to_labels.items():
        if len(labels) != 1:
            ambiguous_count += 1
            continue
        label = next(iter(labels))
        label_to_frames[label].append(frame_idx)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "samples_metadata.csv"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    rows = []
    exported_counts = {}
    video_stem = video_path.stem

    for label, frames in sorted(label_to_frames.items()):
        frames = sorted(frames)
        sampled_frames = evenly_sample(frames, args.samples_per_class)
        label_dir = output_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for frame_idx in sampled_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            out_name = f"{video_stem}_f{frame_idx:06d}.jpg"
            out_path = label_dir / out_name
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])

            rows.append(
                {
                    "image_path": str(out_path.resolve()),
                    "true_label": label,
                    "frame_idx": frame_idx,
                    "video_name": video_path.name,
                    "json_name": json_path.name,
                    "stream": args.stream,
                    "label_prefix": args.label_prefix,
                    "frame_shift": frame_shift,
                }
            )
            written += 1

        exported_counts[label] = written

    cap.release()

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "true_label",
                "frame_idx",
                "video_name",
                "json_name",
                "stream",
                "label_prefix",
                "frame_shift",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"json: {json_path.name}")
    print(f"video: {video_path.name}")
    print(f"stream: {stream_key}")
    print(f"label_prefix: {args.label_prefix}")
    print(f"frame_shift: {frame_shift}")
    print(f"ambiguous frames skipped: {ambiguous_count}")
    print(f"metadata: {metadata_path}")

    print("\nRaw frame counts")
    for label, count in sorted(raw_counts.items()):
        print(f"{label}: {count}")

    print("\nExported samples")
    for label, count in sorted(exported_counts.items()):
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()

# python tools\sample_dmd_frames_by_action.py `
#   --json "dmd-dataset-mini-sample-gA-3-s1\dmd\gA\3\s1\gA_3_s1_2019-03-08T10;27;38+01;00_rgb_ann_distraction.json" `
#   --output-dir "runs\dmd_samples\driver_actions_body_100" `
#   --stream body `
#   --label-prefix driver_actions `
#   --samples-per-class 100
