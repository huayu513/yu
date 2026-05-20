import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2


LABEL_CANDIDATES = [
    "label",
    "class",
    "class_name",
    "category",
    "state",
    "driver_state",
]
SPLIT_CANDIDATES = ["split", "subset", "phase"]
VIDEO_CANDIDATES = [
    "video",
    "video_path",
    "video_name",
    "filename",
    "file_name",
    "source",
]
FRAME_CANDIDATES = ["frame", "frame_idx", "frame_id", "image_index", "index"]
START_FRAME_CANDIDATES = ["start_frame", "frame_start", "begin_frame"]
END_FRAME_CANDIDATES = ["end_frame", "frame_end", "stop_frame"]
START_TIME_CANDIDATES = ["start_time", "time_start", "begin_time"]
END_TIME_CANDIDATES = ["end_time", "time_end", "stop_time"]
BOX_CANDIDATES = ["bbox", "box", "rect"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert video + JSON annotations to a YOLO classification dataset."
    )
    parser.add_argument("--json", required=True, help="Annotation JSON path.")
    parser.add_argument(
        "--videos-dir",
        required=True,
        help="Directory containing the source videos.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output dataset directory. Result format: split/class_name/*.jpg",
    )
    parser.add_argument(
        "--default-split",
        default="train",
        choices=["train", "val", "test"],
        help="Split to use when JSON does not provide one.",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=3,
        help="Save one frame every N frames inside each labeled span.",
    )
    parser.add_argument(
        "--limit-per-label",
        type=int,
        default=0,
        help="Optional cap for exported images of each class. 0 means no limit.",
    )
    parser.add_argument(
        "--crop-box",
        action="store_true",
        help="Crop the frame using bbox fields if the JSON contains them.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality from 0 to 100.",
    )
    return parser.parse_args()


def load_annotations(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("annotations", "items", "labels", "data", "samples"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    raise ValueError("Unsupported JSON structure. Expected a list or dict containing a list.")


def pick_first(mapping, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return default


def normalize_label(label):
    return str(label).strip().replace("/", "_").replace("\\", "_")


def resolve_video_path(raw_value, videos_dir: Path):
    raw_path = Path(str(raw_value))
    if raw_path.exists():
        return raw_path

    candidate = videos_dir / raw_path.name
    if candidate.exists():
        return candidate

    stem = raw_path.stem.lower()
    matches = [p for p in videos_dir.rglob("*") if p.is_file() and p.stem.lower() == stem]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Video not found for annotation value: {raw_value}")


def parse_bbox(item):
    bbox = pick_first(item, BOX_CANDIDATES)
    if bbox is None:
        return None

    if isinstance(bbox, dict):
        x = bbox.get("x", bbox.get("left"))
        y = bbox.get("y", bbox.get("top"))
        w = bbox.get("w", bbox.get("width"))
        h = bbox.get("h", bbox.get("height"))
        if None not in (x, y, w, h):
            return int(x), int(y), int(w), int(h)

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return tuple(int(v) for v in bbox[:4])

    return None


def frame_range_from_item(item, fps: float):
    single = pick_first(item, FRAME_CANDIDATES)
    if single is not None:
        idx = int(single)
        return idx, idx

    start_frame = pick_first(item, START_FRAME_CANDIDATES)
    end_frame = pick_first(item, END_FRAME_CANDIDATES)
    if start_frame is not None and end_frame is not None:
        return int(start_frame), int(end_frame)

    start_time = pick_first(item, START_TIME_CANDIDATES)
    end_time = pick_first(item, END_TIME_CANDIDATES)
    if start_time is not None and end_time is not None:
        return int(float(start_time) * fps), int(float(end_time) * fps)

    raise ValueError(
        "Annotation is missing frame info. Need frame/frame_idx or "
        "start_frame+end_frame or start_time+end_time."
    )


def clamp_box(box, width, height):
    x, y, w, h = box
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def export_annotation(item, args, counters):
    label = normalize_label(pick_first(item, LABEL_CANDIDATES))
    if not label or label == "None":
        raise ValueError("Annotation is missing label/class.")

    split = str(pick_first(item, SPLIT_CANDIDATES, args.default_split)).strip().lower()
    if split not in {"train", "val", "test"}:
        split = args.default_split

    raw_video = pick_first(item, VIDEO_CANDIDATES)
    if raw_video is None:
        raise ValueError("Annotation is missing video path/name.")

    video_path = resolve_video_path(raw_video, Path(args.videos_dir))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame, end_frame = frame_range_from_item(item, fps)
    if end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame

    dest_dir = Path(args.output_dir) / split / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    bbox = parse_bbox(item) if args.crop_box else None
    total_written = 0
    video_stem = video_path.stem

    for frame_idx in range(start_frame, end_frame + 1, args.every_n_frames):
        if args.limit_per_label and counters[(split, label)] >= args.limit_per_label:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        if bbox is not None:
            h, w = frame.shape[:2]
            x, y, bw, bh = clamp_box(bbox, w, h)
            frame = frame[y : y + bh, x : x + bw]
            if frame.size == 0:
                continue

        out_path = dest_dir / f"{video_stem}_f{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])
        counters[(split, label)] += 1
        total_written += 1

    cap.release()
    return split, label, video_path.name, total_written


def main():
    args = parse_args()
    annotations = load_annotations(Path(args.json))
    counters = defaultdict(int)
    summary = defaultdict(int)

    for idx, item in enumerate(annotations, start=1):
        try:
            split, label, video_name, written = export_annotation(item, args, counters)
            summary[(split, label)] += written
            print(
                f"[{idx}/{len(annotations)}] {video_name} -> {split}/{label}, "
                f"exported {written} images"
            )
        except Exception as exc:
            print(f"[{idx}/{len(annotations)}] skipped: {exc}")

    print("\nSummary")
    for (split, label), count in sorted(summary.items()):
        print(f"{split}/{label}: {count}")


if __name__ == "__main__":
    main()
