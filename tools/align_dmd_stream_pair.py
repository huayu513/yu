from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Align a DMD body/face video pair using frame_shift from OpenLABEL.")
    parser.add_argument("--json", required=True, help="Path to DMD OpenLABEL json.")
    parser.add_argument("--body-video", required=True, help="Path to body video.")
    parser.add_argument("--face-video", required=True, help="Path to face video.")
    parser.add_argument("--output-body", required=True, help="Output aligned body video path.")
    parser.add_argument("--output-face", required=True, help="Output aligned face video path.")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec, default: mp4v")
    return parser.parse_args()


def load_shifts(json_path: Path) -> tuple[int, int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))["openlabel"]["streams"]
    body_shift = int(data["body_camera"].get("stream_properties", {}).get("sync", {}).get("frame_shift", 0))
    face_shift = int(data["face_camera"].get("stream_properties", {}).get("sync", {}).get("frame_shift", 0))
    return body_shift, face_shift


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def main():
    args = parse_args()
    json_path = Path(args.json)
    body_video = Path(args.body_video)
    face_video = Path(args.face_video)
    output_body = Path(args.output_body)
    output_face = Path(args.output_face)

    body_shift, face_shift = load_shifts(json_path)

    # 同一“原始时间”下：
    # original = body_frame + body_shift = face_frame + face_shift
    # 所以需要把 shift 更小的那一路往后裁，最终让两路第 0 帧对应同一时刻。
    start_body = max(0, face_shift - body_shift)
    start_face = max(0, body_shift - face_shift)

    body_cap = open_video(body_video)
    face_cap = open_video(face_video)

    body_fps = body_cap.get(cv2.CAP_PROP_FPS) or 25.0
    face_fps = face_cap.get(cv2.CAP_PROP_FPS) or 25.0
    if abs(body_fps - face_fps) > 1e-3:
        raise RuntimeError(f"FPS mismatch: body={body_fps}, face={face_fps}")

    body_width = int(body_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    body_height = int(body_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    face_width = int(face_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    face_height = int(face_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    body_frames = int(body_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    face_frames = int(face_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    aligned_count = min(body_frames - start_body, face_frames - start_face)
    if aligned_count <= 0:
        raise RuntimeError("No overlapping frames after alignment.")

    output_body.parent.mkdir(parents=True, exist_ok=True)
    output_face.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    body_writer = cv2.VideoWriter(str(output_body), fourcc, body_fps, (body_width, body_height))
    face_writer = cv2.VideoWriter(str(output_face), fourcc, face_fps, (face_width, face_height))
    if not body_writer.isOpened() or not face_writer.isOpened():
        raise RuntimeError("Failed to create output writers.")

    print(
        f"body_shift={body_shift}, face_shift={face_shift}, "
        f"start_body={start_body}, start_face={start_face}, aligned_frames={aligned_count}"
    )

    for i in range(aligned_count):
        body_cap.set(cv2.CAP_PROP_POS_FRAMES, start_body + i)
        face_cap.set(cv2.CAP_PROP_POS_FRAMES, start_face + i)

        ok_body, body_frame = body_cap.read()
        ok_face, face_frame = face_cap.read()
        if not ok_body or body_frame is None or not ok_face or face_frame is None:
            break

        body_writer.write(body_frame)
        face_writer.write(face_frame)

    body_cap.release()
    face_cap.release()
    body_writer.release()
    face_writer.release()

    print(f"Aligned body video: {output_body}")
    print(f"Aligned face video: {output_face}")


if __name__ == "__main__":
    main()
