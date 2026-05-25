from __future__ import annotations

import argparse
import csv
import json
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# 避免 Ultralytics 把配置写到系统默认目录，统一落到项目内。
if "YOLO_CONFIG_DIR" not in os.environ:
    os.environ["YOLO_CONFIG_DIR"] = str((Path(__file__).resolve().parent / ".yolo_config").resolve())

from train_multimodal_v1 import MultiModalDriverNet, build_transforms, load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal body+face inference on paired videos.")
    parser.add_argument(
        "--config", default="train_multimodal_v14.yaml", help="Training config used to build the model."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to multimodal best.pt/last.pt checkpoint.")
    parser.add_argument("--body-video", required=True, help="Path to body camera video.")
    parser.add_argument("--face-video", required=True, help="Path to face camera video.")
    parser.add_argument("--output-video", default="", help="Optional path to save visualization video.")
    parser.add_argument("--output-csv", default="", help="Optional path to save per-frame predictions.")
    parser.add_argument("--sample-interval", type=int, default=3, help="Run model every N body frames.")
    parser.add_argument("--window-size", type=int, default=8, help="Probability smoothing window size.")
    parser.add_argument(
        "--body-frame-shift",
        type=int,
        default=0,
        help="Sync shift used by the body stream. original_frame = body_frame + body_shift.",
    )
    parser.add_argument(
        "--face-frame-shift",
        type=int,
        default=0,
        help="Sync shift used by the face stream. face_frame = original_frame - face_shift.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit on processed body frames.")
    parser.add_argument("--show", action="store_true", help="Display a live cv2 window while processing.")
    parser.add_argument("--window-name", default="Multimodal Driver Inference", help="Window title used with --show.")
    parser.add_argument("--device", default="", help="Override device from config, e.g. cpu or 0.")
    return parser.parse_args()


def resolve_device(device_arg: str, cfg_device) -> torch.device:
    requested = device_arg if device_arg != "" else str(cfg_device)
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        if requested.isdigit():
            return torch.device(f"cuda:{requested}")
        return torch.device("cuda")
    return torch.device("cpu")


def to_tensor(transform, frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


class MultiModalVideoInferencer:
    """Small reusable inference wrapper for future GUI integration."""

    def __init__(self, config_path: str | Path, checkpoint_path: str | Path, device_override: str = "") -> None:
        self.cfg = load_yaml(config_path)
        self.dataset_root = Path(self.cfg["data_root"])
        self.metadata = json.loads((self.dataset_root / "metadata.json").read_text(encoding="utf-8"))
        self.distraction_classes = self.metadata["distraction_classes"]
        self.fatigue_classes = self.metadata["fatigue_classes"]
        self.transform = build_transforms(self.cfg, "val")
        self.device = resolve_device(device_override, self.cfg["device"])

        self.model = MultiModalDriverNet(
            body_weight=self.cfg["body_weight"],
            face_weight=self.cfg["face_weight"],
            num_distraction_classes=len(self.distraction_classes),
            num_fatigue_classes=len(self.fatigue_classes),
            proj_dim=int(self.cfg["proj_dim"]),
            fusion_dim=int(self.cfg["fusion_dim"]),
            task_fusion_dim=int(self.cfg.get("task_fusion_dim", 0)),
            task_head_dim=int(self.cfg.get("task_head_dim", 0)),
            gate_hidden_dim=int(self.cfg.get("gate_hidden_dim", 0)),
            aux_head_dim=int(self.cfg.get("aux_head_dim", 0)),
            fatigue_face_dominant=bool(self.cfg.get("fatigue_face_dominant", False)),
            task_interaction_dim=int(self.cfg.get("task_interaction_dim", 0)),
            task_attention_heads=int(self.cfg.get("task_attention_heads", 0)),
            fatigue_refine_dim=int(self.cfg.get("fatigue_refine_dim", 0)),
            fatigue_residual_dim=int(self.cfg.get("fatigue_residual_dim", 0)),
            distraction_residual_dim=int(self.cfg.get("distraction_residual_dim", 0)),
            fatigue_prior_fusion=bool(self.cfg.get("fatigue_prior_fusion", False)),
            temporal_face_dim=int(self.cfg.get("temporal_face_dim", 0)),
            dropout=float(self.cfg["dropout"]),
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model = self.model.to(self.device).eval()

    def predict(self, body_frame: np.ndarray, face_frame: np.ndarray) -> dict[str, object]:
        body_tensor = to_tensor(self.transform, body_frame, self.device)
        face_tensor = to_tensor(self.transform, face_frame, self.device)
        with torch.no_grad():
            outputs = self.model(body_tensor, face_tensor)
            d_probs = torch.softmax(outputs["distraction_logits"], dim=1)[0].cpu().numpy()
            f_probs = torch.softmax(outputs["fatigue_logits"], dim=1)[0].cpu().numpy()
        return {
            "distraction_probs": d_probs,
            "fatigue_probs": f_probs,
        }


def open_video(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def get_smoothed_prediction(prob_window: deque[np.ndarray], class_names: list[str]) -> tuple[str, float, np.ndarray]:
    avg_probs = np.mean(np.array(prob_window), axis=0)
    pred_idx = int(np.argmax(avg_probs))
    return class_names[pred_idx], float(avg_probs[pred_idx]), avg_probs


def draw_prob_lines(
    canvas: np.ndarray,
    x: int,
    y: int,
    title: str,
    class_names: list[str],
    probs: np.ndarray,
    color: tuple[int, int, int],
) -> int:
    cv2.putText(canvas, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    y += 28
    top_indices = np.argsort(probs)[::-1][:2]
    for idx in top_indices:
        name = class_names[idx]
        prob = float(probs[idx])
        cv2.putText(
            canvas, f"{name}: {prob:.3f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA
        )
        y += 24
    return y


def render_side_by_side(
    body_frame: np.ndarray,
    face_frame: np.ndarray,
    body_frame_idx: int,
    face_frame_idx: int,
    distraction_pred: str,
    distraction_conf: float,
    distraction_probs: np.ndarray,
    distraction_classes: list[str],
    fatigue_pred: str,
    fatigue_conf: float,
    fatigue_probs: np.ndarray,
    fatigue_classes: list[str],
) -> np.ndarray:
    body_h, body_w = body_frame.shape[:2]
    face_h, face_w = face_frame.shape[:2]
    width = max(body_w, face_w)
    content_h = body_h + face_h

    # 上下布局更适合 GUI，不会把画面横向拉得太长。
    canvas = np.zeros((content_h + 160, width, 3), dtype=np.uint8)
    canvas[:body_h, :body_w] = body_frame
    canvas[body_h : body_h + face_h, :face_w] = face_frame

    cv2.putText(canvas, "Body View", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Face View",
        (20, body_h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Body frame: {body_frame_idx}",
        (20, content_h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Face frame: {face_frame_idx}",
        (width // 2 + 20, content_h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )

    y_left = content_h + 65
    cv2.putText(
        canvas,
        f"Distraction: {distraction_pred} ({distraction_conf:.3f})",
        (20, y_left),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    draw_prob_lines(canvas, 20, y_left + 24, "Top distraction", distraction_classes, distraction_probs, (0, 255, 0))

    y_right = content_h + 65
    cv2.putText(
        canvas,
        f"Fatigue: {fatigue_pred} ({fatigue_conf:.3f})",
        (width // 2 + 20, y_right),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 180, 0),
        2,
        cv2.LINE_AA,
    )
    draw_prob_lines(
        canvas,
        width // 2 + 20,
        y_right + 24,
        "Top fatigue",
        fatigue_classes,
        fatigue_probs,
        (255, 180, 0),
    )
    return canvas


def main():
    args = parse_args()
    inferencer = MultiModalVideoInferencer(args.config, args.checkpoint, args.device)

    body_cap = open_video(args.body_video)
    face_cap = open_video(args.face_video)

    fps = body_cap.get(cv2.CAP_PROP_FPS) or 25.0
    body_width = int(body_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    body_height = int(body_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    face_width = int(face_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    face_height = int(face_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = max(body_width, face_width)
    out_height = body_height + face_height + 160

    writer = None
    if args.output_video:
        output_video = Path(args.output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (out_width, out_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create output video: {output_video}")

    csv_file = None
    csv_writer = None
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = output_csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "body_frame_idx",
                "face_frame_idx",
                "distraction_pred",
                "distraction_conf",
                "fatigue_pred",
                "fatigue_conf",
            ],
        )
        csv_writer.writeheader()

    distraction_window: deque[np.ndarray] = deque(maxlen=args.window_size)
    fatigue_window: deque[np.ndarray] = deque(maxlen=args.window_size)

    current_distraction_pred = "warming_up"
    current_distraction_conf = 0.0
    current_distraction_probs = np.zeros(len(inferencer.distraction_classes), dtype=np.float32)
    current_fatigue_pred = "warming_up"
    current_fatigue_conf = 0.0
    current_fatigue_probs = np.zeros(len(inferencer.fatigue_classes), dtype=np.float32)

    body_frame_idx = -1
    if args.show:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    while True:
        ok, body_frame = body_cap.read()
        if not ok:
            break
        body_frame_idx += 1
        if args.max_frames and body_frame_idx >= args.max_frames:
            break

        original_frame_idx = body_frame_idx + args.body_frame_shift
        face_frame_idx = original_frame_idx - args.face_frame_shift
        if face_frame_idx < 0:
            continue

        face_frame = read_frame_at(face_cap, face_frame_idx)
        if face_frame is None:
            break

        if body_frame_idx % args.sample_interval == 0:
            preds = inferencer.predict(body_frame, face_frame)
            distraction_window.append(preds["distraction_probs"])
            fatigue_window.append(preds["fatigue_probs"])

            current_distraction_pred, current_distraction_conf, current_distraction_probs = get_smoothed_prediction(
                distraction_window, inferencer.distraction_classes
            )
            current_fatigue_pred, current_fatigue_conf, current_fatigue_probs = get_smoothed_prediction(
                fatigue_window, inferencer.fatigue_classes
            )

        canvas = render_side_by_side(
            body_frame=body_frame,
            face_frame=face_frame,
            body_frame_idx=body_frame_idx,
            face_frame_idx=face_frame_idx,
            distraction_pred=current_distraction_pred,
            distraction_conf=current_distraction_conf,
            distraction_probs=current_distraction_probs,
            distraction_classes=inferencer.distraction_classes,
            fatigue_pred=current_fatigue_pred,
            fatigue_conf=current_fatigue_conf,
            fatigue_probs=current_fatigue_probs,
            fatigue_classes=inferencer.fatigue_classes,
        )

        if writer is not None:
            writer.write(canvas)
        if csv_writer is not None:
            csv_writer.writerow(
                {
                    "body_frame_idx": body_frame_idx,
                    "face_frame_idx": face_frame_idx,
                    "distraction_pred": current_distraction_pred,
                    "distraction_conf": f"{current_distraction_conf:.6f}",
                    "fatigue_pred": current_fatigue_pred,
                    "fatigue_conf": f"{current_fatigue_conf:.6f}",
                }
            )

        if args.show:
            cv2.imshow(args.window_name, canvas)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key in {27, ord("q")}:
                break

        if body_frame_idx % 120 == 0:
            print(
                f"body={body_frame_idx} face={face_frame_idx} "
                f"distraction={current_distraction_pred}({current_distraction_conf:.3f}) "
                f"fatigue={current_fatigue_pred}({current_fatigue_conf:.3f})"
            )

    body_cap.release()
    face_cap.release()
    if writer is not None:
        writer.release()
    if csv_file is not None:
        csv_file.close()
    if args.show:
        cv2.destroyAllWindows()

    print("Inference complete.")
    if args.output_video:
        print(f"Saved video to: {args.output_video}")
    if args.output_csv:
        print(f"Saved csv to: {args.output_csv}")


if __name__ == "__main__":
    main()
