import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# 复用你现有 run.py 的启发式风险权重：
# 模型输出每个 c0-c9 的概率后，用“概率 * 风险权重”求期望风险。
DEFAULT_RISK_MAP = {
    "c0": 0.0,
    "c1": 1.0,
    "c2": 1.0,
    "c3": 1.0,
    "c4": 1.0,
    "c5": 0.6,
    "c6": 0.5,
    "c7": 0.9,
    "c8": 0.8,
    "c9": 0.5,
}

# DMD 的 OpenLABEL 标注里用的是这几个流名称，这里给命令行参数做一个简化映射。
STREAM_ALIAS = {
    "face": "face_camera",
    "body": "body_camera",
    "hands": "hands_camera",
}

# 这里只是一个“粗对应关系”提示，方便观察跨域泛化是否大致合理，
# 不是严格的一一标签映射。
DMD_TO_STATEFARM_HINT = {
    "safe_drive": "c0",
    "drinking": "c1",
    "radio": "c5",
    "talking_to_passenger": "c3/c4?",
    "reach_side": "c7/c8?",
    "unclassified": "?",
}

# 用于窗口里显示“aligned / mismatch”。
# 这是人为定义的粗规则，只适合做可视化参考，不适合拿来当正式指标。
DMD_MATCH_RULES = {
    "safe_drive": {"c0"},
    "drinking": {"c1"},
    "radio": {"c5"},
    "talking_to_passenger": {"c2", "c3", "c4"},
    "reach_side": {"c7", "c8"},
    "unclassified": set(),
}


def parse_args():
    # 这个脚本是命令行工具：输入模型、视频和可选标注，输出可视化视频。
    parser = argparse.ArgumentParser(
        description="Visualize a State Farm classifier running on DMD video."
    )
    parser.add_argument("--model", required=True, help="Path to State Farm classifier weights.")
    parser.add_argument("--video", required=True, help="Path to DMD video.")
    parser.add_argument("--output", required=True, help="Path to output visualization video.")
    parser.add_argument("--ann-json", default="", help="Optional DMD OpenLABEL annotation json.")
    parser.add_argument(
        "--stream",
        default="body",
        choices=["face", "body", "hands"],
        help="DMD stream name used for frame alignment when ann-json is provided.",
    )
    parser.add_argument(
        "--label-prefix",
        default="driver_actions",
        help="DMD label prefix to visualize from the annotation json.",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=3,
        help="Run model every N frames and reuse the last prediction in between.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Moving-average window size for probability smoothing.",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.6,
        help="Risk score threshold used to count alerts.",
    )
    parser.add_argument(
        "--alert-patience",
        type=int,
        default=5,
        help="How many consecutive risky sampled frames trigger alarm.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="How many top predictions to display.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on total processed frames. 0 means full video.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the visualization in a cv2 window while processing.",
    )
    parser.add_argument(
        "--window-name",
        default="State Farm on DMD",
        help="cv2 window title used with --show.",
    )
    return parser.parse_args()


def load_dmd_frame_labels(json_path: Path, stream: str, label_prefix: str):
    # 从 DMD 的 OpenLABEL json 中读取某一类标签（例如 driver_actions），
    # 并展开成“frame_idx -> label集合”的形式，便于逐帧对齐显示。
    with json_path.open("r", encoding="utf-8") as f:
        openlabel = json.load(f)["openlabel"]

    stream_key = STREAM_ALIAS[stream]
    stream_info = openlabel["streams"][stream_key]
    # DMD 的不同视角之间存在 frame_shift，这里做一次对齐。
    frame_shift = int(stream_info.get("stream_properties", {}).get("sync", {}).get("frame_shift", 0))

    frame_to_labels = {}
    for action in openlabel.get("actions", {}).values():
        action_type = action.get("type", "")
        if not action_type.startswith(label_prefix + "/"):
            continue

        label = action_type.split("/")[-1]
        for interval in action.get("frame_intervals", []):
            start = max(0, int(interval["frame_start"]) - frame_shift)
            end = max(start, int(interval["frame_end"]) - frame_shift)
            for frame_idx in range(start, end + 1):
                frame_to_labels.setdefault(frame_idx, set()).add(label)

    return frame_to_labels


def format_dmd_label(frame_labels, frame_idx):
    # 当前帧没有 DMD 标签时直接显示 N/A；
    # 如果一帧落入多个动作区间，就把它标成 ambiguous。
    labels = frame_labels.get(frame_idx)
    if not labels:
        return "N/A", "N/A"
    if len(labels) == 1:
        label = next(iter(labels))
        return label, DMD_TO_STATEFARM_HINT.get(label, "?")
    merged = "|".join(sorted(labels))
    return merged, "ambiguous"


def judge_alignment(pred_class, dmd_label):
    # 这里只做“粗对齐”判断，帮助你肉眼看模型是否大致预测到了同一类行为。
    if dmd_label in {"N/A", ""}:
        return "unknown", (180, 180, 180)
    if "|" in dmd_label:
        return "ambiguous", (0, 255, 255)

    valid_preds = DMD_MATCH_RULES.get(dmd_label, set())
    if not valid_preds:
        return "no_rule", (180, 180, 180)
    if pred_class in valid_preds:
        return "aligned", (0, 255, 0)
    return "mismatch", (0, 0, 255)


def get_topk(avg_probs, class_names, topk):
    # 取平滑后概率最高的 top-k，显示在窗口右侧条形图里。
    top_indices = np.argsort(avg_probs)[::-1][:topk]
    return [(class_names[idx], float(avg_probs[idx])) for idx in top_indices]


def draw_overlay(frame, lines, topk_items, risk_score, alarm):
    # 在视频左上角叠加半透明信息面板。
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (620, 270), (0, 0, 0), -1)
    alpha = 0.58
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    y = 50
    for text, color, scale in lines:
        cv2.putText(frame, text, (35, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += int(34 * scale / 0.75)

    bar_x = 35
    bar_y = 180
    bar_w = 250
    bar_h = 18
    for idx, (name, prob) in enumerate(topk_items):
        yy = bar_y + idx * 28
        cv2.rectangle(frame, (bar_x, yy), (bar_x + bar_w, yy + bar_h), (80, 80, 80), 1)
        # 用条形长度直观表示 top-k 概率大小。
        cv2.rectangle(frame, (bar_x, yy), (bar_x + int(bar_w * prob), yy + bar_h), (60, 180, 75), -1)
        cv2.putText(
            frame,
            f"{name}: {prob:.3f}",
            (bar_x + bar_w + 12, yy + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    risk_text = f"Risk {risk_score:.3f}"
    risk_color = (0, 255, 255) if risk_score < 0.6 else (0, 165, 255)
    cv2.putText(frame, risk_text, (35, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2, cv2.LINE_AA)

    if alarm:
        cv2.putText(
            frame,
            "ALARM",
            (frame.shape[1] - 180, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    return frame


def main():
    args = parse_args()
    model = YOLO(args.model)
    class_names = [model.names[i] for i in sorted(model.names)]
    # 如果模型类名和 risk_map 不完全一致，未覆盖到的类默认给 0.5 风险。
    risk_map = {name: DEFAULT_RISK_MAP.get(name, 0.5) for name in class_names}

    frame_labels = {}
    if args.ann_json:
        frame_labels = load_dmd_frame_labels(Path(args.ann_json), args.stream, args.label_prefix)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {output_path}")

    # 用滑动窗口平滑概率，减少跨帧抖动。
    prob_window = deque(maxlen=args.window_size)
    alert_count = 0
    frame_id = -1
    sampled_count = 0
    current_pred = "warming_up"
    current_conf = 0.0
    current_topk = []
    current_risk = 0.0
    current_alarm = False

    if args.show:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1
        if args.max_frames and frame_id >= args.max_frames:
            break

        if frame_id % args.sample_interval == 0:
            # 为了提速，不是每一帧都推理，而是隔 sample_interval 帧推理一次。
            results = model(frame, verbose=False)
            probs = results[0].probs.data.cpu().numpy()
            prob_window.append(probs)
            sampled_count += 1

            if len(prob_window) > 0:
                avg_probs = np.mean(np.array(prob_window), axis=0)
                pred_idx = int(np.argmax(avg_probs))
                current_pred = class_names[pred_idx]
                current_conf = float(avg_probs[pred_idx])
                current_topk = get_topk(avg_probs, class_names, args.topk)
                # 风险分数是“当前类别分布下的期望风险”，不是单一类别的分数。
                current_risk = sum(avg_probs[i] * risk_map[class_names[i]] for i in range(len(class_names)))
                # 只有连续多次高风险才触发报警，避免单帧误报。
                if current_risk >= args.alert_threshold:
                    alert_count += 1
                else:
                    alert_count = 0
                current_alarm = alert_count >= args.alert_patience

        dmd_label, coarse_hint = format_dmd_label(frame_labels, frame_id)
        alignment_text, alignment_color = judge_alignment(current_pred, dmd_label)
        lines = [
            (f"Frame {frame_id}", (255, 255, 255), 0.75),
            (f"StateFarm pred: {current_pred} ({current_conf:.3f})", (0, 255, 0), 0.68),
            (f"DMD label: {dmd_label}", (255, 255, 0), 0.68),
            (f"Coarse hint: {coarse_hint}", (200, 200, 255), 0.6),
            (f"Alignment: {alignment_text}", alignment_color, 0.6),
            (
                f"Sample every {args.sample_interval}f | Window {args.window_size} | Alerts {alert_count}",
                (220, 220, 220),
                0.52,
            ),
        ]
        frame = draw_overlay(frame, lines, current_topk, current_risk, current_alarm)
        writer.write(frame)

        if args.show:
            # 本地调试时可以边处理边看窗口；按 q 或 Esc 可提前退出。
            cv2.imshow(args.window_name, frame)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key in {27, ord("q")}:
                print("Stopped by user.")
                break

        if frame_id % 300 == 0:
            # 每隔 300 帧打印一次快照，方便在终端快速观察运行状态。
            print(
                f"frame={frame_id} pred={current_pred} conf={current_conf:.3f} "
                f"risk={current_risk:.3f} dmd={dmd_label} alarm={current_alarm}"
            )

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Saved visualization to: {output_path}")
    print(f"Processed frames: {frame_id + 1}")
    print(f"Sampled frames: {sampled_count}")


if __name__ == "__main__":
    main()

# python tools\visualize_statefarm_on_dmd.py `
#   --model "runs\classify\train4\weights\best.pt" `
#   --video "dmd-dataset-mini-sample-gA-3-s1\dmd\gA\3\s1\gA_3_s1_2019-03-08T10;27;38+01;00_rgb_body.mp4" `
#   --ann-json "dmd-dataset-mini-sample-gA-3-s1\dmd\gA\3\s1\gA_3_s1_2019-03-08T10;27;38+01;00_rgb_ann_distraction.json" `
#   --stream body `
#   --label-prefix driver_actions `
#   --output "runs\viz\statefarm_on_dmd_body.mp4" `
#   --show

