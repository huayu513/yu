import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO


DISTRACTION_4_PRESET = {
    "0": "safe_drive",
    "5": "radio",
    "6": "drinking",
    "9": "talking_to_passenger",
    "c0": "safe_drive",
    "c5": "radio",
    "c6": "drinking",
    "c9": "talking_to_passenger",
    0: "safe_drive",
    5: "radio",
    6: "drinking",
    9: "talking_to_passenger",
    "safe_drive": "safe_drive",
    "radio": "radio",
    "drinking": "drinking",
    "talking_to_passenger": "talking_to_passenger",
}

DISTRACTION_5_PRESET = {
    **DISTRACTION_4_PRESET,
    "1": "phone_use",
    "2": "phone_use",
    "3": "phone_use",
    "4": "phone_use",
    "c1": "phone_use",
    "c2": "phone_use",
    "c3": "phone_use",
    "c4": "phone_use",
    1: "phone_use",
    2: "phone_use",
    3: "phone_use",
    4: "phone_use",
    "phone_use": "phone_use",
}

FATIGUE_2_PRESET = {
    "0": "fatigue",
    "1": "non_fatigue",
    0: "fatigue",
    1: "non_fatigue",
    "fatigue": "fatigue",
    "non_fatigue": "non_fatigue",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO classify model and export accuracy/precision/recall/F1 metrics."
    )
    parser.add_argument("--model", required=True, help="Path to classifier weights.")
    parser.add_argument("--samples-dir", required=True, help="Directory containing evaluation samples.")
    parser.add_argument("--topk", type=int, default=3, help="How many top predictions to save in details CSV.")
    parser.add_argument("--batch", type=int, default=32, help="Prediction batch size.")
    parser.add_argument("--device", default=None, help="Optional inference device, e.g. cpu or 0.")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional inference image size override.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference when running on CUDA.")
    parser.add_argument(
        "--preset",
        default="none",
        choices=["none", "distraction4", "distraction5", "fatigue2"],
        help="Optional prediction label mapping preset.",
    )
    parser.add_argument(
        "--map-file",
        default=None,
        help="Optional JSON file for prediction label mapping. Overrides preset when provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to samples-dir.",
    )
    parser.add_argument(
        "--include-labels",
        default=None,
        help="Optional comma-separated true labels to include in evaluation, e.g. safe_drive,radio,drinking,talking_to_passenger",
    )
    return parser.parse_args()


def infer_training_imgsz(model_path: Path) -> int | None:
    args_path = model_path.resolve().parent.parent / "args.yaml"
    if not args_path.exists():
        return None

    for line in args_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("imgsz:"):
            value = line.split(":", 1)[1].strip()
            try:
                return int(value)
            except ValueError:
                return None
    return None


def load_sample_rows(samples_dir: Path):
    metadata_path = samples_dir / "samples_metadata.csv"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["image_path"] = str(Path(row["image_path"]))
            if "frame_idx" in row and row["frame_idx"] not in ("", None):
                row["frame_idx"] = int(row["frame_idx"])
        return rows

    rows = []
    class_dirs = [p for p in sorted(samples_dir.iterdir()) if p.is_dir()]
    for class_dir in class_dirs:
        if class_dir.name.startswith("selected_") or "__to__" in class_dir.name:
            continue
        for image_path in sorted(class_dir.glob("*.jpg")):
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "true_label": class_dir.name,
                    "frame_idx": -1,
                    "video_name": "",
                    "json_name": "",
                    "stream": "",
                    "label_prefix": "",
                    "frame_shift": "",
                }
            )
    return rows


def load_mapping(args) -> dict:
    if args.map_file:
        with open(args.map_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        mapping = {}
        for key, value in raw.items():
            mapping[key] = value
            if isinstance(key, str) and key.isdigit():
                mapping[int(key)] = value
        return mapping

    if args.preset == "distraction4":
        return DISTRACTION_4_PRESET
    if args.preset == "distraction5":
        return DISTRACTION_5_PRESET
    if args.preset == "fatigue2":
        return FATIGUE_2_PRESET
    return {}


def normalize_label(label, mapping: dict, unknown_label: str | None = None):
    if label in mapping:
        return mapping[label]
    label_str = str(label)
    if label_str in mapping:
        return mapping[label_str]
    if mapping and unknown_label is not None:
        return unknown_label
    return label_str


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main():
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir) if args.output_dir else samples_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_sample_rows(samples_dir)
    if not rows:
        raise FileNotFoundError(f"No sampled images found under: {samples_dir}")

    include_labels = None
    if args.include_labels:
        include_labels = {item.strip() for item in args.include_labels.split(",") if item.strip()}
        rows = [row for row in rows if str(row["true_label"]) in include_labels]
        if not rows:
            raise ValueError(f"No rows left after include-labels filter: {sorted(include_labels)}")

    label_mapping = load_mapping(args)

    image_paths = [row["image_path"] for row in rows]
    model = YOLO(args.model)
    infer_imgsz = args.imgsz or infer_training_imgsz(model_path)
    predict_kwargs = {"source": image_paths, "verbose": False, "batch": args.batch, "stream": True}
    if args.device is not None:
        predict_kwargs["device"] = args.device
    if infer_imgsz is not None:
        predict_kwargs["imgsz"] = infer_imgsz
    if args.half:
        predict_kwargs["half"] = True
    results = model.predict(**predict_kwargs)

    model_class_names = [model.names[i] for i in sorted(model.names)]
    normalized_pred_names = [normalize_label(name, label_mapping, unknown_label="other") for name in model_class_names]

    details_path = output_dir / "prediction_details.csv"
    summary_path = output_dir / "prediction_summary.csv"
    report_path = output_dir / "classification_report.csv"
    metrics_path = output_dir / "metrics.json"

    confusion = defaultdict(Counter)
    detailed_rows = []
    labels_seen = set()

    for row, result in zip(rows, results):
        probs = result.probs.data.cpu().numpy()
        pred_idx = int(probs.argmax())
        raw_pred_class = model_class_names[pred_idx]
        pred_class = normalize_label(raw_pred_class, label_mapping, unknown_label="other")
        pred_conf = float(probs[pred_idx])
        true_label = normalize_label(row["true_label"], label_mapping)

        labels_seen.add(true_label)
        labels_seen.add(pred_class)
        confusion[true_label][pred_class] += 1

        top_indices = probs.argsort()[::-1][: args.topk]
        topk_pairs = []
        for i in top_indices:
            raw_name = model_class_names[i]
            norm_name = normalize_label(raw_name, label_mapping, unknown_label="other")
            topk_pairs.append((raw_name, norm_name, float(probs[i])))

        detail = {
            **row,
            "true_label_normalized": true_label,
            "pred_class_raw": raw_pred_class,
            "pred_class": pred_class,
            "pred_conf": f"{pred_conf:.6f}",
        }
        for rank in range(args.topk):
            if rank < len(topk_pairs):
                detail[f"top{rank + 1}_class_raw"] = topk_pairs[rank][0]
                detail[f"top{rank + 1}_class"] = topk_pairs[rank][1]
                detail[f"top{rank + 1}_conf"] = f"{topk_pairs[rank][2]:.6f}"
            else:
                detail[f"top{rank + 1}_class_raw"] = ""
                detail[f"top{rank + 1}_class"] = ""
                detail[f"top{rank + 1}_conf"] = ""
        detailed_rows.append(detail)

    labels = sorted(labels_seen)

    preferred_detail_fields = [
        "image_path",
        "true_label",
        "true_label_normalized",
        "mapped_class",
        "frame_idx",
        "video_name",
        "json_name",
        "stream",
        "label_prefix",
        "frame_shift",
        "source_manifest",
        "source_split",
    ]
    row_fields = []
    for field in preferred_detail_fields:
        if any(field in row for row in detailed_rows):
            row_fields.append(field)

    extra_row_fields = sorted({key for row in detailed_rows for key in row.keys()} - set(row_fields))
    detail_fields = row_fields + extra_row_fields
    with details_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary_fields = ["true_label", "count", "correct", "recall"] + [f"pred_{label}" for label in labels]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for true_label in labels:
            total = sum(confusion[true_label].values())
            correct = confusion[true_label].get(true_label, 0)
            row = {
                "true_label": true_label,
                "count": total,
                "correct": correct,
                "recall": f"{safe_div(correct, total):.6f}",
            }
            for pred_label in labels:
                row[f"pred_{pred_label}"] = confusion[true_label].get(pred_label, 0)
            writer.writerow(row)

    report_rows = []
    correct_total = 0
    total = 0
    f1_values = []
    true_labels = [label for label in labels if sum(confusion[label].values()) > 0]
    for label in labels:
        tp = confusion[label].get(label, 0)
        fp = sum(confusion[other].get(label, 0) for other in labels if other != label)
        fn = sum(confusion[label].values()) - tp
        support = sum(confusion[label].values())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        report_rows.append(
            {
                "label": label,
                "support": support,
                "precision": f"{precision:.6f}",
                "recall": f"{recall:.6f}",
                "f1": f"{f1:.6f}",
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
        correct_total += tp
        total += support
        if support > 0:
            f1_values.append(f1)

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "support", "precision", "recall", "f1", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(report_rows)

    accuracy = safe_div(correct_total, total)
    macro_f1 = safe_div(sum(f1_values), len(f1_values))
    metrics = {
        "model": args.model,
        "samples_dir": str(samples_dir),
        "images": total,
        "accuracy_top1": accuracy,
        "macro_f1": macro_f1,
        "true_labels": true_labels,
        "labels": labels,
        "prediction_labels_raw": model_class_names,
        "prediction_labels_normalized": normalized_pred_names,
        "mapping_preset": args.preset,
        "mapping_file": args.map_file,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"details: {details_path}")
    print(f"summary: {summary_path}")
    print(f"report: {report_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
