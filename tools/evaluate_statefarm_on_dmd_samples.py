import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a State Farm classifier on sampled DMD images and export per-image and per-class summaries."
    )
    parser.add_argument("--model", required=True, help="Path to classifier weights, e.g. runs/classify/train4/weights/best.pt.")
    parser.add_argument(
        "--samples-dir",
        required=True,
        help="Sample directory created by sample_dmd_frames_by_action.py.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="How many top predictions to save in the detailed CSV.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Prediction batch size.",
    )
    return parser.parse_args()


def load_sample_rows(samples_dir: Path):
    metadata_path = samples_dir / "samples_metadata.csv"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["image_path"] = str(Path(row["image_path"]))
            row["frame_idx"] = int(row["frame_idx"])
        return rows

    rows = []
    # When no metadata CSV is available, treat only immediate child folders as class directories.
    # This avoids accidentally re-reading analysis outputs such as selected_misclassified_*.
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


def main():
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    rows = load_sample_rows(samples_dir)
    if not rows:
        raise FileNotFoundError(f"No sampled images found under: {samples_dir}")

    image_paths = [row["image_path"] for row in rows]
    model = YOLO(args.model)
    results = model.predict(source=image_paths, verbose=False, batch=args.batch)

    class_names = [model.names[i] for i in sorted(model.names)]
    details_path = samples_dir / "prediction_details.csv"
    summary_path = samples_dir / "prediction_summary.csv"

    summary = defaultdict(Counter)
    top1_counter = defaultdict(Counter)
    detailed_rows = []

    for row, result in zip(rows, results):
        probs = result.probs.data.cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_class = class_names[pred_idx]
        pred_conf = float(probs[pred_idx])

        top_indices = probs.argsort()[::-1][: args.topk]
        topk_pairs = [(class_names[i], float(probs[i])) for i in top_indices]

        summary[row["true_label"]]["count"] += 1
        top1_counter[row["true_label"]][pred_class] += 1

        detail = {
            **row,
            "pred_class": pred_class,
            "pred_conf": f"{pred_conf:.6f}",
        }
        for rank in range(args.topk):
            if rank < len(topk_pairs):
                detail[f"top{rank + 1}_class"] = topk_pairs[rank][0]
                detail[f"top{rank + 1}_conf"] = f"{topk_pairs[rank][1]:.6f}"
            else:
                detail[f"top{rank + 1}_class"] = ""
                detail[f"top{rank + 1}_conf"] = ""
        detailed_rows.append(detail)

    for true_label, counter in top1_counter.items():
        total = sum(counter.values())
        for pred_class, count in counter.items():
            summary[true_label][f"pred_{pred_class}"] = count
        if total:
            pred_class, count = counter.most_common(1)[0]
            summary[true_label]["most_common_pred"] = pred_class
            summary[true_label]["most_common_ratio"] = f"{count / total:.4f}"

    preferred_detail_fields = [
        "image_path",
        "true_label",
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
        if any(field in row for row in rows):
            row_fields.append(field)

    extra_row_fields = sorted({key for row in rows for key in row.keys()} - set(row_fields))
    detail_fields = row_fields + extra_row_fields + [
        "pred_class",
        "pred_conf",
    ]
    for rank in range(args.topk):
        detail_fields.extend([f"top{rank + 1}_class", f"top{rank + 1}_conf"])

    with details_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary_fields = ["true_label", "count", "most_common_pred", "most_common_ratio"] + [
        f"pred_{name}" for name in class_names
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for true_label in sorted(summary):
            row = {"true_label": true_label}
            for field in summary_fields[1:]:
                row[field] = summary[true_label].get(field, 0 if field.startswith("pred_") or field == "count" else "")
            writer.writerow(row)

    print(f"model: {args.model}")
    print(f"samples_dir: {samples_dir}")
    print(f"images: {len(rows)}")
    print(f"details: {details_path}")
    print(f"summary: {summary_path}")

    print("\nTop-1 prediction distribution by DMD label")
    for true_label in sorted(top1_counter):
        total = sum(top1_counter[true_label].values())
        print(f"\n{true_label} ({total})")
        for pred_class, count in top1_counter[true_label].most_common():
            print(f"  {pred_class}: {count} ({count / total:.3f})")


if __name__ == "__main__":
    main()

# python tools\evaluate_statefarm_on_dmd_samples.py `
#   --model "runs\classify\train24\weights\best.pt" `
#   --samples-dir "runs\dmd_samples\driver_actions_body_100" `
#   --topk 3 `
#   --batch 32
