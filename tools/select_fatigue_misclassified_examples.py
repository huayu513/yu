import argparse
import csv
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pick representative fatigue/non_fatigue misclassified samples from prediction_details.csv."
    )
    parser.add_argument("--details-csv", required=True, help="Path to prediction_details.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory to store exported examples.")
    parser.add_argument(
        "--samples-per-pair",
        type=int,
        default=24,
        help="How many top-confidence samples to export for each confusion pair.",
    )
    return parser.parse_args()


def safe_imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def safe_imwrite(path: Path, image):
    suffix = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def slugify_pair(true_label, pred_class):
    return f"{true_label}__to__{pred_class}"


def create_contact_sheet(image_paths, out_path: Path, title: str, cell_w=224, cell_h=224, cols=4):
    if not image_paths:
        return

    rows = math.ceil(len(image_paths) / cols)
    title_h = 42
    sheet = np.full((title_h + rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    cv2.putText(sheet, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)

    for idx, image_path in enumerate(image_paths):
        image = safe_imread(image_path)
        if image is None:
            continue
        image = cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        row = idx // cols
        col = idx % cols
        y0 = title_h + row * cell_h
        x0 = col * cell_w
        sheet[y0 : y0 + cell_h, x0 : x0 + cell_w] = image
        cv2.rectangle(sheet, (x0, y0), (x0 + cell_w, y0 + cell_h), (255, 255, 255), 2)

    safe_imwrite(out_path, sheet)


def load_rows(details_csv: Path):
    with details_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["pred_conf"] = float(row["pred_conf"])
        row["image_path"] = str(Path(row["image_path"]))
    return rows


def main():
    args = parse_args()
    details_csv = Path(args.details_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(details_csv)
    grouped = defaultdict(list)
    pair_counts = Counter()

    for row in rows:
        if row["true_label"] == row["pred_class"]:
            continue
        key = (row["true_label"], row["pred_class"])
        grouped[key].append(row)
        pair_counts[key] += 1

    summary_rows = []
    manifest_rows = []

    for (true_label, pred_class), count in pair_counts.most_common():
        pair_rows = sorted(grouped[(true_label, pred_class)], key=lambda x: x["pred_conf"], reverse=True)
        selected = pair_rows[: args.samples_per_pair]
        pair_dir = output_dir / slugify_pair(true_label, pred_class)
        pair_dir.mkdir(parents=True, exist_ok=True)

        copied_paths = []
        for idx, row in enumerate(selected, start=1):
            src = Path(row["image_path"])
            dst = pair_dir / f"{idx:02d}_{src.name}"
            shutil.copy2(src, dst)
            copied_paths.append(dst)
            manifest_rows.append(
                {
                    "pair_dir": str(pair_dir.resolve()),
                    "true_label": true_label,
                    "pred_class": pred_class,
                    "pred_conf": f"{row['pred_conf']:.6f}",
                    "image_path": str(dst.resolve()),
                    "source_image_path": str(src.resolve()),
                    "top2_class": row.get("top2_class", ""),
                    "top2_conf": row.get("top2_conf", ""),
                }
            )

        sheet_path = pair_dir / "contact_sheet.jpg"
        create_contact_sheet(copied_paths, sheet_path, f"{true_label} -> {pred_class} ({len(selected)} samples)")

        summary_rows.append(
            {
                "true_label": true_label,
                "pred_class": pred_class,
                "count_in_csv": count,
                "exported": len(selected),
                "pair_dir": str(pair_dir.resolve()),
                "contact_sheet": str(sheet_path.resolve()),
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_dir",
                "true_label",
                "pred_class",
                "pred_conf",
                "image_path",
                "source_image_path",
                "top2_class",
                "top2_conf",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["true_label", "pred_class", "count_in_csv", "exported", "pair_dir", "contact_sheet"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"details_csv: {details_csv}")
    print(f"output_dir: {output_dir}")
    print(f"manifest: {manifest_path}")
    print(f"summary: {summary_path}")
    print("\nSelected confusion pairs")
    for row in summary_rows:
        print(f"{row['true_label']} -> {row['pred_class']}: count={row['count_in_csv']}, exported={row['exported']}")


if __name__ == "__main__":
    main()
