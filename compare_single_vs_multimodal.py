from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from train_multimodal_v1 import MultiModalDriverNet, build_transforms, load_yaml
from ultralytics.data.augment import classify_transforms
from ultralytics.nn.tasks import load_checkpoint


DISTRACTION_SINGLE_MAP = {
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
}


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def predict_single(model, image_path: str, transform, device: torch.device) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        pred_name = model.names[pred_idx] if isinstance(model.names, dict) else model.names[pred_idx]
        conf = float(probs[pred_idx].item())
    return str(pred_name), conf


def predict_multimodal(model, body_path: str, face_path: str, transform, device: torch.device, distraction_classes: list[str], fatigue_classes: list[str]) -> dict[str, tuple[str, float]]:
    body = transform(Image.open(body_path).convert("RGB")).unsqueeze(0).to(device)
    face = transform(Image.open(face_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(body, face)
        d_probs = torch.softmax(outputs["distraction_logits"], dim=1)[0]
        f_probs = torch.softmax(outputs["fatigue_logits"], dim=1)[0]
    d_idx = int(d_probs.argmax().item())
    f_idx = int(f_probs.argmax().item())
    return {
        "distraction": (distraction_classes[d_idx], float(d_probs[d_idx].item())),
        "fatigue": (fatigue_classes[f_idx], float(f_probs[f_idx].item())),
    }


def update_summary(summary: dict[str, dict[str, int]], true_label: str, pred_label: str) -> None:
    summary.setdefault(true_label, {})
    summary[true_label][pred_label] = summary[true_label].get(pred_label, 0) + 1


def calc_acc(summary: dict[str, dict[str, int]]) -> float:
    correct = 0
    total = 0
    for true_label, preds in summary.items():
        for pred_label, count in preds.items():
            total += count
            if pred_label == true_label:
                correct += count
    return correct / total if total else 0.0


def write_summary(path: Path, summary: dict[str, dict[str, int]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "pred_label", "count"])
        for true_label in sorted(summary):
            for pred_label, count in sorted(summary[true_label].items(), key=lambda x: (-x[1], x[0])):
                writer.writerow([true_label, pred_label, count])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair comparison: single-modality vs multimodal on same split.")
    parser.add_argument("--config", type=str, default="train_multimodal_v1.yaml")
    parser.add_argument("--multimodal-checkpoint", type=str, required=True)
    parser.add_argument("--body-checkpoint", type=str, default="runs/classify/train24/weights/best.pt")
    parser.add_argument("--face-checkpoint", type=str, default="runs/classify/train29/weights/best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_root = Path(cfg["data_root"])
    metadata = json.loads((dataset_root / "metadata.json").read_text(encoding="utf-8"))
    distraction_classes = metadata["distraction_classes"]
    fatigue_classes = metadata["fatigue_classes"]
    rows = read_rows(dataset_root / f"{args.split}.csv")

    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg["device"]) != "cpu" else "cpu")
    if str(cfg["device"]).isdigit() and device.type == "cuda":
        device = torch.device(f"cuda:{cfg['device']}")

    body_model, _ = load_checkpoint(args.body_checkpoint, device=device, fuse=False)
    face_model, _ = load_checkpoint(args.face_checkpoint, device=device, fuse=False)
    body_model.eval()
    face_model.eval()

    body_imgsz = int(getattr(body_model, "args", {}).get("imgsz", 224))
    face_imgsz = int(getattr(face_model, "args", {}).get("imgsz", 224))
    body_transform = classify_transforms(size=body_imgsz)
    face_transform = classify_transforms(size=face_imgsz)

    mm_model = MultiModalDriverNet(
        body_weight=cfg["body_weight"],
        face_weight=cfg["face_weight"],
        num_distraction_classes=len(distraction_classes),
        num_fatigue_classes=len(fatigue_classes),
        proj_dim=int(cfg["proj_dim"]),
        fusion_dim=int(cfg["fusion_dim"]),
        dropout=float(cfg["dropout"]),
    )
    mm_ckpt = torch.load(args.multimodal_checkpoint, map_location="cpu", weights_only=False)
    mm_model.load_state_dict(mm_ckpt["model_state_dict"], strict=False)
    mm_model = mm_model.to(device).eval()
    mm_transform = build_transforms(cfg, "val")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.multimodal_checkpoint).resolve().parent.parent / f"compare_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    distraction_single_summary: dict[str, dict[str, int]] = {}
    distraction_multimodal_summary: dict[str, dict[str, int]] = {}
    fatigue_single_summary: dict[str, dict[str, int]] = {}
    fatigue_multimodal_summary: dict[str, dict[str, int]] = {}
    details: list[dict[str, str | float]] = []

    for row in rows:
        detail = {
            "sample_id": row["sample_id"],
            "source": row["source"],
            "true_distraction": row["distraction_label"],
            "body_only_pred": "",
            "body_only_conf": "",
            "multimodal_distraction_pred": "",
            "multimodal_distraction_conf": "",
            "true_fatigue": row["fatigue_label"],
            "face_only_pred": "",
            "face_only_conf": "",
            "multimodal_fatigue_pred": "",
            "multimodal_fatigue_conf": "",
        }

        mm_preds = predict_multimodal(
            mm_model,
            row["body_path"],
            row["face_path"],
            mm_transform,
            device,
            distraction_classes,
            fatigue_classes,
        )

        if row["distraction_label"] != "-1":
            pred_name, conf = predict_single(body_model, row["body_path"], body_transform, device)
            mapped_pred = DISTRACTION_SINGLE_MAP.get(pred_name, f"other:{pred_name}")
            detail["body_only_pred"] = mapped_pred
            detail["body_only_conf"] = conf
            detail["multimodal_distraction_pred"] = mm_preds["distraction"][0]
            detail["multimodal_distraction_conf"] = mm_preds["distraction"][1]
            update_summary(distraction_single_summary, row["distraction_label"], mapped_pred)
            update_summary(distraction_multimodal_summary, row["distraction_label"], mm_preds["distraction"][0])

        if row["fatigue_label"] != "-1":
            pred_name, conf = predict_single(face_model, row["face_path"], face_transform, device)
            detail["face_only_pred"] = pred_name
            detail["face_only_conf"] = conf
            detail["multimodal_fatigue_pred"] = mm_preds["fatigue"][0]
            detail["multimodal_fatigue_conf"] = mm_preds["fatigue"][1]
            update_summary(fatigue_single_summary, row["fatigue_label"], pred_name)
            update_summary(fatigue_multimodal_summary, row["fatigue_label"], mm_preds["fatigue"][0])

        details.append(detail)

    with open(output_dir / "comparison_details.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    write_summary(output_dir / "distraction_single_summary.csv", distraction_single_summary)
    write_summary(output_dir / "distraction_multimodal_summary.csv", distraction_multimodal_summary)
    write_summary(output_dir / "fatigue_single_summary.csv", fatigue_single_summary)
    write_summary(output_dir / "fatigue_multimodal_summary.csv", fatigue_multimodal_summary)

    metrics = {
        "split": args.split,
        "distraction_single_accuracy": calc_acc(distraction_single_summary),
        "distraction_multimodal_accuracy": calc_acc(distraction_multimodal_summary),
        "fatigue_single_accuracy": calc_acc(fatigue_single_summary),
        "fatigue_multimodal_accuracy": calc_acc(fatigue_multimodal_summary),
        "distraction_count": sum(sum(v.values()) for v in distraction_single_summary.values()),
        "fatigue_count": sum(sum(v.values()) for v in fatigue_single_summary.values()),
    }
    (output_dir / "comparison_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
