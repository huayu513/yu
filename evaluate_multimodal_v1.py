from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_multimodal_v1 import (
    MultiModalDriverDataset,
    MultiModalDriverNet,
    build_transforms,
    multimodal_collate_fn,
)


def evaluate_task(logits: torch.Tensor, labels: torch.Tensor, class_names: list[str]) -> tuple[list[dict], dict]:
    mask = labels >= 0
    rows: list[dict] = []
    summary: dict[str, dict[str, int]] = {}
    if not mask.any():
        return rows, summary

    probs = torch.softmax(logits[mask], dim=1)
    preds = probs.argmax(dim=1)
    labels_valid = labels[mask]

    for pred_idx, true_idx, prob_vec in zip(preds.tolist(), labels_valid.tolist(), probs.tolist()):
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        rows.append(
            {
                "true_label": true_name,
                "pred_label": pred_name,
                "pred_conf": float(prob_vec[pred_idx]),
            }
        )
        summary.setdefault(true_name, {})
        summary[true_name][pred_name] = summary[true_name].get(pred_name, 0) + 1
    return rows, summary


def write_summary(path: Path, summary: dict[str, dict[str, int]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "pred_label", "count"])
        for true_label in sorted(summary):
            for pred_label, count in sorted(summary[true_label].items(), key=lambda x: (-x[1], x[0])):
                writer.writerow([true_label, pred_label, count])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multimodal driver model on a split CSV.")
    parser.add_argument("--config", type=str, default="train_multimodal_v1.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = json.loads(json.dumps(__import__("yaml").safe_load(Path(args.config).read_text(encoding="utf-8"))))
    dataset_root = Path(cfg["data_root"])
    metadata = json.loads((dataset_root / "metadata.json").read_text(encoding="utf-8"))
    distraction_classes = metadata["distraction_classes"]
    fatigue_classes = metadata["fatigue_classes"]

    dataset = MultiModalDriverDataset(
        csv_path=dataset_root / f"{args.split}.csv",
        distraction_classes=distraction_classes,
        fatigue_classes=fatigue_classes,
        body_transform=build_transforms(cfg, "val"),
        face_transform=build_transforms(cfg, "val"),
        temporal_face=bool(cfg.get("temporal_face_dim", 0)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch or int(cfg["batch"]),
        shuffle=False,
        num_workers=args.workers if args.workers is not None else int(cfg["workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        collate_fn=multimodal_collate_fn,
        drop_last=False,
    )

    model = MultiModalDriverNet(
        body_weight=cfg["body_weight"],
        face_weight=cfg["face_weight"],
        num_distraction_classes=len(distraction_classes),
        num_fatigue_classes=len(fatigue_classes),
        proj_dim=int(cfg["proj_dim"]),
        fusion_dim=int(cfg["fusion_dim"]),
        dropout=float(cfg["dropout"]),
        task_fusion_dim=int(cfg.get("task_fusion_dim", 0)),
        task_head_dim=int(cfg.get("task_head_dim", 0)),
        gate_hidden_dim=int(cfg.get("gate_hidden_dim", 0)),
        aux_head_dim=int(cfg.get("aux_head_dim", 0)),
        fatigue_face_dominant=bool(cfg.get("fatigue_face_dominant", False)),
        task_interaction_dim=int(cfg.get("task_interaction_dim", 0)),
        task_attention_heads=int(cfg.get("task_attention_heads", 0)),
        fatigue_refine_dim=int(cfg.get("fatigue_refine_dim", 0)),
        fatigue_residual_dim=int(cfg.get("fatigue_residual_dim", 0)),
        distraction_residual_dim=int(cfg.get("distraction_residual_dim", 0)),
        fatigue_prior_fusion=bool(cfg.get("fatigue_prior_fusion", False)),
        temporal_face_dim=int(cfg.get("temporal_face_dim", 0)),
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg["device"]) != "cpu" else "cpu")
    if str(cfg["device"]).isdigit() and device.type == "cuda":
        device = torch.device(f"cuda:{cfg['device']}")
    model = model.to(device).eval()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.checkpoint).resolve().parent.parent / f"eval_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    distraction_rows: list[dict] = []
    fatigue_rows: list[dict] = []
    distraction_summary: dict[str, dict[str, int]] = {}
    fatigue_summary: dict[str, dict[str, int]] = {}

    with torch.no_grad():
        for batch in loader:
            body = batch["body"].to(device, non_blocking=device.type == "cuda")
            face = batch["face"].to(device, non_blocking=device.type == "cuda")
            face_prev = batch.get("face_prev")
            face_next = batch.get("face_next")
            if face_prev is not None:
                face_prev = face_prev.to(device, non_blocking=device.type == "cuda")
            if face_next is not None:
                face_next = face_next.to(device, non_blocking=device.type == "cuda")
            distraction_label = batch["distraction_label"].to(device, non_blocking=device.type == "cuda")
            fatigue_label = batch["fatigue_label"].to(device, non_blocking=device.type == "cuda")

            outputs = model(body, face, face_prev=face_prev, face_next=face_next)

            d_rows, d_summary = evaluate_task(
                outputs["distraction_logits"].cpu(), distraction_label.cpu(), distraction_classes
            )
            f_rows, f_summary = evaluate_task(outputs["fatigue_logits"].cpu(), fatigue_label.cpu(), fatigue_classes)
            distraction_rows.extend(d_rows)
            fatigue_rows.extend(f_rows)
            for true_label, preds in d_summary.items():
                distraction_summary.setdefault(true_label, {})
                for pred_label, count in preds.items():
                    distraction_summary[true_label][pred_label] = (
                        distraction_summary[true_label].get(pred_label, 0) + count
                    )
            for true_label, preds in f_summary.items():
                fatigue_summary.setdefault(true_label, {})
                for pred_label, count in preds.items():
                    fatigue_summary[true_label][pred_label] = fatigue_summary[true_label].get(pred_label, 0) + count

    with open(output_dir / "distraction_details.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["true_label", "pred_label", "pred_conf"])
        writer.writeheader()
        writer.writerows(distraction_rows)
    with open(output_dir / "fatigue_details.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["true_label", "pred_label", "pred_conf"])
        writer.writeheader()
        writer.writerows(fatigue_rows)

    write_summary(output_dir / "distraction_summary.csv", distraction_summary)
    write_summary(output_dir / "fatigue_summary.csv", fatigue_summary)

    def calc_acc(summary: dict[str, dict[str, int]]) -> float:
        correct = 0
        total = 0
        for true_label, preds in summary.items():
            for pred_label, count in preds.items():
                total += count
                if pred_label == true_label:
                    correct += count
        return correct / total if total else 0.0

    metrics = {
        "split": args.split,
        "distraction_accuracy": calc_acc(distraction_summary),
        "fatigue_accuracy": calc_acc(fatigue_summary),
        "distraction_count": sum(sum(preds.values()) for preds in distraction_summary.values()),
        "fatigue_count": sum(sum(preds.values()) for preds in fatigue_summary.values()),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
