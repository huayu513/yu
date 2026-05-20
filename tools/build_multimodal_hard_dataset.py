from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_multimodal_v1 import (
    MultiModalDriverDataset,
    MultiModalDriverNet,
    build_transforms,
    multimodal_collate_fn,
)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        label = row[key]
        if label != "-1":
            counter[label] += 1
    return dict(sorted(counter.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard-example oversampled multimodal dataset.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pair-label-a", type=str, default="safe_drive")
    parser.add_argument("--pair-label-b", type=str, default="talking_to_passenger")
    parser.add_argument("--distraction-mis-repeat", type=int, default=2)
    parser.add_argument("--distraction-lowconf-repeat", type=int, default=1)
    parser.add_argument("--distraction-lowconf-thresh", type=float, default=0.75)
    parser.add_argument("--fatigue-mis-repeat", type=int, default=2)
    parser.add_argument("--fatigue-lowconf-repeat", type=int, default=1)
    parser.add_argument("--fatigue-lowconf-thresh", type=float, default=0.70)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    source_root = Path(cfg["data_root"])
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((source_root / "metadata.json").read_text(encoding="utf-8"))
    distraction_classes = metadata["distraction_classes"]
    fatigue_classes = metadata["fatigue_classes"]
    distraction_to_idx = {name: i for i, name in enumerate(distraction_classes)}
    fatigue_to_idx = {name: i for i, name in enumerate(fatigue_classes)}

    train_dataset = MultiModalDriverDataset(
        csv_path=source_root / "train.csv",
        distraction_classes=distraction_classes,
        fatigue_classes=fatigue_classes,
        body_transform=build_transforms(cfg, "val"),
        face_transform=build_transforms(cfg, "val"),
        temporal_face=bool(cfg.get("temporal_face_dim", 0)),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
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
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg["device"]) != "cpu" else "cpu")
    if str(cfg["device"]).isdigit() and device.type == "cuda":
        device = torch.device(f"cuda:{cfg['device']}")
    model = model.to(device).eval()

    source_rows = train_dataset.rows
    row_by_id = {row["sample_id"]: row for row in source_rows}

    hard_duplicate_rows: list[dict[str, str]] = []
    mining_stats = Counter()

    pair_set = {args.pair_label_a, args.pair_label_b}

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
            outputs = model(body, face, face_prev=face_prev, face_next=face_next)

            d_probs = torch.softmax(outputs["distraction_logits"], dim=1).cpu()
            d_preds = d_probs.argmax(dim=1).cpu()
            f_probs = torch.softmax(outputs["fatigue_logits"], dim=1).cpu()
            f_preds = f_probs.argmax(dim=1).cpu()

            for i, sample_id in enumerate(batch["sample_id"]):
                row = row_by_id[sample_id]
                extra_repeats = 0
                tags: list[str] = []

                if row["distraction_label"] != "-1":
                    true_name = row["distraction_label"]
                    pred_name = distraction_classes[int(d_preds[i].item())]
                    pred_conf = float(d_probs[i, int(d_preds[i].item())].item())
                    if true_name in pair_set:
                        if pred_name != true_name:
                            extra_repeats += args.distraction_mis_repeat
                            tags.append("hard_distraction_mis")
                            mining_stats["hard_distraction_mis"] += 1
                        elif pred_conf < args.distraction_lowconf_thresh:
                            extra_repeats += args.distraction_lowconf_repeat
                            tags.append("hard_distraction_lowconf")
                            mining_stats["hard_distraction_lowconf"] += 1

                if row["fatigue_label"] != "-1":
                    true_name = row["fatigue_label"]
                    pred_name = fatigue_classes[int(f_preds[i].item())]
                    pred_conf = float(f_probs[i, int(f_preds[i].item())].item())
                    if pred_name != true_name:
                        extra_repeats += args.fatigue_mis_repeat
                        tags.append("hard_fatigue_mis")
                        mining_stats["hard_fatigue_mis"] += 1
                    elif pred_conf < args.fatigue_lowconf_thresh:
                        extra_repeats += args.fatigue_lowconf_repeat
                        tags.append("hard_fatigue_lowconf")
                        mining_stats["hard_fatigue_lowconf"] += 1

                for rep in range(extra_repeats):
                    dup = deepcopy(row)
                    dup["sample_id"] = f"{row['sample_id']}__hard{rep + 1}"
                    base_notes = row.get("notes", "")
                    joined_tags = ",".join(tags)
                    dup["notes"] = f"{base_notes};{joined_tags}".strip(";")
                    hard_duplicate_rows.append(dup)

    new_train_rows = source_rows + hard_duplicate_rows

    val_rows = list(csv.DictReader(open(source_root / "val.csv", "r", encoding="utf-8")))
    test_rows = list(csv.DictReader(open(source_root / "test.csv", "r", encoding="utf-8")))

    write_csv(output_root / "train.csv", new_train_rows)
    write_csv(output_root / "val.csv", val_rows)
    write_csv(output_root / "test.csv", test_rows)

    new_metadata = deepcopy(metadata)
    new_metadata["train_count"] = len(new_train_rows)
    new_metadata["train_distraction_counts"] = compute_counts(new_train_rows, "distraction_label")
    new_metadata["train_fatigue_counts"] = compute_counts(new_train_rows, "fatigue_label")
    new_metadata["hard_mining"] = {
        "base_checkpoint": str(args.checkpoint),
        "pair_labels": [args.pair_label_a, args.pair_label_b],
        "distraction_mis_repeat": args.distraction_mis_repeat,
        "distraction_lowconf_repeat": args.distraction_lowconf_repeat,
        "distraction_lowconf_thresh": args.distraction_lowconf_thresh,
        "fatigue_mis_repeat": args.fatigue_mis_repeat,
        "fatigue_lowconf_repeat": args.fatigue_lowconf_repeat,
        "fatigue_lowconf_thresh": args.fatigue_lowconf_thresh,
        "duplicate_count": len(hard_duplicate_rows),
        "stats": dict(mining_stats),
    }
    (output_root / "metadata.json").write_text(json.dumps(new_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Hard-mined dataset written to {output_root}")
    print(f"Base train rows: {len(source_rows)}")
    print(f"Duplicate rows: {len(hard_duplicate_rows)}")
    print(f"New train rows: {len(new_train_rows)}")
    print(json.dumps(dict(mining_stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
