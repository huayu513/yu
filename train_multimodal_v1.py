from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ultralytics.data.augment import classify_augmentations, classify_transforms
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.files import increment_path


def seed_everything(seed: int) -> None:
    # 让每次训练尽量可复现。
    # 这里统一固定 Python、NumPy、PyTorch 的随机种子，避免“同样代码每次结果差很多”。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict:
    # 所有训练超参数都放在 yaml 里，脚本本身只负责“读取并执行”。
    # 这样后面换路径、batch、学习率时，不需要改 Python 代码。
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    # 把本次训练实际使用的参数再保存一份到输出目录。
    # 这样后面回头看 train 结果时，你能知道当时到底是用什么配置跑的。
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


class MultiModalDriverDataset(Dataset):
    """CSV-driven paired body/face dataset with optional labels per task.

    这里的一条样本，不再是一张图，而是两张图：
    1. body 图
    2. face 图

    同时还可能带两个标签：
    1. distraction_label
    2. fatigue_label

    但注意：不是每条样本都有两个标签。
    - S1 样本只有分心标签，没有疲劳标签
    - S5 样本只有疲劳标签，没有分心标签

    所以这里会把“缺失标签”统一记成 -1，后面 loss 会根据 -1 自动跳过。
    """

    def __init__(
        self,
        csv_path: str | Path,
        distraction_classes: list[str],
        fatigue_classes: list[str],
        body_transform,
        face_transform,
        temporal_face: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.rows = self._read_rows(self.csv_path)
        self.body_transform = body_transform
        self.face_transform = face_transform
        self.temporal_face = bool(temporal_face)
        # 把字符串标签映射成整数类别索引，便于后面做交叉熵损失。
        self.distraction_to_idx = {name: i for i, name in enumerate(distraction_classes)}
        self.fatigue_to_idx = {name: i for i, name in enumerate(fatigue_classes)}
        self.prev_face_index: dict[int, int] = {}
        self.next_face_index: dict[int, int] = {}
        if self.temporal_face:
            self._build_temporal_neighbors()

    @staticmethod
    def _read_rows(csv_path: Path) -> list[dict[str, str]]:
        # CSV 里每一行就是一条多模态样本的元信息。
        with open(csv_path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _label_to_index(label: str, mapping: dict[str, int]) -> int:
        # -1 表示“这条样本在这个任务上没有标签”。
        return -1 if label == "-1" else mapping[label]

    def _build_temporal_neighbors(self) -> None:
        groups: dict[str, list[tuple[int, int]]] = {}
        for idx, row in enumerate(self.rows):
            if row["fatigue_label"] == "-1":
                continue
            groups.setdefault(row["source"], []).append((idx, int(row["face_frame_idx"])))
        for items in groups.values():
            items.sort(key=lambda x: x[1])
            for pos, (idx, _frame) in enumerate(items):
                self.prev_face_index[idx] = items[pos - 1][0] if pos > 0 else idx
                self.next_face_index[idx] = items[pos + 1][0] if pos < len(items) - 1 else idx

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]

        # 这里分别读取 body 和 face 图像。
        # 第一版多模态设计里，这两张图是同一时刻对应的两个视角。
        body_img = Image.open(row["body_path"]).convert("RGB")
        face_img = Image.open(row["face_path"]).convert("RGB")

        sample = {
            "sample_id": row["sample_id"],
            "source": row["source"],
            # 两个模态各自走自己的图像增强/预处理流程。
            "body": self.body_transform(body_img),
            "face": self.face_transform(face_img),
            "distraction_label": self._label_to_index(row["distraction_label"], self.distraction_to_idx),
            "fatigue_label": self._label_to_index(row["fatigue_label"], self.fatigue_to_idx),
        }
        if self.temporal_face:
            prev_row = self.rows[self.prev_face_index.get(index, index)]
            next_row = self.rows[self.next_face_index.get(index, index)]
            prev_face_img = Image.open(prev_row["face_path"]).convert("RGB")
            next_face_img = Image.open(next_row["face_path"]).convert("RGB")
            sample["face_prev"] = self.face_transform(prev_face_img)
            sample["face_next"] = self.face_transform(next_face_img)
        return sample


def multimodal_collate_fn(batch: list[dict]) -> dict:
    # DataLoader 默认的拼接逻辑对这种“字典 + 双图像 + 双标签”的结构不够直观，
    # 所以这里手动定义 batch 该怎么拼。
    #
    # 最后得到的 batch 结构是：
    # - body: [B, 3, H, W]
    # - face: [B, 3, H, W]
    # - distraction_label: [B]
    # - fatigue_label: [B]
    collated = {
        "sample_id": [item["sample_id"] for item in batch],
        "source": [item["source"] for item in batch],
        "body": torch.stack([item["body"] for item in batch], dim=0),
        "face": torch.stack([item["face"] for item in batch], dim=0),
        "distraction_label": torch.tensor([item["distraction_label"] for item in batch], dtype=torch.long),
        "fatigue_label": torch.tensor([item["fatigue_label"] for item in batch], dtype=torch.long),
    }
    if "face_prev" in batch[0]:
        collated["face_prev"] = torch.stack([item["face_prev"] for item in batch], dim=0)
        collated["face_next"] = torch.stack([item["face_next"] for item in batch], dim=0)
    return collated


class UltralyticsClassificationEncoder(nn.Module):
    """Reuses a trained Ultralytics classification model as a feature extractor.

    这一步是整个多模态第一版最关键的“复用已有成果”思路：
    - body 分支，不从零训练，而是复用你已经训练好的分心模型
    - face 分支，不从零训练，而是复用你已经训练好的疲劳模型

    这里不会直接复用原来的最终分类头，而是把分类模型改造成“编码器”： 输入图片 -> 输出一个特征向量
    """

    def __init__(self, weight_path: str | Path) -> None:
        super().__init__()
        # 直接加载你已经训好的单模态权重。
        model, _ = load_checkpoint(weight_path, device="cpu", fuse=False)
        if not hasattr(model, "model"):
            raise TypeError(f"{weight_path} is not an Ultralytics classification checkpoint.")

        layers = list(model.model.children())
        if len(layers) < 2:
            raise ValueError(f"{weight_path} does not contain enough layers for feature extraction.")

        head = layers[-1]
        if not hasattr(head, "conv") or not hasattr(head, "pool") or not hasattr(head, "linear"):
            raise TypeError(f"Unsupported classifier head in {weight_path}.")

        # backbone 负责提取空间特征图。
        self.backbone = nn.Sequential(*layers[:-1])
        # 这里保留分类头里的 conv + pool，但去掉最后 linear。
        # 这样就能把一张图变成一个固定长度向量，而不是直接输出类别。
        #
        # 之所以 deepcopy，是为了让两个分支各自拥有独立参数，后面可以单独微调。
        self.head_conv = deepcopy(head.conv)
        self.head_pool = deepcopy(head.pool)
        # linear.in_features 就是这个编码器最终特征向量的维度。
        self.out_dim = head.linear.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出形状最终会变成 [B, out_dim]。
        x = self.backbone(x)
        x = self.head_conv(x)
        x = self.head_pool(x).flatten(1)
        return x


def load_ultralytics_classifier_linear(weight_path: str | Path) -> nn.Linear:
    model, _ = load_checkpoint(weight_path, device="cpu", fuse=False)
    if not hasattr(model, "model"):
        raise TypeError(f"{weight_path} is not an Ultralytics classification checkpoint.")
    layers = list(model.model.children())
    if not layers:
        raise ValueError(f"{weight_path} does not contain enough layers.")
    head = layers[-1]
    if not hasattr(head, "linear"):
        raise TypeError(f"Unsupported classifier head in {weight_path}.")
    return deepcopy(head.linear)


class MultiModalDriverNet(nn.Module):
    """body + face 双分支，多任务输出的第一版多模态模型。.

    结构非常朴素，目的是先跑通并让你容易读懂：
    1. body_encoder 提取 body 特征
    2. face_encoder 提取 face 特征
    3. 各自先过一层 projection，统一到相同维度
    4. 拼接后再经过 fusion MLP
    5. 输出两个任务头：
    - distraction_head
    - fatigue_head
    """

    def __init__(
        self,
        body_weight: str | Path,
        face_weight: str | Path,
        num_distraction_classes: int,
        num_fatigue_classes: int,
        proj_dim: int = 256,
        fusion_dim: int = 256,
        dropout: float = 0.2,
        task_fusion_dim: int = 0,
        task_head_dim: int = 0,
        gate_hidden_dim: int = 0,
        aux_head_dim: int = 0,
        fatigue_face_dominant: bool = False,
        task_interaction_dim: int = 0,
        task_attention_heads: int = 0,
        fatigue_refine_dim: int = 0,
        fatigue_residual_dim: int = 0,
        distraction_residual_dim: int = 0,
        fatigue_prior_fusion: bool = False,
        temporal_face_dim: int = 0,
    ) -> None:
        super().__init__()
        # 两个分支都从已有单模态 best.pt 初始化，而不是从随机参数开始。
        self.body_encoder = UltralyticsClassificationEncoder(body_weight)
        self.face_encoder = UltralyticsClassificationEncoder(face_weight)

        # projection 的作用：
        # - 把两个编码器输出先压到统一维度
        # - 顺便给模型一点轻量非线性变换能力
        self.body_proj = nn.Sequential(
            nn.Linear(self.body_encoder.out_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.face_proj = nn.Sequential(
            nn.Linear(self.face_encoder.out_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # B 模块：模态门控融合。
        # 在拼接前先让模型根据当前 body/face 联合上下文，学习两个模态各自应被强调多少。
        self.gate_hidden_dim = int(gate_hidden_dim)
        if self.gate_hidden_dim > 0:
            self.gate_proj = nn.Sequential(
                nn.Linear(proj_dim * 2, self.gate_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.gate_logits = nn.Linear(self.gate_hidden_dim, proj_dim * 2)
        else:
            self.gate_proj = nn.Identity()
            self.gate_logits = nn.Linear(proj_dim * 2, proj_dim * 2)
        # 第一版先用最稳的融合方式：直接拼接两个模态特征，再过一层 MLP。
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # 更强一级的大改动：任务查询注意力融合。
        # 不再手工拼 [body, face, |diff|, product] 做 MLP，
        # 而是让分心/疲劳两个任务各自用一个 query 去注意 body/face 两个模态 token，
        # 再经过各自的 FFN 完成任务解耦的多模态建模。
        self.task_attention_heads = int(task_attention_heads)
        if self.task_attention_heads > 0:
            self.distraction_query = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
            self.fatigue_query = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
            self.distraction_attention = nn.MultiheadAttention(
                embed_dim=proj_dim,
                num_heads=self.task_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.fatigue_attention = nn.MultiheadAttention(
                embed_dim=proj_dim,
                num_heads=self.task_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.distraction_attention_norm1 = nn.LayerNorm(proj_dim)
            self.distraction_attention_ffn = nn.Sequential(
                nn.Linear(proj_dim, proj_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(proj_dim * 2, proj_dim),
                nn.Dropout(dropout),
            )
            self.distraction_attention_norm2 = nn.LayerNorm(proj_dim)
            self.fatigue_attention_norm1 = nn.LayerNorm(proj_dim)
            self.fatigue_attention_ffn = nn.Sequential(
                nn.Linear(proj_dim, proj_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(proj_dim * 2, proj_dim),
                nn.Dropout(dropout),
            )
            self.fatigue_attention_norm2 = nn.LayerNorm(proj_dim)
        else:
            self.distraction_query = None
            self.fatigue_query = None
            self.distraction_attention = nn.Identity()
            self.fatigue_attention = nn.Identity()
            self.distraction_attention_norm1 = nn.Identity()
            self.distraction_attention_ffn = nn.Identity()
            self.distraction_attention_norm2 = nn.Identity()
            self.fatigue_attention_norm1 = nn.Identity()
            self.fatigue_attention_ffn = nn.Identity()
            self.fatigue_attention_norm2 = nn.Identity()
        # 更大改动：完全任务解耦的交互式多模态融合。
        # 每个任务分别接收 [body, face, |body-face|, body*face]，
        # 不再强依赖一份共享融合表示。
        self.task_interaction_dim = int(task_interaction_dim)
        interaction_input_dim = proj_dim * 4
        if self.task_attention_heads > 0:
            self.distraction_interaction = nn.Identity()
            self.fatigue_interaction = nn.Identity()
            self.distraction_fusion = nn.Identity()
            self.fatigue_fusion = nn.Identity()
        elif self.task_interaction_dim > 0:
            self.distraction_interaction = nn.Sequential(
                nn.Linear(interaction_input_dim, self.task_interaction_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.task_interaction_dim, self.task_interaction_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.fatigue_interaction = nn.Sequential(
                nn.Linear(interaction_input_dim, self.task_interaction_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.task_interaction_dim, self.task_interaction_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.distraction_fusion = nn.Identity()
            self.fatigue_fusion = nn.Identity()
        else:
            self.distraction_interaction = nn.Identity()
            self.fatigue_interaction = nn.Identity()
        # A 模块：任务专属融合层。
        # 先做一层共享浅融合，再分别为分心/疲劳构建各自的融合塔，
        # 比“最后一层再分头”更早完成任务解耦。
        self.task_fusion_dim = int(task_fusion_dim)
        task_input_dim = proj_dim * 2 + fusion_dim
        if self.task_attention_heads > 0:
            head_in_dim = proj_dim
        elif self.task_interaction_dim > 0:
            head_in_dim = self.task_interaction_dim
        elif self.task_fusion_dim > 0:
            self.distraction_fusion = nn.Sequential(
                nn.Linear(task_input_dim, self.task_fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.fatigue_fusion = nn.Sequential(
                nn.Linear(task_input_dim, self.task_fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            head_in_dim = self.task_fusion_dim
        else:
            self.distraction_fusion = nn.Identity()
            self.fatigue_fusion = nn.Identity()
            head_in_dim = fusion_dim
        # fatigue face-dominant 分支：以 face 为主，body 只作为可学习的弱辅助提示。
        self.fatigue_face_dominant = bool(fatigue_face_dominant)
        if self.fatigue_face_dominant:
            self.fatigue_body_hint = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.fatigue_hint_gate = nn.Linear(proj_dim * 2, proj_dim)
            self.fatigue_face_fuse = nn.Sequential(
                nn.Linear(proj_dim * 2, head_in_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.fatigue_body_hint = nn.Identity()
            self.fatigue_hint_gate = None
            self.fatigue_face_fuse = nn.Identity()
        # 只增强 fatigue 分支：在主 fatigue 表示之后，再与 face_feat 做一次专属细化。
        # 这样可以保留 v22 对 distraction 的提升，同时只对 fatigue 做额外建模。
        self.fatigue_refine_dim = int(fatigue_refine_dim)
        if self.fatigue_refine_dim > 0:
            self.fatigue_refine_face_proj = nn.Sequential(
                nn.Linear(proj_dim, head_in_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.fatigue_refine = nn.Sequential(
                nn.Linear(head_in_dim * 4, self.fatigue_refine_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.fatigue_refine_dim, head_in_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.fatigue_refine_gate = nn.Linear(head_in_dim * 2, 1)
        else:
            self.fatigue_refine_face_proj = nn.Identity()
            self.fatigue_refine = nn.Identity()
            self.fatigue_refine_gate = None
        # 方案 1：在共享融合层之后，再给每个任务一个轻量专属头。
        # 当 task_head_dim <= 0 时，退化为原来的直接线性分类头，保持旧配置兼容。
        self.task_head_dim = int(task_head_dim)
        if self.task_head_dim > 0:
            self.distraction_tower = nn.Sequential(
                nn.Linear(head_in_dim, self.task_head_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.fatigue_tower = nn.Sequential(
                nn.Linear(head_in_dim, self.task_head_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.distraction_head = nn.Linear(self.task_head_dim, num_distraction_classes)
            self.fatigue_head = nn.Linear(self.task_head_dim, num_fatigue_classes)
        else:
            self.distraction_tower = nn.Identity()
            self.fatigue_tower = nn.Identity()
            self.distraction_head = nn.Linear(head_in_dim, num_distraction_classes)
            self.fatigue_head = nn.Linear(head_in_dim, num_fatigue_classes)
        # B 模块：辅助单模态监督。
        # body 分支保留分心辅助头，face 分支保留疲劳辅助头，
        # 让联合训练时不要把单模态已经学好的能力冲坏。
        self.aux_head_dim = int(aux_head_dim)
        if self.aux_head_dim > 0:
            self.body_aux_tower = nn.Sequential(
                nn.Linear(proj_dim, self.aux_head_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.face_aux_tower = nn.Sequential(
                nn.Linear(proj_dim, self.aux_head_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.body_aux_head = nn.Linear(self.aux_head_dim, num_distraction_classes)
            self.face_aux_head = nn.Linear(self.aux_head_dim, num_fatigue_classes)
        else:
            self.body_aux_tower = nn.Identity()
            self.face_aux_tower = nn.Identity()
            self.body_aux_head = nn.Linear(proj_dim, num_distraction_classes)
            self.face_aux_head = nn.Linear(proj_dim, num_fatigue_classes)

        # 新 B 模块（另一方向）：distraction body 残差专家。
        # 主 distraction 头仍走多模态特征，同时让 body 分支单独输出一份分心校正 logits，
        # 通过可学习 gate 加回最终 distraction 预测，专门缓解 safe_drive 等 body 主导类别被融合冲坏的问题。
        self.distraction_residual_dim = int(distraction_residual_dim)
        if self.distraction_residual_dim > 0:
            self.distraction_residual_tower = nn.Sequential(
                nn.Linear(proj_dim, self.distraction_residual_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.distraction_residual_head = nn.Linear(self.distraction_residual_dim, num_distraction_classes)
            self.distraction_residual_gate = nn.Linear(head_in_dim + proj_dim, 1)
        else:
            self.distraction_residual_tower = nn.Identity()
            self.distraction_residual_head = nn.Linear(proj_dim, num_distraction_classes)
            self.distraction_residual_gate = None

        # 新 B 模块：fatigue face 残差专家。
        # 主 fatigue 头仍然走多模态特征，但再让 face 分支单独输出一份校正 logits，
        # 通过可学习 gate 参与最终 fatigue 预测，避免多模态共享表示把 face 的疲劳判别能力冲坏。
        self.fatigue_residual_dim = int(fatigue_residual_dim)
        if self.fatigue_residual_dim > 0:
            self.fatigue_residual_tower = nn.Sequential(
                nn.Linear(proj_dim, self.fatigue_residual_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.fatigue_residual_head = nn.Linear(self.fatigue_residual_dim, num_fatigue_classes)
            self.fatigue_residual_gate = nn.Linear(head_in_dim + proj_dim, 1)
        else:
            self.fatigue_residual_tower = nn.Identity()
            self.fatigue_residual_head = nn.Linear(proj_dim, num_fatigue_classes)
            self.fatigue_residual_gate = None

        # 新 B：直接复用 train42 的 face fatigue 分类边界，作为多模态 fatigue 的可学习先验。
        self.fatigue_prior_fusion = bool(fatigue_prior_fusion)
        if self.fatigue_prior_fusion:
            self.face_prior_head = load_ultralytics_classifier_linear(face_weight)
            self.fatigue_prior_scale = nn.Parameter(torch.tensor(0.5))
        else:
            self.face_prior_head = None
            self.fatigue_prior_scale = None

        # 新 B：fatigue 短时序 face 模块。
        # 使用 prev/current/next 三帧 face 特征，为 fatigue 提供短时序上下文。
        self.temporal_face_dim = int(temporal_face_dim)
        if self.temporal_face_dim > 0:
            self.temporal_face_tower = nn.Sequential(
                nn.Linear(proj_dim * 5, self.temporal_face_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.temporal_face_dim, head_in_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.temporal_face_tower = nn.Identity()

    def forward(
        self,
        body: torch.Tensor,
        face: torch.Tensor,
        face_prev: torch.Tensor | None = None,
        face_next: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # 先各自编码。
        body_raw_feat = self.body_encoder(body)
        face_raw_feat = self.face_encoder(face)
        body_feat = self.body_proj(body_raw_feat)
        face_feat = self.face_proj(face_raw_feat)
        if self.temporal_face_dim > 0:
            if face_prev is None:
                face_prev = face
            if face_next is None:
                face_next = face
            prev_face_feat = self.face_proj(self.face_encoder(face_prev))
            next_face_feat = self.face_proj(self.face_encoder(face_next))
        else:
            prev_face_feat = face_feat
            next_face_feat = face_feat
        gate_input = torch.cat([body_feat, face_feat], dim=1)
        gate_feat = self.gate_proj(gate_input)
        # 更合理的门控：对每个通道在 body/face 两个模态上做 softmax 竞争，
        # 再用残差式重标定，避免某个模态被直接压成接近 0。
        gate_logits = self.gate_logits(gate_feat).view(gate_feat.shape[0], 2, -1)
        gate_weights = torch.softmax(gate_logits, dim=1)
        body_feat = body_feat * (1.0 + gate_weights[:, 0, :])
        face_feat = face_feat * (1.0 + gate_weights[:, 1, :])
        # 然后做最简单也最稳定的 early fusion / feature fusion。
        multimodal_feat = torch.cat([body_feat, face_feat], dim=1)
        shared_fused = self.fusion(multimodal_feat)
        modality_tokens = torch.stack([body_feat, face_feat], dim=1)
        interaction_feat = torch.cat(
            [body_feat, face_feat, torch.abs(body_feat - face_feat), body_feat * face_feat],
            dim=1,
        )
        task_fusion_input = torch.cat([multimodal_feat, shared_fused], dim=1)
        if self.task_attention_heads > 0:
            distraction_query = self.distraction_query.expand(body_feat.shape[0], -1, -1)
            fatigue_query = self.fatigue_query.expand(body_feat.shape[0], -1, -1)
            distraction_attended, _ = self.distraction_attention(distraction_query, modality_tokens, modality_tokens)
            fatigue_attended, _ = self.fatigue_attention(fatigue_query, modality_tokens, modality_tokens)
            distraction_feat = self.distraction_attention_norm1(distraction_query + distraction_attended).squeeze(1)
            fatigue_feat = self.fatigue_attention_norm1(fatigue_query + fatigue_attended).squeeze(1)
            distraction_feat = self.distraction_attention_norm2(
                distraction_feat + self.distraction_attention_ffn(distraction_feat)
            )
            fatigue_feat = self.fatigue_attention_norm2(fatigue_feat + self.fatigue_attention_ffn(fatigue_feat))
        elif self.task_interaction_dim > 0:
            distraction_feat = self.distraction_interaction(interaction_feat)
            fatigue_feat = self.fatigue_interaction(interaction_feat)
        elif self.task_fusion_dim > 0:
            distraction_feat = self.distraction_fusion(task_fusion_input)
            fatigue_feat = self.fatigue_fusion(task_fusion_input)
        else:
            distraction_feat = shared_fused
            fatigue_feat = shared_fused
        if self.fatigue_face_dominant:
            hint_gate = torch.sigmoid(self.fatigue_hint_gate(torch.cat([face_feat, body_feat], dim=1)))
            body_hint = self.fatigue_body_hint(body_feat) * hint_gate
            fatigue_feat = self.fatigue_face_fuse(torch.cat([face_feat, body_hint], dim=1))
        if self.fatigue_refine_dim > 0:
            face_refine_feat = self.fatigue_refine_face_proj(face_feat)
            refine_input = torch.cat(
                [
                    fatigue_feat,
                    face_refine_feat,
                    torch.abs(fatigue_feat - face_refine_feat),
                    fatigue_feat * face_refine_feat,
                ],
                dim=1,
            )
            refine_delta = self.fatigue_refine(refine_input)
            refine_gate = torch.sigmoid(self.fatigue_refine_gate(torch.cat([fatigue_feat, face_refine_feat], dim=1)))
            fatigue_feat = fatigue_feat + refine_gate * refine_delta
        if self.temporal_face_dim > 0:
            temporal_input = torch.cat(
                [
                    prev_face_feat,
                    face_feat,
                    next_face_feat,
                    face_feat - prev_face_feat,
                    next_face_feat - face_feat,
                ],
                dim=1,
            )
            fatigue_feat = fatigue_feat + self.temporal_face_tower(temporal_input)
        distraction_feat = self.distraction_tower(distraction_feat)
        fatigue_feat = self.fatigue_tower(fatigue_feat)
        body_aux_feat = self.body_aux_tower(body_feat)
        face_aux_feat = self.face_aux_tower(face_feat)
        distraction_logits = self.distraction_head(distraction_feat)
        if self.distraction_residual_dim > 0:
            distraction_residual_feat = self.distraction_residual_tower(body_feat)
            distraction_residual_logits = self.distraction_residual_head(distraction_residual_feat)
            distraction_residual_gate = torch.sigmoid(
                self.distraction_residual_gate(torch.cat([distraction_feat, body_feat], dim=1))
            )
            distraction_logits = distraction_logits + distraction_residual_gate * distraction_residual_logits
        else:
            distraction_residual_logits = self.distraction_residual_head(body_feat)
            distraction_residual_gate = None
        fatigue_logits = self.fatigue_head(fatigue_feat)
        if self.fatigue_residual_dim > 0:
            fatigue_residual_feat = self.fatigue_residual_tower(face_feat)
            fatigue_residual_logits = self.fatigue_residual_head(fatigue_residual_feat)
            residual_gate = torch.sigmoid(self.fatigue_residual_gate(torch.cat([fatigue_feat, face_feat], dim=1)))
            fatigue_logits = fatigue_logits + residual_gate * fatigue_residual_logits
        else:
            fatigue_residual_logits = self.fatigue_residual_head(face_feat)
            residual_gate = None
        if self.fatigue_prior_fusion:
            fatigue_prior_logits = self.face_prior_head(face_raw_feat)
            fatigue_logits = fatigue_logits + self.fatigue_prior_scale * fatigue_prior_logits
        else:
            fatigue_prior_logits = None
        return {
            "distraction_logits": distraction_logits,
            "fatigue_logits": fatigue_logits,
            "body_distraction_logits": self.body_aux_head(body_aux_feat),
            "face_fatigue_logits": self.face_aux_head(face_aux_feat),
            "distraction_residual_logits": distraction_residual_logits,
            "distraction_residual_gate": distraction_residual_gate,
            "fatigue_residual_logits": fatigue_residual_logits,
            "fatigue_residual_gate": residual_gate,
            "fatigue_prior_logits": fatigue_prior_logits,
        }


def set_stage_trainability(model: MultiModalDriverNet, train_body_encoder: bool, train_face_encoder: bool) -> None:
    # 这个函数用来实现“两阶段训练”。
    #
    # 第一阶段：
    #   冻住两个 encoder，只训练 projection / fusion / heads
    # 第二阶段：
    #   再解冻 encoder，一起小学习率微调
    #
    # 这是迁移学习里很常见也很稳的做法。
    for param in model.body_encoder.parameters():
        param.requires_grad = train_body_encoder
    for param in model.face_encoder.parameters():
        param.requires_grad = train_face_encoder

    # 新加的投影层、融合层、任务头始终要训练。
    for module in (
        model.body_proj,
        model.face_proj,
        model.gate_proj,
        model.gate_logits,
        model.fusion,
        model.distraction_attention,
        model.fatigue_attention,
        model.distraction_attention_ffn,
        model.fatigue_attention_ffn,
        model.distraction_attention_norm1,
        model.distraction_attention_norm2,
        model.fatigue_attention_norm1,
        model.fatigue_attention_norm2,
        model.distraction_interaction,
        model.fatigue_interaction,
        model.distraction_fusion,
        model.fatigue_fusion,
        model.fatigue_body_hint,
        model.fatigue_face_fuse,
        model.fatigue_refine_face_proj,
        model.fatigue_refine,
        model.distraction_tower,
        model.fatigue_tower,
        model.distraction_head,
        model.fatigue_head,
        model.body_aux_tower,
        model.face_aux_tower,
        model.body_aux_head,
        model.face_aux_head,
        model.distraction_residual_tower,
        model.distraction_residual_head,
        model.fatigue_residual_tower,
        model.fatigue_residual_head,
        model.temporal_face_tower,
    ):
        for param in module.parameters():
            param.requires_grad = True
    if model.fatigue_hint_gate is not None:
        for param in model.fatigue_hint_gate.parameters():
            param.requires_grad = True
    if model.fatigue_refine_gate is not None:
        for param in model.fatigue_refine_gate.parameters():
            param.requires_grad = True
    if model.distraction_query is not None:
        model.distraction_query.requires_grad = True
    if model.fatigue_query is not None:
        model.fatigue_query.requires_grad = True
    if model.fatigue_residual_gate is not None:
        for param in model.fatigue_residual_gate.parameters():
            param.requires_grad = True
    if model.distraction_residual_gate is not None:
        for param in model.distraction_residual_gate.parameters():
            param.requires_grad = True
    if model.face_prior_head is not None:
        for param in model.face_prior_head.parameters():
            param.requires_grad = True
    if model.fatigue_prior_scale is not None:
        model.fatigue_prior_scale.requires_grad = True


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor | None, int]:
    # 这是本脚本最值得你理解的一个点。
    #
    # 因为不是每条样本都有两个任务的标签，所以不能对所有样本都硬算交叉熵。
    # 这里规定：
    # - labels >= 0：说明这个任务有标签，参与损失
    # - labels == -1：说明这个任务没标签，跳过
    #
    # 返回：
    # - loss: 这个任务在当前 batch 上的损失；如果全都没标签，则返回 None
    # - count: 当前 batch 有多少条样本真的参与了这个任务
    mask = labels >= 0
    if not mask.any():
        return None, 0
    return F.cross_entropy(logits[mask], labels[mask]), int(mask.sum().item())


def masked_focal_loss(
    logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0
) -> tuple[torch.Tensor | None, int]:
    mask = labels >= 0
    if not mask.any():
        return None, 0
    masked_logits = logits[mask]
    masked_labels = labels[mask]
    ce = F.cross_entropy(masked_logits, masked_labels, reduction="none")
    pt = torch.exp(-ce)
    focal = ((1.0 - pt) ** gamma) * ce
    return focal.mean(), int(mask.sum().item())


def compute_batch_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    # 训练日志里的准确率也同样只在“有标签”的样本上统计。
    mask = labels >= 0
    if not mask.any():
        return 0, 0
    preds = logits[mask].argmax(dim=1)
    correct = int((preds == labels[mask]).sum().item())
    total = int(mask.sum().item())
    return correct, total


def run_epoch(
    model: MultiModalDriverNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    distraction_loss_weight: float = 1.0,
    fatigue_loss_weight: float = 1.0,
    distraction_focal_gamma: float = 0.0,
    fatigue_focal_gamma: float = 0.0,
    aux_distraction_loss_weight: float = 0.0,
    aux_fatigue_loss_weight: float = 0.0,
) -> dict[str, float]:
    # optimizer 为 None 表示验证；不为 None 表示训练。
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0
    distraction_correct = 0
    distraction_total = 0
    fatigue_correct = 0
    fatigue_total = 0

    progress = tqdm(loader, leave=False)
    for batch in progress:
        # 把 batch 搬到 GPU/CPU。
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

        # 前向传播：一次同时得到两个任务的 logits。
        outputs = model(body, face, face_prev=face_prev, face_next=face_next)
        if distraction_focal_gamma > 0:
            distraction_loss, distraction_count = masked_focal_loss(
                outputs["distraction_logits"], distraction_label, gamma=distraction_focal_gamma
            )
        else:
            distraction_loss, _distraction_count = masked_cross_entropy(
                outputs["distraction_logits"], distraction_label
            )
        if fatigue_focal_gamma > 0:
            fatigue_loss, fatigue_count = masked_focal_loss(
                outputs["fatigue_logits"], fatigue_label, gamma=fatigue_focal_gamma
            )
        else:
            fatigue_loss, _fatigue_count = masked_cross_entropy(outputs["fatigue_logits"], fatigue_label)
        body_aux_loss, _ = masked_cross_entropy(outputs["body_distraction_logits"], distraction_label)
        face_aux_loss, _ = masked_cross_entropy(outputs["face_fatigue_logits"], fatigue_label)

        # 某个 batch 可能只来自单一任务，所以这里只把有效 loss 加起来。
        losses = []
        if distraction_loss is not None:
            losses.append(distraction_loss_weight * distraction_loss)
        if fatigue_loss is not None:
            losses.append(fatigue_loss_weight * fatigue_loss)
        if body_aux_loss is not None and aux_distraction_loss_weight > 0:
            losses.append(aux_distraction_loss_weight * body_aux_loss)
        if face_aux_loss is not None and aux_fatigue_loss_weight > 0:
            losses.append(aux_fatigue_loss_weight * face_aux_loss)
        if not losses:
            continue

        loss = sum(losses)

        if is_train:
            # 标准训练三件套：
            # 1. 清梯度
            # 2. 反向传播
            # 3. 参数更新
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        # 这里分开统计两个任务的 batch-level 正确数。
        dc, dt = compute_batch_metrics(outputs["distraction_logits"], distraction_label)
        fc, ft = compute_batch_metrics(outputs["fatigue_logits"], fatigue_label)
        distraction_correct += dc
        distraction_total += dt
        fatigue_correct += fc
        fatigue_total += ft

        # tqdm 上显示的是当前 batch 的局部情况，不是整轮的最终结果。
        progress.set_description(f"loss={loss.item():.4f} d_acc={dc}/{max(dt, 1)} f_acc={fc}/{max(ft, 1)}")

    # 返回整轮统计。
    return {
        "loss": total_loss / max(total_batches, 1),
        "distraction_acc": distraction_correct / distraction_total if distraction_total else 0.0,
        "fatigue_acc": fatigue_correct / fatigue_total if fatigue_total else 0.0,
        "distraction_count": distraction_total,
        "fatigue_count": fatigue_total,
    }


def build_transforms(cfg: dict, mode: str):
    # 训练和验证使用不同的图像预处理：
    # - train: 带轻量增强
    # - val/test: 只做确定性 resize/crop/normalize
    if mode == "train":
        return classify_augmentations(
            size=cfg["imgsz"],
            scale=(cfg["scale"], 1.0),
            hflip=cfg["fliplr"],
            vflip=cfg["flipud"],
            auto_augment=cfg["auto_augment"],
            hsv_h=cfg["hsv_h"],
            hsv_s=cfg["hsv_s"],
            hsv_v=cfg["hsv_v"],
            erasing=cfg["erasing"],
        )
    return classify_transforms(size=cfg["imgsz"])


def save_checkpoint(path: Path, model: nn.Module, epoch: int, metrics: dict, config: dict) -> None:
    # 这里只保存 state_dict，而不是直接保存整个模型对象。
    # 这样后面兼容性更好，也更符合 PyTorch 常见做法。
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def main() -> None:
    # 入口只接受一个参数：配置文件路径。
    parser = argparse.ArgumentParser(description="Train multimodal driver model v1.")
    parser.add_argument("--config", type=str, default="train_multimodal_v1.yaml")
    args = parser.parse_args()

    # 先读配置，再固定随机种子。
    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))

    # metadata.json 里保存了类别名称和数据集格式信息。
    dataset_root = Path(cfg["data_root"])
    metadata = json.loads((dataset_root / "metadata.json").read_text(encoding="utf-8"))
    distraction_classes = metadata["distraction_classes"]
    fatigue_classes = metadata["fatigue_classes"]

    # train/val 各自构建一套数据集对象。
    # 注意 body 和 face 使用的是同一种增强策略，但它们是分别应用到各自图像上的。
    train_dataset = MultiModalDriverDataset(
        csv_path=dataset_root / "train.csv",
        distraction_classes=distraction_classes,
        fatigue_classes=fatigue_classes,
        body_transform=build_transforms(cfg, "train"),
        face_transform=build_transforms(cfg, "train"),
        temporal_face=bool(cfg.get("temporal_face_dim", 0)),
    )
    val_dataset = MultiModalDriverDataset(
        csv_path=dataset_root / "val.csv",
        distraction_classes=distraction_classes,
        fatigue_classes=fatigue_classes,
        body_transform=build_transforms(cfg, "val"),
        face_transform=build_transforms(cfg, "val"),
        temporal_face=bool(cfg.get("temporal_face_dim", 0)),
    )

    # DataLoader 的职责是：
    # - 按 batch 取样
    # - 多线程读图
    # - 调用 collate_fn 拼成 batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch"]),
        shuffle=True,
        num_workers=int(cfg["workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        collate_fn=multimodal_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["batch"]),
        shuffle=False,
        num_workers=int(cfg["workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        collate_fn=multimodal_collate_fn,
        drop_last=False,
    )

    # 构建多模态网络，并指定两个分支分别从哪个 best.pt 初始化。
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

    # 优先使用 CUDA；如果配置里写的是 cpu，就强制用 cpu。
    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg["device"]) != "cpu" else "cpu")
    if str(cfg["device"]).isdigit() and device.type == "cuda":
        device = torch.device(f"cuda:{cfg['device']}")
    model = model.to(device)

    # 为本次训练创建独立输出目录，避免覆盖上一轮结果。
    save_dir = increment_path(Path(cfg["project"]) / cfg["name"], exist_ok=bool(cfg.get("exist_ok", False)), mkdir=True)
    save_yaml(save_dir / "args.yaml", cfg)
    (save_dir / "weights").mkdir(parents=True, exist_ok=True)

    # 结果表格里每行对应一个 epoch，方便你后面用 Excel 或 pandas 直接分析。
    results_path = save_dir / "results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stage",
                "epoch",
                "train_loss",
                "train_distraction_acc",
                "train_fatigue_acc",
                "val_loss",
                "val_distraction_acc",
                "val_fatigue_acc",
                "val_score",
            ]
        )

    best_score = -1.0
    global_epoch = 0
    best_metrics: dict[str, float] = {}

    # 这里开始按 stage 训练。
    # stage 的好处是：你可以很明确地控制“先训哪里，再训哪里”。
    for stage in cfg["stages"]:
        set_stage_trainability(
            model,
            train_body_encoder=bool(stage["train_body_encoder"]),
            train_face_encoder=bool(stage["train_face_encoder"]),
        )

        # 只把 requires_grad=True 的参数交给优化器。
        # 这样被冻结的层就不会被更新。
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(stage["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )

        patience = int(stage["patience"])
        patience_counter = 0
        stage_best = -1.0

        for _ in range(int(stage["epochs"])):
            global_epoch += 1
            # 先训练一轮，再验证一轮。
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                distraction_loss_weight=float(cfg.get("distraction_loss_weight", 1.0)),
                fatigue_loss_weight=float(cfg.get("fatigue_loss_weight", 1.0)),
                distraction_focal_gamma=float(cfg.get("distraction_focal_gamma", 0.0)),
                fatigue_focal_gamma=float(cfg.get("fatigue_focal_gamma", 0.0)),
                aux_distraction_loss_weight=float(cfg.get("aux_distraction_loss_weight", 0.0)),
                aux_fatigue_loss_weight=float(cfg.get("aux_fatigue_loss_weight", 0.0)),
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    device,
                    optimizer=None,
                    distraction_loss_weight=float(cfg.get("distraction_loss_weight", 1.0)),
                    fatigue_loss_weight=float(cfg.get("fatigue_loss_weight", 1.0)),
                    distraction_focal_gamma=float(cfg.get("distraction_focal_gamma", 0.0)),
                    fatigue_focal_gamma=float(cfg.get("fatigue_focal_gamma", 0.0)),
                    aux_distraction_loss_weight=float(cfg.get("aux_distraction_loss_weight", 0.0)),
                    aux_fatigue_loss_weight=float(cfg.get("aux_fatigue_loss_weight", 0.0)),
                )

            # 这里用了一个非常简单的联合评分：
            # 分心准确率和疲劳准确率各占 50%。
            # 这样第一版里，两个任务都能被纳入“best.pt”选择标准。
            val_score = 0.5 * (val_metrics["distraction_acc"] + val_metrics["fatigue_acc"])
            metrics = {
                "stage": stage["name"],
                "epoch": global_epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "val_score": val_score,
            }

            with open(results_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        stage["name"],
                        global_epoch,
                        train_metrics["loss"],
                        train_metrics["distraction_acc"],
                        train_metrics["fatigue_acc"],
                        val_metrics["loss"],
                        val_metrics["distraction_acc"],
                        val_metrics["fatigue_acc"],
                        val_score,
                    ]
                )

            # 训练时打印最关键的四个指标，便于你肉眼追踪。
            print(
                f"[{stage['name']}] epoch={global_epoch} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_distraction_acc={val_metrics['distraction_acc']:.4f} "
                f"val_fatigue_acc={val_metrics['fatigue_acc']:.4f} "
                f"val_score={val_score:.4f}"
            )

            # last.pt 每轮都覆盖保存，方便中断后继续分析当前最新状态。
            save_checkpoint(save_dir / "weights" / "last.pt", model, global_epoch, metrics, cfg)

            # best.pt 只在综合验证指标更好时更新。
            if val_score > best_score:
                best_score = val_score
                best_metrics = metrics
                save_checkpoint(save_dir / "weights" / "best.pt", model, global_epoch, metrics, cfg)

            # 这是 stage 内部的 early stopping。
            # 如果一个 stage 连续若干轮都没有提升，就提前结束，进入下一个 stage。
            if val_score > stage_best:
                stage_best = val_score
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping stage '{stage['name']}' after {patience} stale epochs.")
                    break

    # best_metrics.json 保存了最终最佳轮次的摘要，后面看结果很方便。
    with open(save_dir / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)

    print(f"Training complete. Results saved to {save_dir}")


if __name__ == "__main__":
    main()

# python train_multimodal_v1.py --config train_multimodal_v1.yaml
