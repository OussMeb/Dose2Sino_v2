#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanity_overfit_same_patient_attention_in_agg_tunable.py

Tunable same-patient sanity check for the accepted baseline:

    Attention + InstanceNorm3D + learned Conv3D aggregation

Goal:
    Tune important baseline parameters safely without editing model/training files.

Main tunable parameters:
    activation
    activation_scale
    base_filters
    learning_rate
    loss weights
    epochs
    scheduler
    AMP

Run:
    python sanity_overfit_same_patient_attention_in_agg_tunable.py \
      --config configs/sanity_baseline_tuning.yaml

Override examples:
    python sanity_overfit_same_patient_attention_in_agg_tunable.py \
      --config configs/sanity_baseline_tuning.yaml \
      --activation scaled_sigmoid \
      --activation-scale 1.5 \
      --open-weight 8 \
      --learning-rate 1e-4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

from models.unet_attention_in_agg import DosePredictionAttentionInAgg
from utils.patient import RTDataset


PATCH_VERSION = "baseline_tunable_sanity_v1"


class SamePatientDataset(Dataset):
    def __init__(self, base_dataset: RTDataset, indices: list[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.base_dataset[self.indices[idx]]


class TunableActivation(nn.Module):
    """
    Convert raw model output to [0, 1] prediction.

    sigmoid:
        standard bounded output

    scaled_sigmoid:
        sigmoid(scale * raw), sharper for high amplitudes

    calibrated_sigmoid:
        sigmoid(scale * raw + bias), useful for amplitude calibration

    hard_sigmoid:
        clamp((scale * raw + bias + 1) / 2, 0, 1)

    softplus_squash:
        softplus(z) / (1 + softplus(z)), bounded and smoother near 0

    clamp:
        clamp(scale * raw + bias, 0, 1)
    """

    def __init__(self, name: str = "sigmoid", scale: float = 1.0, bias: float = 0.0):
        super().__init__()
        self.name = str(name)
        self.scale = float(scale)
        self.bias = float(bias)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        z = self.scale * raw + self.bias

        if self.name == "sigmoid":
            return torch.sigmoid(z)

        if self.name == "scaled_sigmoid":
            return torch.sigmoid(z)

        if self.name == "calibrated_sigmoid":
            return torch.sigmoid(z)

        if self.name == "hard_sigmoid":
            return torch.clamp((z + 1.0) / 2.0, 0.0, 1.0)

        if self.name == "softplus_squash":
            s = F.softplus(z)
            return s / (1.0 + s)

        if self.name == "clamp":
            return torch.clamp(z, 0.0, 1.0)

        raise ValueError(
            f"Unknown activation={self.name!r}. "
            "Choose: sigmoid, scaled_sigmoid, calibrated_sigmoid, hard_sigmoid, softplus_squash, clamp."
        )


class TunableSinogramLoss(nn.Module):
    """
    Probability-space loss for activation tuning.

    This intentionally works after activation so different output activations
    can be compared under the same metric/loss logic.
    """

    def __init__(
        self,
        open_threshold: float = 1e-3,
        charbonnier_eps: float = 1e-3,
        closed_weight: float = 1.0,
        open_weight: float = 5.0,
        high_value_weight: float = 2.0,
        high_value_power: float = 2.0,
        open_l1_weight: float = 1.0,
        closed_penalty_weight: float = 0.2,
        gradient_weight: float = 0.1,
        bce_weight: float = 0.0,
    ):
        super().__init__()
        self.open_threshold = float(open_threshold)
        self.eps2 = float(charbonnier_eps) ** 2
        self.closed_weight = float(closed_weight)
        self.open_weight = float(open_weight)
        self.high_value_weight = float(high_value_weight)
        self.high_value_power = float(high_value_power)
        self.open_l1_weight = float(open_l1_weight)
        self.closed_penalty_weight = float(closed_penalty_weight)
        self.gradient_weight = float(gradient_weight)
        self.bce_weight = float(bce_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        open_mask = target > self.open_threshold
        closed_mask = ~open_mask

        weights = (
            self.closed_weight
            + self.open_weight * open_mask.float()
            + self.high_value_weight * torch.pow(torch.clamp(target, 0.0, 1.0), self.high_value_power)
        )

        charbonnier = torch.sqrt((pred - target) ** 2 + self.eps2)
        recon = torch.sum(weights * charbonnier) / torch.clamp(weights.sum(), min=1.0)

        if open_mask.any():
            open_l1 = torch.mean(torch.abs(pred[open_mask] - target[open_mask]))
        else:
            open_l1 = torch.zeros((), device=pred.device, dtype=pred.dtype)

        if closed_mask.any():
            closed_penalty = torch.mean(pred[closed_mask] ** 2)
        else:
            closed_penalty = torch.zeros((), device=pred.device, dtype=pred.dtype)

        grad = self.gradient_loss(pred, target)

        if self.bce_weight > 0:
            pred_clamped = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
            target_open = open_mask.float()

            with torch.amp.autocast(device_type=pred.device.type, enabled=False):
                bce = F.binary_cross_entropy(
                    pred_clamped.float(),
                    target_open.float(),
                ).to(pred.dtype)
        else:
            bce = torch.zeros((), device=pred.device, dtype=pred.dtype)

        total = (
            recon
            + self.open_l1_weight * open_l1
            + self.closed_penalty_weight * closed_penalty
            + self.gradient_weight * grad
            + self.bce_weight * bce
        )

        return total, {
            "loss": float(total.detach().item()),
            "recon": float(recon.detach().item()),
            "open_l1_loss": float(open_l1.detach().item()),
            "closed_penalty": float(closed_penalty.detach().item()),
            "gradient_loss": float(grad.detach().item()),
            "bce": float(bce.detach().item()),
        }

    @staticmethod
    def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cp_pred = pred[:, 1:, :] - pred[:, :-1, :]
        cp_target = target[:, 1:, :] - target[:, :-1, :]

        leaf_pred = pred[:, :, 1:] - pred[:, :, :-1]
        leaf_target = target[:, :, 1:] - target[:, :, :-1]

        return F.l1_loss(cp_pred, cp_target) + F.l1_loss(leaf_pred, leaf_target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tunable same-patient sanity check for accepted baseline.")
    parser.add_argument("--config", type=str, default="configs/sanity_baseline_tuning.yaml")

    parser.add_argument("--data-path", type=str)
    parser.add_argument("--cache-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--patient-id", type=str)

    parser.add_argument("--base-filters", type=int)
    parser.add_argument("--attention-kernel-size", type=int)
    parser.add_argument("--detector-width", type=int)
    parser.add_argument("--activation", type=str)
    parser.add_argument("--activation-scale", type=float)
    parser.add_argument("--activation-bias", type=float)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--device", type=str, choices=("auto", "cuda", "cpu"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--open-weight", type=float)
    parser.add_argument("--closed-weight", type=float)
    parser.add_argument("--high-value-weight", type=float)
    parser.add_argument("--open-l1-weight", type=float)
    parser.add_argument("--closed-penalty-weight", type=float)
    parser.add_argument("--gradient-weight", type=float)
    parser.add_argument("--bce-weight", type=float)

    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping at root: {path}")
    return data


def deep_update(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = config
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def merge_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(config)

    mapping = {
        "data_path": ("data", "data_path"),
        "cache_dir": ("data", "cache_dir"),
        "output_dir": ("output", "output_dir"),
        "patient_id": ("data", "patient_id"),
        "base_filters": ("model", "base_filters"),
        "attention_kernel_size": ("model", "attention_kernel_size"),
        "detector_width": ("model", "detector_width"),
        "activation": ("model", "activation"),
        "activation_scale": ("model", "activation_scale"),
        "activation_bias": ("model", "activation_bias"),
        "epochs": ("training", "epochs"),
        "learning_rate": ("training", "learning_rate"),
        "weight_decay": ("training", "weight_decay"),
        "save_every": ("training", "save_every"),
        "log_every": ("training", "log_every"),
        "device": ("training", "device"),
        "seed": ("training", "seed"),
        "open_weight": ("loss", "open_weight"),
        "closed_weight": ("loss", "closed_weight"),
        "high_value_weight": ("loss", "high_value_weight"),
        "open_l1_weight": ("loss", "open_l1_weight"),
        "closed_penalty_weight": ("loss", "closed_penalty_weight"),
        "gradient_weight": ("loss", "gradient_weight"),
        "bce_weight": ("loss", "bce_weight"),
    }

    for attr, path in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            deep_update(cfg, path, value)

    if args.amp:
        deep_update(cfg, ("training", "amp"), True)
    if args.no_amp:
        deep_update(cfg, ("training", "amp"), False)
    if args.no_cache:
        deep_update(cfg, ("data", "use_cache"), False)

    return cfg


def get_cfg(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return config.get(section, {}).get(key, default)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(output_dir / "same_patient_tunable.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(file_handler)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_patient_indices(dataset: RTDataset, patient_id: str) -> list[int]:
    indices = [
        idx for idx, sample in enumerate(dataset.samples)
        if str(sample.get("patient_id")) == str(patient_id)
    ]
    if not indices:
        available = sorted({str(sample.get("patient_id")) for sample in dataset.samples})
        raise ValueError(f"Patient {patient_id!r} not found. Available preview: {available[:20]}")
    return sorted(indices, key=lambda idx: int(dataset.samples[idx].get("pareto_index", idx)))


def align_target(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 5 and target.shape[1] == 1 and target.shape[-1] == 1:
        return target[:, 0, :, :, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        return target[:, 0, :, :]
    if target.ndim == 3:
        return target
    raise ValueError(f"Unexpected target shape: {tuple(target.shape)}")


def align_raw_output(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 5 and output.shape[1] == 1 and output.shape[-1] == 1:
        return output[:, 0, :, :, 0]
    if output.ndim == 4 and output.shape[1] == 1:
        return output[:, 0, :, :]
    if output.ndim == 3:
        return output
    raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}")


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return torch.amp.autocast(device_type="cpu", enabled=False)


def batch_patient_id(batch: dict[str, Any]) -> str:
    value = batch.get("patient_id", "unknown")
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def batch_pareto_index(batch: dict[str, Any]) -> str:
    value = batch.get("pareto_index", "unknown")
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return str(int(value.item()))
        return "_".join(str(int(x)) for x in value.flatten().tolist())
    if isinstance(value, (list, tuple)):
        first = value[0]
        if isinstance(first, torch.Tensor) and first.numel() == 1:
            return str(int(first.item()))
        return str(first)
    return str(value)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, loss_value: float, components: dict[str, float]) -> dict[str, float]:
    with torch.no_grad():
        diff = torch.abs(pred - target)
        open_mask = target > 1e-6
        closed_mask = ~open_mask

        open_l1 = diff[open_mask].mean() if open_mask.any() else torch.tensor(float("nan"), device=target.device)
        closed_abs_pred = pred[closed_mask].mean() if closed_mask.any() else torch.tensor(float("nan"), device=target.device)

        metrics = {
            "loss": float(loss_value),
            "mae": float(diff.mean().item()),
            "max_abs": float(diff.max().item()),
            "open_l1": float(open_l1.item()),
            "closed_abs_pred": float(closed_abs_pred.item()),
            "target_open_fraction": float(open_mask.float().mean().item()),
            "pred_min": float(pred.min().item()),
            "pred_max": float(pred.max().item()),
            "pred_mean": float(pred.mean().item()),
            "target_mean": float(target.mean().item()),
        }
        metrics.update(components)
        return metrics


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    out: dict[str, float] = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row and not np.isnan(row[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_visualization(
    path: Path,
    batch: dict[str, Any],
    pred: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    metrics: dict[str, float],
    max_dose: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    pred_np = pred[0].detach().float().cpu().numpy()
    target_np = target[0].detach().float().cpu().numpy()
    diff_np = np.abs(pred_np - target_np)

    inp = batch["input"][0].detach().float().cpu()
    n_cp = int(inp.shape[1])
    mid = n_cp // 2

    ct = inp[0, mid].numpy()
    dose = inp[1, mid].numpy() * float(max_dose)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def show(ax, img, title, cmap, vmin=None, vmax=None):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    show(axes[0, 0], pred_np, f"Prediction epoch {epoch}", "hot", 0, 1)
    show(axes[0, 1], target_np, f"GT open_l1={metrics.get('open_l1', float('nan')):.5f}", "hot", 0, 1)
    show(axes[0, 2], diff_np, "|pred-GT|", "RdYlGn_r", 0, 0.5)
    show(axes[1, 0], ct, "CT Berlingo", "gray", 0, 1)
    show(axes[1, 1], dose, "Dose Berlingo [Gy]", "hot", 0, max_dose)

    axes[1, 2].hist(target_np.ravel(), bins=80, range=(0, 1), alpha=0.55, label="GT")
    axes[1, 2].hist(pred_np.ravel(), bins=80, range=(0, 1), alpha=0.55, label="Pred")
    axes[1, 2].set_title("Value histogram")
    axes[1, 2].legend()

    plt.suptitle(
        f"Patient: {batch_patient_id(batch)} | Pareto: {batch_pareto_index(batch)} | Slice: {mid}/{n_cp}",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "patch_version": PATCH_VERSION,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def evaluate(
    model: nn.Module,
    activation: TunableActivation,
    loss_fn: TunableSinogramLoss,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    rows = []
    per_sample = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            target = align_target(batch["target"].to(device, non_blocking=True))

            with autocast_context(device, amp):
                raw = align_raw_output(model(inputs))
                pred = activation(raw)
                loss, components = loss_fn(pred, target)

            metrics = compute_metrics(pred, target, float(loss.item()), components)
            rows.append(metrics)
            per_sample.append({
                "patient_id": batch_patient_id(batch),
                "pareto_index": batch_pareto_index(batch),
                **metrics,
            })

            del inputs, target, raw, pred, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return mean_metrics(rows), per_sample


def save_final_arrays(
    model: nn.Module,
    activation: TunableActivation,
    loss_fn: TunableSinogramLoss,
    loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    amp: bool,
    epoch: int,
    max_dose: float,
) -> None:
    final_dir = output_dir / "visualizations" / "final_all_paretos"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            target = align_target(batch["target"].to(device, non_blocking=True))

            with autocast_context(device, amp):
                raw = align_raw_output(model(inputs))
                pred = activation(raw)
                loss, components = loss_fn(pred, target)

            metrics = compute_metrics(pred, target, float(loss.item()), components)
            pareto = batch_pareto_index(batch)

            save_visualization(
                final_dir / f"pareto_{pareto}_epoch_{epoch:04d}.png",
                batch=batch,
                pred=pred,
                target=target,
                epoch=epoch,
                metrics=metrics,
                max_dose=max_dose,
            )

            np.save(final_dir / f"pareto_{pareto}_pred_prob.npy", pred[0].detach().float().cpu().numpy())
            np.save(final_dir / f"pareto_{pareto}_target.npy", target[0].detach().float().cpu().numpy())

            del inputs, target, raw, pred, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    raw_config = load_yaml_config(args.config)
    config = merge_cli_overrides(raw_config, args)

    requested_device = str(get_cfg(config, "training", "device", "auto"))
    device = resolve_device(requested_device)
    seed = int(get_cfg(config, "training", "seed", 42))
    set_seed(seed)

    patient_id = str(get_cfg(config, "data", "patient_id", "324181"))
    base_output = Path(str(get_cfg(config, "output", "output_dir", "sanity_outputs/baseline_tuning"))).expanduser()
    run_name = (
        f"patient_{patient_id}_"
        f"act_{get_cfg(config, 'model', 'activation', 'sigmoid')}_"
        f"bf{get_cfg(config, 'model', 'base_filters', 16)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = base_output / run_name

    setup_logging(output_dir)

    config["runtime"] = {
        "device": str(device),
        "patch_version": PATCH_VERSION,
        "config_file": str(Path(args.config).expanduser()),
    }
    (output_dir / "merged_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "merged_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    logging.info("========== TUNABLE SAME-PATIENT SANITY ==========")
    logging.info("Config:\n%s", json.dumps(config, indent=2))

    data_path = str(get_cfg(config, "data", "data_path", "/mnt/data/shared/tomo_data/"))
    cache_dir = str(get_cfg(config, "data", "cache_dir", "/mnt/data/shared/tomo_data/cache_sino"))
    reduction_ratio = int(get_cfg(config, "data", "reduction_ratio", 8))
    max_dose = float(get_cfg(config, "data", "max_dose", 70.0))
    use_cache = bool(get_cfg(config, "data", "use_cache", True))

    dataset = RTDataset(
        root_dir=data_path,
        augmentation=None,
        max_dose=max_dose,
        reduction_ratio=reduction_ratio,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )

    indices = get_patient_indices(dataset, patient_id)
    manifest = [
        {
            "dataset_index": idx,
            "patient_id": str(dataset.samples[idx].get("patient_id")),
            "pareto_index": str(dataset.samples[idx].get("pareto_index")),
            "plan_file": str(dataset.samples[idx].get("plan_file")),
            "dose_file": str(dataset.samples[idx].get("dose_file")),
        }
        for idx in indices
    ]
    (output_dir / "selected_samples.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logging.info("Selected patient %s with %d pareto samples.", patient_id, len(indices))

    batch_size = int(get_cfg(config, "training", "batch_size", 1))
    num_workers = int(get_cfg(config, "training", "num_workers", 0))

    sanity_dataset = SamePatientDataset(dataset, indices)
    train_loader = DataLoader(
        sanity_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        sanity_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DosePredictionAttentionInAgg(
        base_filters=int(get_cfg(config, "model", "base_filters", 16)),
        in_channel=2,
        attention_kernel_size=int(get_cfg(config, "model", "attention_kernel_size", 15)),
        detector_width=int(get_cfg(config, "model", "detector_width", 64)),
    ).to(device)

    activation = TunableActivation(
        name=str(get_cfg(config, "model", "activation", "sigmoid")),
        scale=float(get_cfg(config, "model", "activation_scale", 1.0)),
        bias=float(get_cfg(config, "model", "activation_bias", 0.0)),
    ).to(device)

    loss_fn = TunableSinogramLoss(
        open_threshold=float(get_cfg(config, "loss", "open_threshold", 1e-3)),
        charbonnier_eps=float(get_cfg(config, "loss", "charbonnier_eps", 1e-3)),
        closed_weight=float(get_cfg(config, "loss", "closed_weight", 1.0)),
        open_weight=float(get_cfg(config, "loss", "open_weight", 5.0)),
        high_value_weight=float(get_cfg(config, "loss", "high_value_weight", 2.0)),
        high_value_power=float(get_cfg(config, "loss", "high_value_power", 2.0)),
        open_l1_weight=float(get_cfg(config, "loss", "open_l1_weight", 1.0)),
        closed_penalty_weight=float(get_cfg(config, "loss", "closed_penalty_weight", 0.2)),
        gradient_weight=float(get_cfg(config, "loss", "gradient_weight", 0.1)),
        bce_weight=float(get_cfg(config, "loss", "bce_weight", 0.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(get_cfg(config, "training", "learning_rate", 1e-4)),
        weight_decay=float(get_cfg(config, "training", "weight_decay", 0.0)),
    )

    scheduler_cfg = config.get("scheduler", {})
    scheduler = None
    if bool(scheduler_cfg.get("enabled", True)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 10)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )

    amp = bool(get_cfg(config, "training", "amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp else None

    epochs = int(get_cfg(config, "training", "epochs", 150))
    log_every = int(get_cfg(config, "training", "log_every", 1))
    save_every = int(get_cfg(config, "training", "save_every", 10))
    grad_clip = float(get_cfg(config, "training", "grad_clip", 1.0))

    best_eval = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_rows = []

        for batch in train_loader:
            inputs = batch["input"].to(device, non_blocking=True)
            target = align_target(batch["target"].to(device, non_blocking=True))

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, amp):
                raw = align_raw_output(model(inputs))
                pred = activation(raw)
                loss, components = loss_fn(pred, target)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Invalid loss at epoch {epoch}, patient={batch_patient_id(batch)}.")

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            train_rows.append(compute_metrics(pred.detach(), target.detach(), float(loss.item()), components))

            del inputs, target, raw, pred, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

        train_metrics = mean_metrics(train_rows)
        eval_metrics, per_sample = evaluate(model, activation, loss_fn, eval_loader, device, amp)

        if scheduler is not None:
            scheduler.step(eval_metrics["loss"])

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"eval_{k}": v for k, v in eval_metrics.items()},
        }
        history.append(row)
        write_csv(output_dir / "loss_history.csv", history)
        write_csv(output_dir / "per_sample_metrics_latest.csv", per_sample)

        if epoch % log_every == 0 or epoch == 1:
            logging.info(
                "Epoch %04d/%04d | train_loss=%.6f | eval_loss=%.6f | "
                "eval_mae=%.6f | eval_open_l1=%.6f | eval_closed_abs_pred=%.6f | "
                "pred_mean=%.6f | target_mean=%.6f | lr=%.6g",
                epoch,
                epochs,
                train_metrics.get("loss", float("nan")),
                eval_metrics.get("loss", float("nan")),
                eval_metrics.get("mae", float("nan")),
                eval_metrics.get("open_l1", float("nan")),
                eval_metrics.get("closed_abs_pred", float("nan")),
                eval_metrics.get("pred_mean", float("nan")),
                eval_metrics.get("target_mean", float("nan")),
                optimizer.param_groups[0]["lr"],
            )

        if eval_metrics["loss"] < best_eval:
            best_eval = eval_metrics["loss"]
            save_checkpoint(output_dir / "best_checkpoint.pt", model, optimizer, scheduler, epoch, eval_metrics, config)
            write_csv(output_dir / "per_sample_metrics_best.csv", per_sample)

        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                eval_metrics,
                config,
            )

            first_batch = next(iter(eval_loader))
            inputs = first_batch["input"].to(device, non_blocking=True)
            target = align_target(first_batch["target"].to(device, non_blocking=True))
            model.eval()
            with torch.no_grad():
                with autocast_context(device, amp):
                    raw = align_raw_output(model(inputs))
                    pred = activation(raw)
                    loss, components = loss_fn(pred, target)
            metrics = compute_metrics(pred, target, float(loss.item()), components)
            save_visualization(
                output_dir / "visualizations" / f"epoch_{epoch:04d}_first_pareto.png",
                first_batch,
                pred,
                target,
                epoch,
                metrics,
                max_dose,
            )

            del inputs, target, raw, pred, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

    final_metrics, final_per_sample = evaluate(model, activation, loss_fn, eval_loader, device, amp)
    write_csv(output_dir / "per_sample_metrics_final.csv", final_per_sample)
    save_checkpoint(output_dir / "final_checkpoint.pt", model, optimizer, scheduler, epochs, final_metrics, config)

    save_final_arrays(
        model=model,
        activation=activation,
        loss_fn=loss_fn,
        loader=eval_loader,
        output_dir=output_dir,
        device=device,
        amp=amp,
        epoch=epochs,
        max_dose=max_dose,
    )

    logging.info("========== DONE ==========")
    logging.info("Best eval loss: %.6f", best_eval)
    logging.info("Final metrics:\n%s", json.dumps(final_metrics, indent=2))
    logging.info("Outputs saved in: %s", output_dir)


if __name__ == "__main__":
    main()
