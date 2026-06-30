#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_delta_residual_gan_full.py

Supervised residual-delta refinement with optional delta-PatchGAN.

Main idea:
    baseline = sigmoid(frozen_generator(input))
    delta_target = clamp(GT - baseline, -delta_scale, +delta_scale)
    delta_pred = refiner(condition_maps, baseline)
    refined = clamp(baseline + delta_pred, 0, 1)

The discriminator, when enabled, judges delta maps:
    real = delta_target
    fake = delta_pred

Use --adv-weight 0 for pure supervised residual-delta learning.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
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
from torch.utils.data import DataLoader, Subset

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from models.sino_patchgan_discriminator import SinoPatchGANDiscriminator
from models.sino_residual_refiner import SinoResidualRefiner2D
from models.unet_attention_in_agg import DosePredictionAttentionInAgg
from utils.patient import RTDataset


PATCH_VERSION = "supervised_delta_residual_optional_gan_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised delta residual refinement with optional delta GAN.")

    parser.add_argument("--data-path", type=str, default="/mnt/data/shared/tomo_data/")
    parser.add_argument("--cache-dir", type=str, default="/mnt/data/shared/tomo_data/cache_sino")
    parser.add_argument("--output-dir", type=str, default="runs/delta_residual_gan_full")
    parser.add_argument("--generator-checkpoint", type=str, required=True)

    parser.add_argument("--reduction-ratio", type=int, default=8)
    parser.add_argument("--max-dose", type=float, default=70.0)
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--attention-kernel-size", type=int, default=15)
    parser.add_argument("--detector-width", type=int, default=64)
    parser.add_argument("--condition-reduction", choices=("mean", "max", "meanmax"), default="mean")

    parser.add_argument("--refiner-base-channels", type=int, default=32)
    parser.add_argument("--discriminator-base-channels", type=int, default=32)
    parser.add_argument("--delta-scale", type=float, default=0.15)

    parser.add_argument("--refiner-learning-rate", type=float, default=1e-4)
    parser.add_argument("--discriminator-learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=5)

    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--validation-size", type=float, default=0.1)

    parser.add_argument("--open-threshold", type=float, default=1e-3)
    parser.add_argument("--baseline-open-threshold", type=float, default=0.05)
    parser.add_argument("--open-weight", type=float, default=8.0)
    parser.add_argument("--closed-weight", type=float, default=0.5)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--refined-weight", type=float, default=1.0)
    parser.add_argument("--grad-weight", type=float, default=0.20)
    parser.add_argument("--closed-penalty-weight", type=float, default=0.20)
    parser.add_argument("--delta-reg-weight", type=float, default=0.005)
    parser.add_argument("--adv-weight", type=float, default=0.001)
    parser.add_argument("--disc-open-weight", type=float, default=4.0)
    parser.add_argument("--open-dilation", type=int, default=2)

    parser.add_argument("--selection-metric", choices=("open_l1", "mae", "loss"), default="open_l1")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--visualize-every", type=int, default=1)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)

    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")

    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_dir / "training.log", mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    root.addHandler(console)
    root.addHandler(file_handler)


def config_dict(args: argparse.Namespace, output_dir: Path, device: torch.device) -> dict[str, Any]:
    return {
        "data_path": str(Path(args.data_path).expanduser()),
        "cache_dir": str(Path(args.cache_dir).expanduser()),
        "output_dir": str(output_dir),
        "generator_checkpoint": str(Path(args.generator_checkpoint).expanduser()),
        "reduction_ratio": args.reduction_ratio,
        "max_dose": args.max_dose,
        "base_filters": args.base_filters,
        "attention_kernel_size": args.attention_kernel_size,
        "detector_width": args.detector_width,
        "condition_reduction": args.condition_reduction,
        "refiner_base_channels": args.refiner_base_channels,
        "discriminator_base_channels": args.discriminator_base_channels,
        "delta_scale": args.delta_scale,
        "refiner_learning_rate": args.refiner_learning_rate,
        "discriminator_learning_rate": args.discriminator_learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "train_size": args.train_size,
        "validation_size": args.validation_size,
        "use_cache": not args.no_cache,
        "seed": args.seed,
        "amp": bool(args.amp),
        "device": str(device),
        "open_threshold": args.open_threshold,
        "baseline_open_threshold": args.baseline_open_threshold,
        "open_weight": args.open_weight,
        "closed_weight": args.closed_weight,
        "residual_weight": args.residual_weight,
        "refined_weight": args.refined_weight,
        "grad_weight": args.grad_weight,
        "closed_penalty_weight": args.closed_penalty_weight,
        "delta_reg_weight": args.delta_reg_weight,
        "adv_weight": args.adv_weight,
        "disc_open_weight": args.disc_open_weight,
        "open_dilation": args.open_dilation,
        "selection_metric": args.selection_metric,
        "patch_version": PATCH_VERSION,
    }


def patient_level_split(dataset: RTDataset, train_size: float, val_size: float, seed: int) -> dict[str, Any]:
    patients = sorted({str(sample["patient_id"]) for sample in dataset.samples})
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_train = max(1, int(len(patients) * train_size))
    n_val = max(1, int(len(patients) * val_size))

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    return {
        "train_patients": sorted(train_patients),
        "val_patients": sorted(val_patients),
        "test_patients": sorted(test_patients),
        "train_indices": [i for i, s in enumerate(dataset.samples) if str(s["patient_id"]) in train_patients],
        "val_indices": [i for i, s in enumerate(dataset.samples) if str(s["patient_id"]) in val_patients],
        "test_indices": [i for i, s in enumerate(dataset.samples) if str(s["patient_id"]) in test_patients],
    }


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def load_generator_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> int | None:
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("generator_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state = checkpoint

    cleaned = {str(k).removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)

    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    logging.info("Loaded frozen generator checkpoint: %s", path)
    if epoch is not None:
        logging.info("Checkpoint epoch: %s", epoch)
    return int(epoch) if epoch is not None else None


def align_target(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 5 and target.shape[1] == 1 and target.shape[-1] == 1:
        return target[:, 0, :, :, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        return target[:, 0, :, :]
    if target.ndim == 3:
        return target
    raise ValueError(f"Unexpected target shape: {tuple(target.shape)}")


def align_logits(outputs: torch.Tensor) -> torch.Tensor:
    if outputs.ndim == 5 and outputs.shape[1] == 1 and outputs.shape[-1] == 1:
        return outputs[:, 0, :, :, 0]
    if outputs.ndim == 4 and outputs.shape[1] == 1:
        return outputs[:, 0, :, :]
    if outputs.ndim == 3:
        return outputs
    raise ValueError(f"Unexpected generator output shape: {tuple(outputs.shape)}")


def to_2d_channel(sino: torch.Tensor) -> torch.Tensor:
    if sino.ndim == 3:
        return sino.unsqueeze(1)
    if sino.ndim == 4 and sino.shape[1] == 1:
        return sino
    raise ValueError(f"Expected [B,N,64] or [B,1,N,64], got {tuple(sino.shape)}")


def make_condition_maps(inputs: torch.Tensor, reduction: str) -> torch.Tensor:
    if inputs.ndim != 5 or inputs.shape[1] < 2:
        raise ValueError(f"Expected [B,2,N,64,64], got {tuple(inputs.shape)}")

    x = inputs[:, :2]
    if reduction == "mean":
        return x.mean(dim=-1)
    if reduction == "max":
        return x.amax(dim=-1)
    if reduction == "meanmax":
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
    raise ValueError(f"Unknown condition reduction: {reduction}")


def make_open_focus_map(
    gt: torch.Tensor,
    baseline: torch.Tensor,
    open_threshold: float,
    baseline_open_threshold: float,
    dilation: int,
) -> torch.Tensor:
    open_map = ((gt > open_threshold) | (baseline > baseline_open_threshold)).float().unsqueeze(1)
    if dilation > 0:
        kernel = 2 * dilation + 1
        open_map = F.max_pool2d(open_map, kernel_size=kernel, stride=1, padding=dilation)
    return open_map.clamp(0.0, 1.0)


def compute_delta_target(gt: torch.Tensor, baseline: torch.Tensor, delta_scale: float) -> torch.Tensor:
    return torch.clamp(gt - baseline, min=-float(delta_scale), max=float(delta_scale))


def supervised_weights(open_focus_map: torch.Tensor, open_weight: float, closed_weight: float) -> torch.Tensor:
    return closed_weight + open_weight * open_focus_map[:, 0]


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    values = F.smooth_l1_loss(pred, target, reduction="none", beta=0.02)
    return torch.sum(weights * values) / torch.clamp(weights.sum(), min=1.0)


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(weights * torch.abs(pred - target)) / torch.clamp(weights.sum(), min=1.0)


def weighted_gradient_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    cp_pred = pred[:, 1:, :] - pred[:, :-1, :]
    cp_gt = target[:, 1:, :] - target[:, :-1, :]
    cp_w = 0.5 * (weights[:, 1:, :] + weights[:, :-1, :])

    leaf_pred = pred[:, :, 1:] - pred[:, :, :-1]
    leaf_gt = target[:, :, 1:] - target[:, :, :-1]
    leaf_w = 0.5 * (weights[:, :, 1:] + weights[:, :, :-1])

    cp_loss = torch.sum(cp_w * torch.abs(cp_pred - cp_gt)) / torch.clamp(cp_w.sum(), min=1.0)
    leaf_loss = torch.sum(leaf_w * torch.abs(leaf_pred - leaf_gt)) / torch.clamp(leaf_w.sum(), min=1.0)

    return cp_loss + leaf_loss


def weighted_lsgan_loss(
    logits: torch.Tensor,
    target_is_real: bool,
    open_focus_map: torch.Tensor,
    open_weight: float,
) -> torch.Tensor:
    target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
    patch_open = F.interpolate(open_focus_map, size=logits.shape[-2:], mode="nearest")
    weights = 1.0 + open_weight * patch_open
    return torch.sum(weights * (logits - target) ** 2) / torch.clamp(weights.sum(), min=1.0)


def residual_delta_loss(
    delta_pred: torch.Tensor,
    delta_target: torch.Tensor,
    refined: torch.Tensor,
    baseline: torch.Tensor,
    gt: torch.Tensor,
    open_focus_map: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = supervised_weights(open_focus_map, args.open_weight, args.closed_weight)

    delta_loss = weighted_smooth_l1(delta_pred, delta_target, weights)
    refined_l1 = weighted_l1(refined, gt, weights)
    grad = weighted_gradient_loss(refined, gt, weights)

    closed_mask = (gt <= args.open_threshold).float()
    closed_penalty = torch.sum(closed_mask * refined ** 2) / torch.clamp(closed_mask.sum(), min=1.0)

    delta_reg = torch.mean(torch.abs(delta_pred))
    baseline_keep = F.l1_loss(refined, baseline)

    total = (
        args.residual_weight * delta_loss
        + args.refined_weight * refined_l1
        + args.grad_weight * grad
        + args.closed_penalty_weight * closed_penalty
        + args.delta_reg_weight * delta_reg
        + 0.005 * baseline_keep
    )

    return total, {
        "delta_target_loss": float(delta_loss.detach().item()),
        "refined_l1": float(refined_l1.detach().item()),
        "grad_loss": float(grad.detach().item()),
        "closed_penalty": float(closed_penalty.detach().item()),
        "delta_reg": float(delta_reg.detach().item()),
        "baseline_keep": float(baseline_keep.detach().item()),
    }


def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    loss_value: float,
    baseline: torch.Tensor | None = None,
    delta_pred: torch.Tensor | None = None,
    delta_target: torch.Tensor | None = None,
) -> dict[str, float]:
    with torch.no_grad():
        diff = torch.abs(pred - gt)
        open_mask = gt > 1e-6
        closed_mask = ~open_mask

        open_l1 = diff[open_mask].mean() if open_mask.any() else torch.tensor(float("nan"), device=gt.device)
        closed_abs_pred = pred[closed_mask].mean() if closed_mask.any() else torch.tensor(float("nan"), device=gt.device)

        out = {
            "loss": float(loss_value),
            "mae": float(diff.mean().item()),
            "max_abs": float(diff.max().item()),
            "open_l1": float(open_l1.item()),
            "closed_abs_pred": float(closed_abs_pred.item()),
            "target_open_fraction": float(open_mask.float().mean().item()),
            "pred_mean": float(pred.mean().item()),
            "target_mean": float(gt.mean().item()),
            "pred_min": float(pred.min().item()),
            "pred_max": float(pred.max().item()),
        }

        if baseline is not None:
            base_diff = torch.abs(baseline - gt)
            base_open = base_diff[open_mask].mean() if open_mask.any() else torch.tensor(float("nan"), device=gt.device)
            base_closed = baseline[closed_mask].mean() if closed_mask.any() else torch.tensor(float("nan"), device=gt.device)
            out.update({
                "baseline_mae": float(base_diff.mean().item()),
                "baseline_open_l1": float(base_open.item()),
                "baseline_closed_abs_pred": float(base_closed.item()),
            })

        if delta_pred is not None:
            out["mean_abs_delta"] = float(torch.mean(torch.abs(delta_pred)).item())

        if delta_pred is not None and delta_target is not None:
            out["delta_mae"] = float(torch.mean(torch.abs(delta_pred - delta_target)).item())

        return out


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    out = {}
    for key in keys:
        values = [row[key] for row in rows if key in row and not np.isnan(row[key])]
        out[key] = float(np.mean(values)) if values else float("nan")
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


def batch_patient_id(batch: dict[str, Any]) -> str:
    value = batch.get("patient_id", "unknown")
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def batch_pareto_index(batch: dict[str, Any]) -> str:
    value = batch.get("pareto_index", "unknown")
    if isinstance(value, torch.Tensor):
        return str(int(value.item())) if value.numel() == 1 else "_".join(str(int(x)) for x in value.flatten().tolist())
    if isinstance(value, (list, tuple)):
        first = value[0]
        if isinstance(first, torch.Tensor) and first.numel() == 1:
            return str(int(first.item()))
        return str(first)
    return str(value)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return torch.amp.autocast(device_type="cpu", enabled=False)


def progress(loader: DataLoader, desc: str):
    if tqdm is None:
        return loader
    return tqdm(loader, desc=desc, leave=True)


def save_visualization(
    path: Path,
    batch: dict[str, Any],
    baseline: torch.Tensor,
    refined: torch.Tensor,
    delta_pred: torch.Tensor,
    delta_target: torch.Tensor,
    gt: torch.Tensor,
    epoch: int,
    metrics: dict[str, float],
    max_dose: float,
    delta_scale: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    b = baseline[0].detach().float().cpu().numpy()
    r = refined[0].detach().float().cpu().numpy()
    g = gt[0].detach().float().cpu().numpy()
    dp = delta_pred[0].detach().float().cpu().numpy()
    dt = delta_target[0].detach().float().cpu().numpy()

    inp = batch["input"][0].detach().float().cpu()
    mid = int(inp.shape[1] // 2)
    ct = inp[0, mid].numpy()
    dose = inp[1, mid].numpy() * float(max_dose)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    def show(ax, img, title, cmap, vmin=None, vmax=None):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    show(axes[0, 0], b, "Frozen baseline", "hot", 0, 1)
    show(axes[0, 1], r, f"Delta refined epoch {epoch}", "hot", 0, 1)
    show(axes[0, 2], g, f"GT open_l1={metrics.get('open_l1', float('nan')):.5f}", "hot", 0, 1)

    show(axes[1, 0], np.abs(b - g), "|baseline-GT|", "RdYlGn_r", 0, 0.5)
    show(axes[1, 1], np.abs(r - g), "|refined-GT|", "RdYlGn_r", 0, 0.5)
    show(axes[1, 2], dp, "Predicted delta", "coolwarm", -delta_scale, delta_scale)

    show(axes[2, 0], dt, "Target delta = GT-baseline", "coolwarm", -delta_scale, delta_scale)
    show(axes[2, 1], ct, "CT berlingo [ch0]", "gray", 0, 1)
    show(axes[2, 2], dose, "Dose berlingo [ch1]", "hot", 0, max_dose)

    plt.suptitle(
        f"Patient: {batch_patient_id(batch)} | Pareto: {batch_pareto_index(batch)} | Slice: {mid}/{inp.shape[1]}",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(
    path: Path,
    generator: nn.Module,
    refiner: nn.Module,
    discriminator: nn.Module,
    r_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    payload = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "refiner_state_dict": refiner.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "r_optimizer_state_dict": r_optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    if d_optimizer is not None:
        payload["d_optimizer_state_dict"] = d_optimizer.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def forward_baseline(generator: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.sigmoid(align_logits(generator(inputs)))


def train_epoch(
    generator: nn.Module,
    refiner: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    r_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    r_scaler,
    d_scaler,
    epoch: int,
) -> dict[str, float]:
    freeze_module(generator)
    refiner.train()
    discriminator.train()

    rows = []
    use_gan = args.adv_weight > 0.0

    for batch in progress(loader, f"Epoch {epoch} [delta residual train]"):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        gt = align_target(targets)

        with autocast_context(device, args.amp):
            condition = make_condition_maps(inputs, args.condition_reduction)
            baseline = forward_baseline(generator, inputs)
            baseline_2d = to_2d_channel(baseline)
            gt_2d = to_2d_channel(gt)
            delta_target = compute_delta_target(gt, baseline, args.delta_scale)
            delta_target_2d = to_2d_channel(delta_target)
            open_focus = make_open_focus_map(
                gt,
                baseline,
                args.open_threshold,
                args.baseline_open_threshold,
                args.open_dilation,
            )
            d_condition = torch.cat([condition, baseline_2d, open_focus], dim=1)

        d_loss_value = float("nan")
        d_real_value = float("nan")
        d_fake_value = float("nan")

        if use_gan:
            if d_optimizer is None:
                raise RuntimeError("adv_weight > 0 requires discriminator optimizer.")
            set_requires_grad(discriminator, True)
            d_optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, args.amp):
                with torch.no_grad():
                    fake_delta_2d = refiner(condition, baseline_2d)["delta"].detach()
                real_logits = discriminator(d_condition, delta_target_2d)
                fake_logits = discriminator(d_condition, fake_delta_2d)
                d_real = weighted_lsgan_loss(real_logits, True, open_focus, args.disc_open_weight)
                d_fake = weighted_lsgan_loss(fake_logits, False, open_focus, args.disc_open_weight)
                d_loss = 0.5 * (d_real + d_fake)

            if d_scaler is not None:
                d_scaler.scale(d_loss).backward()
                d_scaler.unscale_(d_optimizer)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                d_scaler.step(d_optimizer)
                d_scaler.update()
            else:
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                d_optimizer.step()

            d_loss_value = float(d_loss.detach().item())
            d_real_value = float(d_real.detach().item())
            d_fake_value = float(d_fake.detach().item())

        set_requires_grad(discriminator, False)
        r_optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, args.amp):
            refined_dict = refiner(condition, baseline_2d)
            delta_pred_2d = refined_dict["delta"]
            delta_pred = delta_pred_2d[:, 0]
            refined = refined_dict["refined"][:, 0]

            supervised, parts = residual_delta_loss(
                delta_pred,
                delta_target,
                refined,
                baseline,
                gt,
                open_focus,
                args,
            )

            if use_gan:
                adv_logits = discriminator(d_condition, delta_pred_2d)
                adv = weighted_lsgan_loss(adv_logits, True, open_focus, args.disc_open_weight)
            else:
                adv = torch.zeros((), device=device, dtype=supervised.dtype)

            total = supervised + args.adv_weight * adv

        if r_scaler is not None:
            r_scaler.scale(total).backward()
            r_scaler.unscale_(r_optimizer)
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
            r_scaler.step(r_optimizer)
            r_scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
            r_optimizer.step()

        metrics = compute_metrics(
            refined.detach(),
            gt.detach(),
            float(supervised.detach().item()),
            baseline.detach(),
            delta_pred.detach(),
            delta_target.detach(),
        )
        metrics.update(parts)
        metrics.update({
            "total_loss": float(total.detach().item()),
            "adv_loss": float(adv.detach().item()),
            "d_loss": d_loss_value,
            "d_real": d_real_value,
            "d_fake": d_fake_value,
        })
        rows.append(metrics)

        del inputs, targets, gt, condition, baseline, baseline_2d, gt_2d
        del delta_target, delta_target_2d, open_focus, d_condition, refined_dict
        del delta_pred_2d, delta_pred, refined, supervised, adv, total
        if device.type == "cuda":
            torch.cuda.empty_cache()

    set_requires_grad(discriminator, True)

    if not rows:
        raise RuntimeError("No training batches completed.")
    return mean_metrics(rows)


def evaluate(
    generator: nn.Module,
    refiner: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    split_name: str,
    vis_dir: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    freeze_module(generator)
    refiner.eval()

    rows = []
    per_sample = []
    saved_vis = False

    with torch.no_grad():
        for batch in progress(loader, f"Epoch {epoch} [{split_name}]"):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            with autocast_context(device, args.amp):
                condition = make_condition_maps(inputs, args.condition_reduction)
                baseline = forward_baseline(generator, inputs)
                baseline_2d = to_2d_channel(baseline)
                gt = align_target(targets)
                delta_target = compute_delta_target(gt, baseline, args.delta_scale)
                open_focus = make_open_focus_map(
                    gt,
                    baseline,
                    args.open_threshold,
                    args.baseline_open_threshold,
                    args.open_dilation,
                )

                refined_dict = refiner(condition, baseline_2d)
                delta_pred = refined_dict["delta"][:, 0]
                refined = refined_dict["refined"][:, 0]

                loss, parts = residual_delta_loss(
                    delta_pred,
                    delta_target,
                    refined,
                    baseline,
                    gt,
                    open_focus,
                    args,
                )

            metrics = compute_metrics(refined, gt, float(loss.item()), baseline, delta_pred, delta_target)
            metrics.update(parts)
            rows.append(metrics)
            per_sample.append({
                "patient_id": batch_patient_id(batch),
                "pareto_index": batch_pareto_index(batch),
                **metrics,
            })

            if vis_dir is not None and not saved_vis:
                save_visualization(
                    vis_dir / f"epoch_{epoch:04d}_{split_name}_patient_{batch_patient_id(batch)}_pareto_{batch_pareto_index(batch)}.png",
                    batch,
                    baseline,
                    refined,
                    delta_pred,
                    delta_target,
                    gt,
                    epoch,
                    metrics,
                    args.max_dose,
                    args.delta_scale,
                )
                saved_vis = True

            del inputs, targets, condition, baseline, baseline_2d, gt, delta_target
            del open_focus, refined_dict, delta_pred, refined, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError(f"No {split_name} batches completed.")
    return mean_metrics(rows), per_sample


def select_score(metrics: dict[str, float], name: str) -> float:
    if name not in metrics:
        raise KeyError(f"Missing selection metric {name}. Available: {sorted(metrics)}")
    return float(metrics[name])


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    run_name = (
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_bf{args.base_filters}_delta{args.delta_scale:g}_adv{args.adv_weight:g}"
    )
    output_dir = Path(args.output_dir).expanduser() / run_name
    setup_logging(output_dir)

    config = config_dict(args, output_dir, device)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    logging.info("========== SUPERVISED DELTA RESIDUAL + OPTIONAL DELTA GAN ==========")
    logging.info("Config:\n%s", json.dumps(config, indent=2))

    dataset = RTDataset(
        root_dir=args.data_path,
        augmentation=None,
        max_dose=args.max_dose,
        reduction_ratio=args.reduction_ratio,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )

    split = patient_level_split(dataset, args.train_size, args.validation_size, args.seed)
    split_manifest = {
        "train_patients": split["train_patients"],
        "val_patients": split["val_patients"],
        "test_patients": split["test_patients"],
        "n_train_samples": len(split["train_indices"]),
        "n_val_samples": len(split["val_indices"]),
        "n_test_samples": len(split["test_indices"]),
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    logging.info("Split:\n%s", json.dumps(split_manifest, indent=2))

    train_loader = DataLoader(
        Subset(dataset, split["train_indices"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(dataset, split["val_indices"]),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        Subset(dataset, split["test_indices"]),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    generator = DosePredictionAttentionInAgg(
        base_filters=args.base_filters,
        in_channel=2,
        attention_kernel_size=args.attention_kernel_size,
        detector_width=args.detector_width,
    ).to(device)
    load_generator_checkpoint(generator, args.generator_checkpoint, device)
    freeze_module(generator)

    condition_channels = 2 if args.condition_reduction in {"mean", "max"} else 4

    refiner = SinoResidualRefiner2D(
        condition_channels=condition_channels,
        base_channels=args.refiner_base_channels,
        delta_scale=args.delta_scale,
    ).to(device)

    discriminator = SinoPatchGANDiscriminator(
        in_channels=condition_channels + 3,
        base_channels=args.discriminator_base_channels,
    ).to(device)

    r_optimizer = torch.optim.Adam(refiner.parameters(), lr=args.refiner_learning_rate, betas=(0.5, 0.999))
    d_optimizer = (
        torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_learning_rate, betas=(0.5, 0.999))
        if args.adv_weight > 0.0
        else None
    )

    r_scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    d_scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" and d_optimizer else None

    baseline_metrics, baseline_per_sample = evaluate(
        generator,
        refiner,
        val_loader,
        device,
        args,
        epoch=0,
        split_name="val_baseline",
        vis_dir=output_dir / "visualizations",
    )
    write_csv(output_dir / "baseline_val_per_sample.csv", baseline_per_sample)

    best_score = select_score(baseline_metrics, args.selection_metric)
    no_improve = 0
    history = []

    save_checkpoint(
        output_dir / "best_delta_residual_checkpoint.pt",
        generator,
        refiner,
        discriminator,
        r_optimizer,
        d_optimizer,
        0,
        baseline_metrics,
        config,
    )
    write_csv(output_dir / "best_val_per_sample.csv", baseline_per_sample)

    logging.info("Baseline val metrics:\n%s", json.dumps(baseline_metrics, indent=2))
    logging.info("Initial best %s = %.6f", args.selection_metric, best_score)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            generator,
            refiner,
            discriminator,
            train_loader,
            r_optimizer,
            d_optimizer,
            device,
            args,
            r_scaler,
            d_scaler,
            epoch,
        )

        val_metrics, val_per_sample = evaluate(
            generator,
            refiner,
            val_loader,
            device,
            args,
            epoch,
            "val",
            output_dir / "visualizations" if epoch % args.visualize_every == 0 else None,
        )

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        write_csv(output_dir / "loss_history.csv", history)
        write_csv(output_dir / "val_per_sample_latest.csv", val_per_sample)

        score = select_score(val_metrics, args.selection_metric)

        logging.info(
            "Epoch %04d/%04d | train_loss=%.6f | val_mae=%.6f | val_open_l1=%.6f | "
            "val_closed_abs_pred=%.6f | baseline_open_l1=%.6f | delta_mae=%.6f | "
            "mean_abs_delta=%.6f | adv=%.6f | d=%.6f | score(%s)=%.6f",
            epoch,
            args.epochs,
            train_metrics.get("loss", float("nan")),
            val_metrics.get("mae", float("nan")),
            val_metrics.get("open_l1", float("nan")),
            val_metrics.get("closed_abs_pred", float("nan")),
            val_metrics.get("baseline_open_l1", float("nan")),
            val_metrics.get("delta_mae", float("nan")),
            val_metrics.get("mean_abs_delta", float("nan")),
            train_metrics.get("adv_loss", float("nan")),
            train_metrics.get("d_loss", float("nan")),
            args.selection_metric,
            score,
        )

        if score < best_score - args.min_delta:
            best_score = score
            no_improve = 0
            save_checkpoint(
                output_dir / "best_delta_residual_checkpoint.pt",
                generator,
                refiner,
                discriminator,
                r_optimizer,
                d_optimizer,
                epoch,
                val_metrics,
                config,
            )
            write_csv(output_dir / "best_val_per_sample.csv", val_per_sample)
        else:
            no_improve += 1
            logging.info("No validation improvement: %d/%d", no_improve, args.early_stop)

        if epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                generator,
                refiner,
                discriminator,
                r_optimizer,
                d_optimizer,
                epoch,
                val_metrics,
                config,
            )

        if no_improve >= args.early_stop:
            logging.info("Early stopping triggered.")
            break

    logging.info("Loading best checkpoint for test evaluation.")
    best_ckpt = torch.load(output_dir / "best_delta_residual_checkpoint.pt", map_location=device, weights_only=False)
    refiner.load_state_dict(best_ckpt["refiner_state_dict"])

    test_metrics, test_per_sample = evaluate(
        generator,
        refiner,
        test_loader,
        device,
        args,
        int(best_ckpt["epoch"]),
        "test",
        output_dir / "visualizations",
    )
    write_csv(output_dir / "test_per_sample.csv", test_per_sample)
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    logging.info("========== DONE ==========")
    logging.info("Best %s = %.6f at epoch %s", args.selection_metric, best_score, best_ckpt["epoch"])
    logging.info("Test metrics:\n%s", json.dumps(test_metrics, indent=2))
    logging.info("Outputs saved in: %s", output_dir)


if __name__ == "__main__":
    main()
