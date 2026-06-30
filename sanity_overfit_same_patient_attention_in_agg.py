#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanity_overfit_same_patient_attention_in_agg.py

Controlled sanity test:
- one patient
- all paretos
- Attention + InstanceNorm3d + learned Conv3D aggregation
- SinogramLoss on raw logits

Run from project root, next to main.py.

Required:
    copy unet_attention_in_agg.py to models/unet_attention_in_agg.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models.unet_attention_in_agg import DosePredictionAttentionInAgg
from utils.patient import RTDataset

try:
    from utils.losses import SinogramLoss
except ImportError:
    class SinogramLoss(nn.Module):
        """Fallback: Charbonnier on sigmoid(logits) + weighted BCEWithLogits."""

        def __init__(self, eps: float = 1e-3, alpha: float = 0.5, pos_weight: float = 3.0):
            super().__init__()
            self.eps2 = eps ** 2
            self.alpha = alpha
            self.pos_weight = pos_weight

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            pred_sigmoid = torch.sigmoid(pred)
            charbonnier = torch.mean(torch.sqrt((pred_sigmoid - target) ** 2 + self.eps2))
            pos_weight = torch.tensor(self.pos_weight, device=pred.device, dtype=pred.dtype)
            weight = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
            bce = F.binary_cross_entropy_with_logits(pred, target, weight=weight)
            return charbonnier + self.alpha * bce


PATCH_VERSION = "attention_instance_norm_learned_agg_sanity_v1"


@dataclass(frozen=True)
class RunConfig:
    data_path: str
    cache_dir: str
    output_dir: str
    patient_id: str
    reduction_ratio: int
    max_dose: float
    base_filters: int
    attention_kernel_size: int
    detector_width: int
    learning_rate: float
    epochs: int
    log_every: int
    save_every: int
    use_cache: bool
    seed: int
    amp: bool
    device: str
    loss_name: str
    loss_alpha: float
    loss_pos_weight: float
    patch_version: str

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit all pareto plans for one patient using attention+InstanceNorm+learned aggregation."
    )
    parser.add_argument("--data-path", type=str, default="/mnt/data/shared/tomo_data/")
    parser.add_argument("--cache-dir", type=str, default="/mnt/data/shared/tomo_data/cache_sino")
    parser.add_argument("--output-dir", type=str, default="sanity_outputs/overfit_same_patient_attention_in_agg")
    parser.add_argument("--patient-id", type=str, default="324181")
    parser.add_argument("--reduction-ratio", type=int, default=8)
    parser.add_argument("--max-dose", type=float, default=70.0)
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--attention-kernel-size", type=int, default=15)
    parser.add_argument("--detector-width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--loss-alpha", type=float, default=0.5)
    parser.add_argument("--loss-pos-weight", type=float, default=3.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
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
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_dir / "same_patient_overfit.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def build_run_config(args: argparse.Namespace, output_dir: Path, device: torch.device) -> RunConfig:
    return RunConfig(
        data_path=str(Path(args.data_path).expanduser()),
        cache_dir=str(Path(args.cache_dir).expanduser()),
        output_dir=str(output_dir),
        patient_id=args.patient_id,
        reduction_ratio=args.reduction_ratio,
        max_dose=args.max_dose,
        base_filters=args.base_filters,
        attention_kernel_size=args.attention_kernel_size,
        detector_width=args.detector_width,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        log_every=args.log_every,
        save_every=args.save_every,
        use_cache=not args.no_cache,
        seed=args.seed,
        amp=bool(args.amp),
        device=str(device),
        loss_name="SinogramLoss",
        loss_alpha=args.loss_alpha,
        loss_pos_weight=args.loss_pos_weight,
        patch_version=PATCH_VERSION,
    )


def get_patient_indices(dataset: RTDataset, patient_id: str) -> list[int]:
    indices = [
        idx for idx, sample in enumerate(dataset.samples)
        if str(sample.get("patient_id")) == str(patient_id)
    ]

    if not indices:
        available = sorted({str(sample.get("patient_id")) for sample in dataset.samples})
        raise ValueError(f"Patient {patient_id!r} was not found. Available preview: {available[:20]}")

    indices.sort(key=lambda idx: int(dataset.samples[idx].get("pareto_index", idx)))
    return indices


def align_for_loss(outputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = outputs
    gt = targets

    if logits.ndim == 5 and logits.shape[1] == 1 and logits.shape[-1] == 1:
        logits = logits[:, 0, :, :, 0]
    elif logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0, :, :]
    elif logits.ndim != 3:
        raise ValueError(f"Unexpected output shape: {tuple(outputs.shape)}")

    if gt.ndim == 5 and gt.shape[1] == 1 and gt.shape[-1] == 1:
        gt = gt[:, 0, :, :, 0]
    elif gt.ndim == 4 and gt.shape[1] == 1:
        gt = gt[:, 0, :, :]
    elif gt.ndim != 3:
        raise ValueError(f"Unexpected target shape: {tuple(targets.shape)}")

    if logits.shape != gt.shape:
        raise ValueError(f"Prediction/target mismatch: logits={tuple(logits.shape)}, target={tuple(gt.shape)}")

    return logits, gt


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return torch.amp.autocast(device_type="cpu", enabled=False)


def detach_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def batch_patient_id(batch: dict[str, Any]) -> str:
    patient = batch.get("patient_id", "unknown")
    if isinstance(patient, (list, tuple)):
        return str(patient[0])
    return str(patient)


def batch_pareto_index(batch: dict[str, Any]) -> str:
    pareto = batch.get("pareto_index", "unknown")
    if isinstance(pareto, torch.Tensor):
        if pareto.numel() == 1:
            return str(int(pareto.item()))
        return "_".join(str(int(x)) for x in pareto.flatten().tolist())
    if isinstance(pareto, (list, tuple)):
        item = pareto[0]
        if isinstance(item, torch.Tensor) and item.numel() == 1:
            return str(int(item.item()))
        return str(item)
    return str(pareto)


def compute_metrics(logits: torch.Tensor, gt: torch.Tensor, loss_value: float) -> dict[str, float]:
    with torch.no_grad():
        pred = torch.sigmoid(logits)
        abs_diff = torch.abs(pred - gt)
        open_mask = gt > 1e-6
        closed_mask = ~open_mask

        open_l1 = abs_diff[open_mask].mean() if open_mask.any() else torch.tensor(float("nan"), device=gt.device)
        closed_abs_pred = pred[closed_mask].mean() if closed_mask.any() else torch.tensor(float("nan"), device=gt.device)

        return {
            "loss": float(loss_value),
            "mae": float(abs_diff.mean().item()),
            "max_abs": float(abs_diff.max().item()),
            "open_l1": float(open_l1.item()),
            "closed_abs_pred": float(closed_abs_pred.item()),
            "target_open_fraction": float(open_mask.float().mean().item()),
            "logit_min": float(logits.min().item()),
            "logit_max": float(logits.max().item()),
            "logit_mean": float(logits.mean().item()),
            "pred_min": float(pred.min().item()),
            "pred_max": float(pred.max().item()),
            "pred_mean": float(pred.mean().item()),
            "target_min": float(gt.min().item()),
            "target_max": float(gt.max().item()),
            "target_mean": float(gt.mean().item()),
        }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    out: dict[str, float] = {}
    for key in rows[0].keys():
        values = [row[key] for row in rows if not np.isnan(row[key])]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def save_visualization(
    output_path: Path,
    batch: dict[str, Any],
    logits: torch.Tensor,
    gt: torch.Tensor,
    epoch: int,
    loss_value: float,
    max_dose: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_np = detach_numpy(torch.sigmoid(logits[0]))
    gt_np = detach_numpy(gt[0])
    diff_np = np.abs(pred_np - gt_np)

    input_tensor = batch["input"][0].detach().float().cpu()
    n_cp = int(input_tensor.shape[1])
    mid_slice = n_cp // 2

    ct_slice = input_tensor[0, mid_slice].numpy()
    dose_slice = input_tensor[1, mid_slice].numpy() * float(max_dose)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def show(ax, img, title, cmap, vmin=None, vmax=None, label=""):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis("off")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if label:
            cb.set_label(label, fontsize=7)

    show(axes[0, 0], pred_np, f"Sigmoid prediction epoch {epoch}", "hot", 0, 1, "leaf open frac.")
    show(axes[0, 1], gt_np, f"Target loss={loss_value:.6f}", "hot", 0, 1, "leaf open frac.")
    show(axes[0, 2], diff_np, "Absolute difference", "RdYlGn_r", 0, 0.5, "|pred-target|")
    show(axes[1, 0], ct_slice, "CT berlingo [ch0]", "gray", 0, 1, "HU norm.")
    show(axes[1, 1], dose_slice, "Dose berlingo [ch1]", "hot", 0, max_dose, "Gy")
    axes[1, 2].axis("off")

    plt.suptitle(
        f"Patient: {batch_patient_id(batch)} | Pareto: {batch_pareto_index(batch)} | Slice: {mid_slice}/{n_cp}",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    metric_rows: list[dict[str, float]] = []
    per_sample: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            with autocast_context(device, amp):
                outputs = model(inputs)
                logits, gt = align_for_loss(outputs, targets)
                loss = loss_function(logits, gt)

            loss_value = float(loss.item())
            metrics = compute_metrics(logits, gt, loss_value)
            metric_rows.append(metrics)
            per_sample.append({
                "patient_id": batch_patient_id(batch),
                "pareto_index": batch_pareto_index(batch),
                **metrics,
            })

            del inputs, targets, outputs, logits, gt, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return mean_metrics(metric_rows), per_sample


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    eval_metrics: dict[str, float],
    config: RunConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "eval_metrics": eval_metrics,
            "config": asdict(config),
        },
        path,
    )


def save_final_visualizations(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    max_dose: float,
    amp: bool,
) -> None:
    model.eval()
    final_dir = output_dir / "visualizations" / "final_all_paretos"
    final_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            with autocast_context(device, amp):
                outputs = model(inputs)
                logits, gt = align_for_loss(outputs, targets)
                loss = loss_function(logits, gt)

            pareto = batch_pareto_index(batch)
            save_visualization(
                output_path=final_dir / f"pareto_{pareto}_epoch_{epoch:04d}.png",
                batch=batch,
                logits=logits,
                gt=gt,
                epoch=epoch,
                loss_value=float(loss.item()),
                max_dose=max_dose,
            )

            np.save(final_dir / f"pareto_{pareto}_pred_prob.npy", detach_numpy(torch.sigmoid(logits[0])))
            np.save(final_dir / f"pareto_{pareto}_target.npy", detach_numpy(gt[0]))

            del inputs, targets, outputs, logits, gt, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir).expanduser() / f"patient_{args.patient_id}"
    setup_logging(output_dir)
    set_seed(args.seed)

    run_config = build_run_config(args, output_dir, device)
    (output_dir / "config.json").write_text(json.dumps(asdict(run_config), indent=2), encoding="utf-8")

    logging.info("========== SAME-PATIENT OVERFIT: ATTENTION + IN + LEARNED AGG ==========")
    logging.info("Config:\n%s", json.dumps(asdict(run_config), indent=2))

    dataset = RTDataset(
        root_dir=args.data_path,
        augmentation=None,
        max_dose=args.max_dose,
        reduction_ratio=args.reduction_ratio,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )

    patient_indices = get_patient_indices(dataset, args.patient_id)
    patient_samples = [dataset.samples[idx] for idx in patient_indices]

    logging.info("Selected patient %s with %d pareto samples.", args.patient_id, len(patient_indices))
    for idx, sample in zip(patient_indices, patient_samples):
        logging.info(
            "  sample_idx=%s patient=%s pareto=%s plan=%s dose=%s",
            idx,
            sample.get("patient_id"),
            sample.get("pareto_index"),
            Path(str(sample.get("plan_file"))).name,
            Path(str(sample.get("dose_file"))).name,
        )

    selected_manifest = [
        {
            "dataset_index": idx,
            "patient_id": dataset.samples[idx].get("patient_id"),
            "pareto_index": dataset.samples[idx].get("pareto_index"),
            "plan_file": str(dataset.samples[idx].get("plan_file")),
            "dose_file": str(dataset.samples[idx].get("dose_file")),
        }
        for idx in patient_indices
    ]
    (output_dir / "selected_samples.json").write_text(json.dumps(selected_manifest, indent=2), encoding="utf-8")

    subset = Subset(dataset, patient_indices)

    train_loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = DosePredictionAttentionInAgg(
        base_filters=args.base_filters,
        in_channel=2,
        attention_kernel_size=args.attention_kernel_size,
        detector_width=args.detector_width,
    ).to(device)

    loss_function = SinogramLoss(alpha=args.loss_alpha, pos_weight=args.loss_pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    history_rows: list[dict[str, Any]] = []
    best_eval_loss = float("inf")

    logging.info("Starting overfit: %d epochs over %d paretos.", args.epochs, len(patient_indices))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_metric_rows: list[dict[str, float]] = []

        for batch in train_loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, args.amp):
                outputs = model(inputs)
                logits, gt = align_for_loss(outputs, targets)
                loss = loss_function(logits, gt)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"Invalid loss at epoch {epoch}, patient={batch_patient_id(batch)}, "
                    f"pareto={batch_pareto_index(batch)}"
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_metric_rows.append(compute_metrics(logits.detach(), gt.detach(), float(loss.item())))

            del inputs, targets, outputs, logits, gt, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

        train_metrics = mean_metrics(train_metric_rows)
        eval_metrics, per_sample_metrics = evaluate(
            model=model,
            loader=eval_loader,
            loss_function=loss_function,
            device=device,
            amp=args.amp,
        )

        row: dict[str, Any] = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"eval_{key}": value for key, value in eval_metrics.items()},
        }
        history_rows.append(row)
        write_csv(output_dir / "loss_history.csv", history_rows)

        if epoch % args.log_every == 0 or epoch == 1:
            logging.info(
                "Epoch %04d/%04d | train_loss=%.6f | eval_loss=%.6f | "
                "eval_mae=%.6f | eval_open_l1=%.6f | eval_closed_abs_pred=%.6f | eval_max_abs=%.6f",
                epoch,
                args.epochs,
                train_metrics.get("loss", float("nan")),
                eval_metrics.get("loss", float("nan")),
                eval_metrics.get("mae", float("nan")),
                eval_metrics.get("open_l1", float("nan")),
                eval_metrics.get("closed_abs_pred", float("nan")),
                eval_metrics.get("max_abs", float("nan")),
            )

        if eval_metrics.get("loss", float("inf")) < best_eval_loss:
            best_eval_loss = eval_metrics["loss"]
            save_checkpoint(output_dir / "best_checkpoint.pt", model, optimizer, epoch, eval_metrics, run_config)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                eval_metrics,
                run_config,
            )
            write_csv(output_dir / "per_sample_metrics_latest.csv", per_sample_metrics)

            first_batch = next(iter(eval_loader))
            inputs = first_batch["input"].to(device, non_blocking=True)
            targets = first_batch["target"].to(device, non_blocking=True)

            model.eval()
            with torch.no_grad():
                with autocast_context(device, args.amp):
                    outputs = model(inputs)
                    logits, gt = align_for_loss(outputs, targets)
                    loss = loss_function(logits, gt)

            save_visualization(
                output_path=output_dir / "visualizations" / f"epoch_{epoch:04d}_first_pareto.png",
                batch=first_batch,
                logits=logits,
                gt=gt,
                epoch=epoch,
                loss_value=float(loss.item()),
                max_dose=args.max_dose,
            )

            del inputs, targets, outputs, logits, gt, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

    final_metrics, final_per_sample_metrics = evaluate(
        model=model,
        loader=eval_loader,
        loss_function=loss_function,
        device=device,
        amp=args.amp,
    )
    write_csv(output_dir / "per_sample_metrics_final.csv", final_per_sample_metrics)
    save_checkpoint(output_dir / "final_checkpoint.pt", model, optimizer, args.epochs, final_metrics, run_config)
    save_final_visualizations(
        model=model,
        loader=eval_loader,
        loss_function=loss_function,
        device=device,
        output_dir=output_dir,
        epoch=args.epochs,
        max_dose=args.max_dose,
        amp=args.amp,
    )

    logging.info("========== DONE ==========")
    logging.info("Best eval loss: %.6f", best_eval_loss)
    logging.info("Final eval metrics:\n%s", json.dumps(final_metrics, indent=2))
    logging.info("Outputs saved in: %s", output_dir)


if __name__ == "__main__":
    main()
