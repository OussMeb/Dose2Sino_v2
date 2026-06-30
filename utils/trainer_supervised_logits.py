#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/trainer_supervised_logits.py

Trainer for models that output raw logits for LOT sinograms.

Use with SinogramLoss:
    loss(logits, target)

For metrics/visualization:
    pred = sigmoid(logits)
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.trainer import Trainer


class TrainerSupervisedLogits(Trainer):
    """Supervised trainer for CT+dose berlingo -> LOT sinogram logits."""

    def _autocast_context(self):
        if self.config.USE_MIXED_PRECISION and self.device.type == "cuda":
            return torch.autocast(device_type="cuda")
        return nullcontext()

    def _align_for_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert model output and target to [B, N_CP, 64].

        outputs are raw logits.
        targets are normalized LOT values in [0, 1].
        """
        pred = outputs
        gt = targets

        if pred.ndim == 5 and pred.shape[1] == 1 and pred.shape[-1] == 1:
            pred = pred[:, 0, :, :, 0]
        elif pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred[:, 0, :, :]
        elif pred.ndim != 3:
            raise ValueError(f"Unexpected output shape for loss: {tuple(outputs.shape)}")

        if gt.ndim == 5 and gt.shape[1] == 1 and gt.shape[-1] == 1:
            gt = gt[:, 0, :, :, 0]
        elif gt.ndim == 4 and gt.shape[1] == 1:
            gt = gt[:, 0, :, :]
        elif gt.ndim != 3:
            raise ValueError(f"Unexpected target shape for loss: {tuple(targets.shape)}")

        if pred.shape != gt.shape:
            raise ValueError(
                f"Prediction/target shape mismatch after alignment: "
                f"pred={tuple(pred.shape)}, target={tuple(gt.shape)}"
            )

        return pred, gt

    def _probability_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_value: float,
    ) -> dict[str, float]:
        with torch.no_grad():
            pred = torch.sigmoid(logits)
            abs_diff = torch.abs(pred - targets)
            open_mask = targets > 1e-6
            closed_mask = ~open_mask

            open_l1 = (
                abs_diff[open_mask].mean()
                if open_mask.any()
                else torch.tensor(float("nan"), device=targets.device)
            )
            closed_abs_pred = (
                pred[closed_mask].mean()
                if closed_mask.any()
                else torch.tensor(float("nan"), device=targets.device)
            )

            return {
                "loss": float(loss_value),
                "mae": float(abs_diff.mean().item()),
                "max_abs": float(abs_diff.max().item()),
                "open_l1": float(open_l1.item()),
                "closed_abs_pred": float(closed_abs_pred.item()),
                "pred_mean": float(pred.mean().item()),
                "target_mean": float(targets.mean().item()),
            }

    def _dose_geometry(self, pid, pareto, n: int):
        """(alpha=90-gantry, tables) for the dose loss, cached per (patient,pareto).
        Returns (None, None) to fall back to the sinogram anchor only (excluded /
        low-fidelity acquisition, or geometry unavailable)."""
        if not hasattr(self, "_geom_cache"):
            self._geom_cache = {}
        pid = str(pid[0] if isinstance(pid, (list, tuple)) else pid)
        pareto = int(pareto[0] if isinstance(pareto, (list, tuple)) else pareto)
        # exact-match exclusion: low-fidelity acquisitions are specific (e.g. "187591"
        # FB is bad but "187591_DIBH" is fine), so do NOT strip the _DIBH suffix.
        exclude = getattr(self.config, "DOSE_EXCLUDE", set()) or set()
        if pid in exclude:
            return None, None
        key = (pid, pareto)
        if key not in self._geom_cache:
            try:
                from utils.dose_operator import find_plan, read_geometry
                ang, tab = read_geometry(find_plan(self.config.DATA_PATH, pid, pareto))
                self._geom_cache[key] = (90.0 - ang, tab)
            except Exception as e:
                logging.warning("dose-loss geometry unavailable for %s/%s: %s -> sino-only",
                                pid, pareto, e)
                self._geom_cache[key] = (None, None)
        a, t = self._geom_cache[key]
        if a is None:
            return None, None
        return (torch.as_tensor(a[:n], device=self.device, dtype=torch.float32),
                torch.as_tensor(t[:n], device=self.device, dtype=torch.float32))

    def _compute_loss(self, logits, gt, inputs, batch):
        """Standard path: loss_function(logits, gt). Dose path (DoseConsistencyLoss):
        feed CT/real-dose channels + per-CP geometry. Returns (loss, dose_components)."""
        from utils.losses import DoseConsistencyLoss
        if not isinstance(self.loss_function, DoseConsistencyLoss):
            return self.loss_function(logits, gt), None
        ct, real = inputs[0, 0], inputs[0, 1]                      # [N,H,W] each (B=1)
        alpha, tables = self._dose_geometry(batch.get("patient_id"),
                                            batch.get("pareto_index", 0), ct.shape[0])
        if alpha is None:                                          # fall back to anchor only
            return self.loss_function.sino(logits.squeeze(0), gt.squeeze(0)), None
        result = self.loss_function(logits.squeeze(0), gt.squeeze(0), ct, real, alpha, tables)
        total, l_dose, l_sino, l_amp = result[:4]
        l_dmax = result[4] if len(result) > 4 else torch.zeros(1)
        return total, (float(l_dose), float(l_sino), float(l_amp), float(l_dmax))

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        valid_batches = 0
        generation_times = []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.EPOCHS} [train]",
            leave=True,
        )

        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch["input"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)

            start_time = datetime.now()
            self.optimizer.zero_grad(set_to_none=True)

            try:
                with self._autocast_context():
                    outputs = self.model(inputs)
                    logits, gt = self._align_for_loss(outputs, targets)
                    loss, dose_comps = self._compute_loss(logits, gt, inputs, batch)
            except Exception:
                logging.exception(
                    "Training batch failed for patient=%s",
                    batch.get("patient_id", "unknown"),
                )
                raise

            generation_times.append((datetime.now() - start_time).total_seconds())

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("Invalid loss detected for batch %s", batch.get("patient_id", "unknown"))
                continue

            if self.scaler is not None and self.device.type == "cuda":
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            loss_value = float(loss.item())
            metrics = self._probability_metrics(logits.detach(), gt.detach(), loss_value)
            epoch_loss += loss_value
            valid_batches += 1

            postfix = {
                "loss": f"{loss_value:.6f}",
                "mae": f"{metrics['mae']:.6f}",
                "open": f"{metrics['open_l1']:.6f}",
                "closed": f"{metrics['closed_abs_pred']:.6f}",
                "avg": f"{epoch_loss / valid_batches:.6f}",
            }
            if dose_comps is not None:
                postfix["Ldose"] = f"{dose_comps[0]:.4f}"
                postfix["Lamp"] = f"{dose_comps[2]:.4f}"
                if len(dose_comps) > 3 and dose_comps[3] > 0:
                    postfix["Ldmax"] = f"{dose_comps[3]:.4f}"
            progress_bar.set_postfix(postfix)

            if batch_idx % 10 == 0:
                logging.info(
                    "Epoch %d/%d, Batch %d/%d, Loss %.6f, MAE %.6f, "
                    "open_l1 %.6f, closed_abs_pred %.6f, Avg %.6f",
                    epoch + 1,
                    self.config.EPOCHS,
                    batch_idx,
                    len(self.train_loader),
                    loss_value,
                    metrics["mae"],
                    metrics["open_l1"],
                    metrics["closed_abs_pred"],
                    epoch_loss / valid_batches,
                )

            del inputs, targets, outputs, logits, gt, loss
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if generation_times:
            avg_generation_time = sum(generation_times) / len(generation_times)
            logging.info("Average generation time per batch: %.3fs", avg_generation_time)

        if valid_batches == 0:
            raise RuntimeError("No valid training batches completed in this epoch.")

        return epoch_loss / valid_batches

    def validate(self, epoch: int) -> float:
        self.model.eval()
        val_loss = 0.0
        valid_batches = 0
        dose_sum = 0.0
        dose_batches = 0

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1}/{self.config.EPOCHS} [val]",
            leave=True,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                inputs = batch["input"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)

                with self._autocast_context():
                    outputs = self.model(inputs)
                    logits, gt = self._align_for_loss(outputs, targets)
                    loss, dose_comps = self._compute_loss(logits, gt, inputs, batch)
                if dose_comps is not None:
                    dose_sum += dose_comps[0]
                    dose_batches += 1

                loss_value = float(loss.item())
                metrics = self._probability_metrics(logits, gt, loss_value)

                val_loss += loss_value
                valid_batches += 1

                progress_bar.set_postfix({
                    "val_loss": f"{loss_value:.6f}",
                    "mae": f"{metrics['mae']:.6f}",
                    "open": f"{metrics['open_l1']:.6f}",
                    "closed": f"{metrics['closed_abs_pred']:.6f}",
                    "avg": f"{val_loss / valid_batches:.6f}",
                })

                logging.info(
                    "Validation loss %.6f, MAE %.6f, open_l1 %.6f, closed_abs_pred %.6f for batch %s",
                    loss_value,
                    metrics["mae"],
                    metrics["open_l1"],
                    metrics["closed_abs_pred"],
                    batch.get("patient_id", "unknown"),
                )

                if batch_idx == 0:
                    outputs_cpu = outputs.float().cpu()
                    targets_cpu = targets.float().cpu()
                    first_batch_loss = loss_value
                    first_batch = batch
                    del inputs, targets, outputs, logits, gt, loss
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    self._save_visualization_logits(outputs_cpu, targets_cpu, first_batch, epoch, first_batch_loss)
                else:
                    del inputs, targets, outputs, logits, gt, loss
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

        if valid_batches == 0:
            raise RuntimeError("No validation batches available.")

        if dose_batches > 0:
            # dose-space val metric (mean L_dose) -- the meaningful signal for the
            # dose loss; open_l1 is saturated at the ~0.15 degeneracy floor.
            logging.info("VAL dose metric (mean L_dose) %.5f over %d/%d batches",
                         dose_sum / dose_batches, dose_batches, valid_batches)

        return val_loss / valid_batches

    def _save_visualization_logits(self, outputs, targets, batch, epoch, loss_value):
        """Save visualization using sigmoid(logits)."""
        vis_dir = self.checkpoint_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        patient_id = batch.get("patient_id", ["unknown"])
        pareto_index = batch.get("pareto_index", ["N/A"])

        logits = outputs[0].squeeze().detach().float()
        if logits.ndim == 3:
            logits = logits[:, :, 0]
        pred_sino = torch.sigmoid(logits).cpu().numpy()
        target_sino = targets[0].squeeze().detach().float().cpu().numpy()

        # CT/dose berlingo are [N_CP, H, W]. Display them ALONG the control-point
        # axis (lengthwise, like the sinogram) via a MAX projection over the ray
        # axis (W) -> [N_CP, H]. This mirrors the model's ray reduction and is
        # spatially aligned with the [N_CP, 64] sinogram above.
        inp = batch["input"][0]
        n_cp = inp.shape[1]
        ct_proj = inp[0].amax(dim=-1).detach().cpu().numpy()                          # [N_CP, H]
        dose_proj = inp[1].amax(dim=-1).detach().cpu().numpy() * self.config.MAX_DOSE  # [N_CP, H]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        def show(ax, img, title, cmap, vmin=None, vmax=None, label=""):
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(title, fontsize=9, pad=4)
            ax.axis("off")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if label:
                cb.set_label(label, fontsize=7)

        show(axes[0, 0], pred_sino, f"Sigmoid prediction epoch {epoch + 1}", "hot", 0, 1, "leaf open frac.")
        show(axes[0, 1], target_sino, f"Target loss={loss_value:.5f}", "hot", 0, 1, "leaf open frac.")
        show(axes[0, 2], np.abs(pred_sino - target_sino), "Absolute difference", "RdYlGn_r", 0, 0.5)
        show(axes[1, 0], ct_proj, "CT berlingo [ch0] (ray max-proj) [N_CP x H]", "gray", 0, 1, "HU norm.")
        show(axes[1, 1], dose_proj, "Dose berlingo [ch1] (ray max-proj) [N_CP x H]", "hot", 0, self.config.MAX_DOSE, "Gy")
        axes[1, 2].axis("off")

        p = pareto_index[0] if isinstance(pareto_index, (list, tuple)) else pareto_index
        safe_pareto = str(p.item() if hasattr(p, "item") else p).replace("/", "_").replace("\\", "_")

        patient = patient_id[0] if isinstance(patient_id, (list, tuple)) else patient_id

        plt.suptitle(
            f"Patient: {patient}   Pareto: {safe_pareto}   N_CP: {n_cp}",
            fontsize=12,
            y=1.01,
        )
        plt.tight_layout()

        save_path = vis_dir / f"{self.session_prefix}_epoch_{epoch + 1}_patient_{patient}_pareto_{safe_pareto}.png"
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        logging.info("Visualization saved: %s", save_path)

    def test(self, model_path: Optional[str] = None, verbose: bool = False) -> float:
        """Test using L1 on sigmoid(logits), not raw logits."""
        if model_path:
            checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()
        test_loss = 0.0
        valid_batches = 0

        progress_bar = tqdm(self.test_loader, desc="[test]", leave=True)

        with torch.no_grad():
            for batch in progress_bar:
                inputs = batch["input"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                logits, gt = self._align_for_loss(outputs, targets)
                pred = torch.sigmoid(logits)
                loss = torch.nn.functional.l1_loss(pred, gt)

                loss_value = float(loss.item())
                test_loss += loss_value
                valid_batches += 1

                progress_bar.set_postfix({
                    "sigmoid_l1": f"{loss_value:.6f}",
                    "avg": f"{test_loss / valid_batches:.6f}",
                })

                if verbose:
                    logging.info("Test sigmoid L1 %.6f for batch %s", loss_value, batch.get("patient_id", "unknown"))

                del inputs, targets, outputs, logits, gt, pred, loss
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        if valid_batches == 0:
            raise RuntimeError("No test batches available.")

        avg_test_loss = test_loss / valid_batches
        logging.info("Test sigmoid L1 Loss: %.6f", avg_test_loss)
        return avg_test_loss

    def train(self):
        logging.info("Starting training for %d epochs", self.config.EPOCHS)
        previous_lr = self.optimizer.param_groups[0]["lr"]

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            # ReduceLROnPlateau steps on the val metric; CosineAnnealingLR (and
            # other epoch schedulers) step with no argument.
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            lr_changed = abs(current_lr - previous_lr) > 1e-8

            if lr_changed:
                logging.info("Learning rate changed from %.6f to %.6f", previous_lr, current_lr)
                logging.info("Resetting early stop counter due to LR change")
                self.early_stop_counter = 0
                previous_lr = current_lr

            logging.info(
                "Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f, LR: %.6f",
                epoch + 1,
                self.config.EPOCHS,
                train_loss,
                val_loss,
                current_lr,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                self.save_checkpoint(epoch, train_loss, val_loss)
            else:
                self.early_stop_counter += 1
                logging.info(
                    "No improvement. Early stop counter: %d/%d",
                    self.early_stop_counter,
                    self.config.AUTO_STOP,
                )
                logging.info("Best validation loss so far: %.6f", self.best_val_loss)

                if self.early_stop_counter >= self.config.AUTO_STOP:
                    logging.info("Early stopping triggered.")
                    break
