from datetime import datetime
import logging
from contextlib import nullcontext
from typing import Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch
from tqdm.auto import tqdm

from utils.trainer import Trainer

DEBUG_VIEWER_DIR = "/tmp/debug_viewer"


class TrainerSupervised(Trainer):
    """Supervised trainer for CT+dose berlingo -> LOT sinogram prediction."""

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

        Expected:
            outputs: [B, 1, N_CP, 64, 1] from current VNet
            targets: [B, N_CP, 64] from RTDataset
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

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch with tqdm progress and CUDA-safe mixed precision."""
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
                    pred, gt = self._align_for_loss(outputs, targets)
                    loss = self.loss_function(pred, gt)
            except Exception:
                logging.exception(f"Training batch failed for patient={batch.get('patient_id', 'unknown')}")
                raise

            generation_times.append((datetime.now() - start_time).total_seconds())

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Invalid loss detected for batch {batch.get('patient_id', 'unknown')}")
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
            epoch_loss += loss_value
            valid_batches += 1

            progress_bar.set_postfix({
                "loss": f"{loss_value:.6f}",
                "avg": f"{epoch_loss / valid_batches:.6f}",
            })

            if batch_idx % 10 == 0:
                logging.info(
                    f"Epoch {epoch + 1}/{self.config.EPOCHS}, "
                    f"Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: current batch {loss_value:.6f}, "
                    f"Average: {epoch_loss / valid_batches:.6f}"
                )

        if generation_times:
            avg_generation_time = sum(generation_times) / len(generation_times)
            logging.info(f"Average generation time per batch: {avg_generation_time:.3f}s")

        if valid_batches == 0:
            raise RuntimeError("No valid training batches completed in this epoch.")

        return epoch_loss / valid_batches

    def validate(self, epoch: int) -> float:
        """Validate with tqdm progress and CUDA-safe mixed precision."""
        self.model.eval()
        val_loss = 0.0
        valid_batches = 0

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
                    pred, gt = self._align_for_loss(outputs, targets)
                    loss = self.loss_function(pred, gt)

                loss_value = float(loss.item())
                val_loss += loss_value
                valid_batches += 1

                progress_bar.set_postfix({
                    "val_loss": f"{loss_value:.6f}",
                    "avg": f"{val_loss / valid_batches:.6f}",
                })

                logging.info(
                    f"Validation loss: {loss_value:.6f} "
                    f"for batch {batch.get('patient_id', 'unknown')}"
                )

                if batch_idx == 0:
                    outputs_cpu = outputs.float().cpu()
                    targets_cpu = targets.float().cpu()
                    first_batch_loss = loss_value
                    del inputs, targets, outputs, pred, gt, loss
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    self._save_visualization(outputs_cpu, targets_cpu, batch, epoch, first_batch_loss)
                else:
                    del inputs, targets, outputs, pred, gt, loss
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

        if valid_batches == 0:
            raise RuntimeError("No validation batches available.")

        return val_loss / valid_batches

    def test(self, model_path: Optional[str] = None, verbose: bool = False) -> float:
        """Test using L1 loss on normalized LOT sinograms."""
        if model_path:
            checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()
        test_loss = 0.0
        valid_batches = 0

        progress_bar = tqdm(
            self.test_loader,
            desc="[test]",
            leave=True,
        )

        with torch.no_grad():
            for batch in progress_bar:
                inputs = batch["input"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                pred, gt = self._align_for_loss(outputs, targets)
                loss = torch.nn.functional.l1_loss(pred, gt)

                loss_value = float(loss.item())
                test_loss += loss_value
                valid_batches += 1

                progress_bar.set_postfix({
                    "l1": f"{loss_value:.6f}",
                    "avg": f"{test_loss / valid_batches:.6f}",
                })

                if verbose:
                    logging.info(
                        "Test loss: {:.6f} for batch {}".format(
                            loss_value,
                            batch.get("patient_id", "unknown"),
                        )
                    )

                del inputs, targets, outputs, pred, gt, loss
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        if valid_batches == 0:
            raise RuntimeError("No test batches available.")

        avg_test_loss = test_loss / valid_batches
        logging.info(f"Test Loss: {avg_test_loss:.6f}")
        return avg_test_loss

    def train(self):
        """Main training loop."""
        logging.info(f"Starting training for {self.config.EPOCHS} epochs")

        previous_lr = self.optimizer.param_groups[0]["lr"]

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            lr_changed = abs(current_lr - previous_lr) > 1e-8

            if lr_changed:
                logging.info(f"Learning rate changed from {previous_lr:.6f} to {current_lr:.6f}")
                logging.info("Resetting early stop counter due to LR change")
                self.early_stop_counter = 0
                previous_lr = current_lr

            logging.info(
                f"Epoch {epoch + 1}/{self.config.EPOCHS} - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.6f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                self.save_checkpoint(epoch, train_loss, val_loss)
            else:
                self.early_stop_counter += 1
                logging.info(
                    f"No improvement. Early stop counter: "
                    f"{self.early_stop_counter}/{self.config.AUTO_STOP}"
                )
                logging.info(f"Best validation loss so far: {self.best_val_loss:.6f}")

                if self.early_stop_counter >= self.config.AUTO_STOP:
                    logging.info("Early stopping triggered")
                    break

        logging.info("Training completed")
        checkpoints = sorted(self.checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            raise RuntimeError("Training finished without saving any checkpoint.")
        return checkpoints[-1]
