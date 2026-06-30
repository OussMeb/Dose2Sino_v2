#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanity_overfit.py

Sanity check: can the model overfit a SINGLE sample?

If the architecture + loss + optimization are wired correctly, the model must be
able to drive the loss on one fixed (input, target) pair down to ~0 and the
sinogram metrics (open_l1, closed_abs_pred) toward 0. If it can't, the problem is
in the model/loss/data, not in the amount of data or the schedule.

Mirrors main_attention_in_agg_full.py:
- DosePredictionAttentionInAgg(base_filters=8, in_channel=2, attn_k=15,
  detector_width=64, n_leaves=64)
- SinogramLoss(alpha=0.5, pos_weight=3.0) on raw logits
- REDUCTION_RATIO=3

Examples:
    python sanity_overfit.py
    python sanity_overfit.py --patient <patient_id> --steps 2000 --lr 1e-3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch

from models.unet_attention_in_agg import DosePredictionAttentionInAgg
from models.sinogram_2p5d import DosePrediction2p5D
from utils.losses import SinogramLoss, SinogramL1RuleLoss, AngularSpectralLoss, DoseConsistencyLoss
from utils.dose_operator import read_geometry, find_plan
from utils.patient import RTDataset


def save_viz(out_dir: Path, step: int, logits: torch.Tensor, target: torch.Tensor,
             inp: torch.Tensor, loss_value: float, patient: str) -> Path:
    """Pred sinogram vs GT (+ abs diff) — same panel style as the trainer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = torch.sigmoid(logits[0]).detach().float().cpu().numpy()   # [N_CP, 64]
    gt = target[0].detach().float().cpu().numpy()                     # [N_CP, 64]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def show(ax, img, title, cmap, vmax):
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    show(axes[0], pred, f"sigmoid(pred) step {step}", "hot", 1)
    show(axes[1], gt, f"target  loss={loss_value:.5f}", "hot", 1)
    show(axes[2], np.abs(pred - gt), "abs diff", "RdYlGn_r", 0.5)
    plt.suptitle(f"sanity overfit  patient={patient}  open_frac={gt.mean():.3f}", y=1.02)
    plt.tight_layout()
    path = out_dir / f"sanity_step_{step:05d}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def align_for_loss(outputs: torch.Tensor, targets: torch.Tensor):
    """Same squeeze logic as TrainerSupervisedLogits._align_for_loss -> [B, N_CP, 64]."""
    pred = outputs
    if pred.ndim == 5 and pred.shape[1] == 1 and pred.shape[-1] == 1:
        pred = pred[:, 0, :, :, 0]
    elif pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0, :, :]
    gt = targets
    if gt.ndim == 4 and gt.shape[1] == 1 and gt.shape[-1] == 1:
        gt = gt[:, 0, :, :, 0]
    if pred.shape != gt.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(gt.shape)}")
    return pred, gt


@torch.no_grad()
def metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    pred = torch.sigmoid(logits)
    abs_diff = (pred - target).abs()
    open_mask = target > 1e-6
    closed_mask = ~open_mask
    return {
        "mae": abs_diff.mean().item(),
        "max_abs": abs_diff.max().item(),
        "open_l1": abs_diff[open_mask].mean().item() if open_mask.any() else float("nan"),
        "closed_pred": pred[closed_mask].mean().item() if closed_mask.any() else float("nan"),
        "pred_mean": pred.mean().item(),
        "target_mean": target.mean().item(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overfit a single sample (sanity check).")
    p.add_argument("--data-path", default="/mnt/data/tomo_data/")
    p.add_argument("--cache-dir", default="/mnt/data/tomo_data/cache_sino_r3")
    p.add_argument("--use-cache", action="store_true", default=True)
    p.add_argument("--no-cache", dest="use_cache", action="store_false")
    p.add_argument("--reduction-ratio", type=int, default=3)
    p.add_argument("--base-filters", type=int, default=8)
    p.add_argument("--arch", choices=["attn3d", "2p5d"], default="attn3d",
                   help="attn3d = 3D V-Net (default); 2p5d = slice-wise 2D rethink.")
    p.add_argument("--patient", default=None, help="Restrict to this patient id (else first sample).")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cosine", action="store_true", default=True,
                   help="Cosine-anneal the LR over the run (polishes fine details late).")
    p.add_argument("--no-cosine", dest="cosine", action="store_false")
    p.add_argument("--min-lr", type=float, default=1e-5,
                   help="Final LR for cosine annealing (eta_min).")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max grad norm (set high/0 to effectively disable).")
    p.add_argument("--pos-weight", type=float, default=8.0,
                   help="SinogramLoss positive (open-leaf) weight.")
    p.add_argument("--fp-weight", type=float, default=4.0,
                   help="SinogramLoss false-positive (over-paint) penalty.")
    p.add_argument("--row-weight", type=float, default=2.0,
                   help="L1-rule loss: closed-control-point (all-zero row) penalty.")
    p.add_argument("--loss", choices=["sinogram", "l1rule", "dose"], default="sinogram",
                   help="sinogram = Charbonnier+FP; l1rule = weighted L1 + rules; "
                        "dose = differentiable dose-consistency (lever B) + sinogram anchor.")
    p.add_argument("--dose-weight", type=float, default=1.0, help="dose loss: weight of L_dose.")
    p.add_argument("--sino-weight", type=float, default=0.1, help="dose loss: weight of the sinogram anchor.")
    p.add_argument("--amp-weight", type=float, default=1.0, help="dose loss: weight of the amplitude (MU) term.")
    p.add_argument("--dmax-weight", type=float, default=0.0,
                   help="dose loss: weight of the Dmax hinge penalty (one-sided, penalises pred>GT). "
                        "Uses real Tomo CCC kernel (corr=0.89). Start with 0.5–2.0.")
    p.add_argument("--n-z", type=int, default=48, help="dose loss: number of z-bins for accumulation.")
    p.add_argument("--reduce-h", action="store_true", default=True,
                   help="2p5d: downsample the leaf axis in the stem (ratio-3). Use --no-reduce-h for ratio-8.")
    p.add_argument("--no-reduce-h", dest="reduce_h", action="store_false")
    p.add_argument("--refine", action="store_true", default=False,
                   help="2p5d: add the refinement head on the [N_CP,64] output plane.")
    p.add_argument("--refine-mode", choices=["conv", "fno", "both"], default="conv",
                   help="2p5d refine head: conv (local CP+/-1), fno (global angular "
                        "Fourier op along N_CP), or both.")
    p.add_argument("--fno-modes", type=int, default=64,
                   help="FNO refine: number of low angular frequencies kept.")
    p.add_argument("--spectral-weight", type=float, default=0.0,
                   help="Weight of the auxiliary AngularSpectralLoss (0 = off).")
    p.add_argument("--spectral-modes", type=int, default=32,
                   help="AngularSpectralLoss: number of low angular freqs matched.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", default=True,
                   help="Mixed precision (matches training's USE_MIXED_PRECISION).")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--viz-dir", default="/mnt/data/sinogram_generator/sanity_viz",
                   help="Where to dump pred-vs-GT PNGs and the latest checkpoint.")
    p.add_argument("--viz-every", type=int, default=25,
                   help="Save a visualization + checkpoint every N steps.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device)

    dataset = RTDataset(
        args.data_path,
        augmentation=None,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
        reduction_ratio=args.reduction_ratio,
        debug=args.patient,  # None -> all patients; a patient_id -> only that one
    )
    if len(dataset) == 0:
        logging.error("No samples found (patient=%s under %s).", args.patient, args.data_path)
        return

    sample = dataset[0]
    inp = sample["input"].unsqueeze(0).to(device)          # [1, 2, D, H, W]
    target = sample["target"].unsqueeze(0).to(device)       # [1, N_CP, 64]
    logging.info("Overfitting patient=%s pareto=%s | input=%s target=%s",
                 sample["patient_id"], sample["pareto_index"],
                 tuple(inp.shape), tuple(target.shape))

    if args.arch == "2p5d":
        model = DosePrediction2p5D(
            base_filters=args.base_filters, in_channel=2, n_leaves=64,
            reduce_h=args.reduce_h, refine=args.refine,
            refine_mode=args.refine_mode, fno_modes=args.fno_modes,
        ).to(device)
    else:
        model = DosePredictionAttentionInAgg(
            base_filters=args.base_filters, in_channel=2,
            attention_kernel_size=15, detector_width=64, n_leaves=64,
        ).to(device)
    logging.info("Arch: %s | base_filters=%d | params=%d", args.arch, args.base_filters,
                 sum(p.numel() for p in model.parameters()))
    model.train()

    dose_geom = None
    if args.loss == "l1rule":
        loss_fn = SinogramL1RuleLoss(pos_weight=args.pos_weight,
                                     fp_weight=args.fp_weight, row_weight=args.row_weight)
    elif args.loss == "dose":
        loss_fn = DoseConsistencyLoss(n_z=args.n_z, dose_weight=args.dose_weight,
                                      sino_weight=args.sino_weight, amp_weight=args.amp_weight,
                                      dmax_weight=args.dmax_weight,
                                      pos_weight=args.pos_weight, fp_weight=args.fp_weight)
        plan = find_plan(args.data_path, sample["patient_id"], sample["pareto_index"])
        ang, tab = read_geometry(plan)
        N = inp.shape[2]
        dose_geom = {
            "ct": inp[0, 0],                                            # [N,H,W]
            "real": inp[0, 1],                                          # [N,H,W]
            "alpha": torch.tensor(90.0 - ang[:N], device=device),      # (90-gantry)
            "tables": torch.tensor(tab[:N], device=device),
        }
        logging.info("DoseConsistencyLoss n_z=%d dose_w=%.2f sino_w=%.2f (plan=%s)",
                     args.n_z, args.dose_weight, args.sino_weight, Path(plan).name)
    else:
        loss_fn = SinogramLoss(eps=1e-3, pos_weight=args.pos_weight, fp_weight=args.fp_weight)
    logging.info("SinogramLoss pos_weight=%.1f fp_weight=%.1f", args.pos_weight, args.fp_weight)
    spectral_fn = (AngularSpectralLoss(modes=args.spectral_modes, weight=args.spectral_weight)
                   if args.spectral_weight > 0 else None)
    if spectral_fn is not None:
        logging.info("AngularSpectralLoss weight=%.3g modes=%d", args.spectral_weight,
                     args.spectral_modes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.min_lr)
        if args.cosine else None
    )
    logging.info("Cosine LR: %s (%.1e -> %.1e)", args.cosine, args.lr, args.min_lr)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    logging.info("Mixed precision: %s", use_amp)

    first_loss = None
    best_loss = float("inf")
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, gt = align_for_loss(model(inp), target)
            if args.loss == "dose":
                result = loss_fn(
                    logits.squeeze(0), gt.squeeze(0), dose_geom["ct"],
                    dose_geom["real"], dose_geom["alpha"], dose_geom["tables"])
                loss, l_dose, l_sino, l_amp = result[:4]
                l_dmax = result[4] if len(result) > 4 else torch.zeros(1)
            else:
                loss = loss_fn(logits, gt)
                if spectral_fn is not None:
                    loss = loss + spectral_fn(logits, gt)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        lv = loss.item()
        first_loss = first_loss if first_loss is not None else lv
        best_loss = min(best_loss, lv)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            m = metrics(logits.detach(), gt)
            if args.loss == "dose":
                extra = (f" | L_dose {l_dose.item():.4f} L_sino {l_sino.item():.4f}"
                         f" L_amp {l_amp.item():.4f} L_dmax {l_dmax.item():.4f}")
            else:
                extra = ""
            logging.info(
                "step %4d | lr %.2e | loss %.6f | mae %.5f | open_l1 %.5f | closed_pred %.5f | "
                "pred_mean %.4f target_mean %.4f%s",
                step, optimizer.param_groups[0]["lr"], lv, m["mae"], m["open_l1"],
                m["closed_pred"], m["pred_mean"], m["target_mean"], extra,
            )

        if step == 1 or step % args.viz_every == 0 or step == args.steps:
            viz_dir = Path(args.viz_dir)
            path = save_viz(viz_dir, step, logits.detach(), gt, inp[0],
                            lv, str(sample["patient_id"]))
            torch.save({"step": step, "model_state_dict": model.state_dict(),
                        "loss": lv}, viz_dir / "latest.pt")
            logging.info("viz saved: %s", path)

    drop = (first_loss - best_loss) / first_loss * 100 if first_loss else 0.0
    logging.info("=== SANITY RESULT === first_loss=%.6f best_loss=%.6f (down %.1f%%)",
                 first_loss, best_loss, drop)
    if best_loss < 0.05:
        logging.info("PASS: model overfits one sample -> wiring/architecture are sound.")
    else:
        logging.warning("FAIL: loss did not collapse -> suspect model/loss/data, not data volume.")


if __name__ == "__main__":
    main()
