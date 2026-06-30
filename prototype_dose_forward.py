#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prototype_dose_forward.py

Minimal DIFFERENTIABLE simplified dose forward (raytracing++ / TERMA), as the
candidate dose-equivalent loss operator (lever B). Built on the berlingo, which
already gives the CT rotated to each gantry angle -> the ray axis W IS beam depth.

Per control point i, leaf row h, depth w:
    rho(CT)        : relative electron density from normalized HU
    mu             : mu_water_per_voxel * rho            (linear attenuation)
    depth_radio    : cumulative mu along W (radiological depth)
    TERMA[i,h,w]   = LOT[i,h] * mu * exp(-depth_radio)   (primary energy released)
This is the "beam's-eye-view primary dose": exactly differentiable, LINEAR in the
sinogram (so amplitude-faithful by construction -- the property the learned CNN
surrogate lacked).

Validates the two checks agreed before any training:
  (1) gradient is non-trivial AND amplitude-sensitive (0.5x LOT -> 0.5x dose EXACTLY,
      vs the surrogate's 0.66%);
  (2) Forward(GT) is physically sensible (open leaves -> attenuated beam deposit),
      qualitatively vs the real planned dose channel.

NOTE (honest scope): this is BEV primary only. Full degeneracy relief needs angular
ACCUMULATION (back-projecting all gantry angles into the patient frame) -- the next
step. BEV already makes the loss leaf-energy-weighted (dosimetric importance), but
the big null-space (many sinos -> same summed 3D dose) only appears after accumulation.
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.patient import RTDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def ct_to_rho(ct_norm: torch.Tensor) -> torch.Tensor:
    """Normalized CT in [0,1] (HU = ct*4095 - 1024) -> relative electron density.
    rho ~ (HU+1000)/1000: air(HU=-1000)->0, water(0)->1, bone(~1000)->2."""
    hu = ct_norm * 4095.0 - 1024.0
    return torch.clamp((hu + 1000.0) / 1000.0, min=0.0)


def _gaussian_blur(x: torch.Tensor, sigma_h: float, sigma_w: float) -> torch.Tensor:
    """Separable Gaussian blur on [N,H,W] (differentiable). Approximates lateral
    scatter (along H, the leaf axis) and depth buildup/spread (along W)."""
    def kern(sig):
        r = max(1, int(3 * sig))
        t = torch.arange(-r, r + 1, dtype=x.dtype)
        k = torch.exp(-(t ** 2) / (2 * sig * sig)); return (k / k.sum()), r
    y = x.unsqueeze(1)                                   # [N,1,H,W]
    if sigma_h > 0:
        kh, rh = kern(sigma_h)
        y = F.conv2d(y, kh.view(1, 1, -1, 1), padding=(rh, 0))
    if sigma_w > 0:
        kw, rw = kern(sigma_w)
        y = F.conv2d(y, kw.view(1, 1, 1, -1), padding=(0, rw))
    return y[:, 0]


def dose_forward(sino: torch.Tensor, ct: torch.Tensor, mu_water: float = 0.03,
                 entry: str = "w0", scatter_h: float = 0.0, scatter_w: float = 0.0) -> torch.Tensor:
    """sino [N_CP,64], ct [N_CP,H,W] (H=64 leaves) -> BEV dose [N_CP,H,W].
    Differentiable, linear in sino. scatter_h/scatter_w add a lateral/depth Gaussian
    spread approximating scatter (TERMA -> dose kernel convolution, the CCC 'lite')."""
    rho = ct_to_rho(ct)                      # [N,H,W]
    mu = mu_water * rho
    if entry == "wmax":                      # beam enters from the high-W side
        mu_path = torch.flip(mu, dims=[-1])
        depth = torch.cumsum(mu_path, dim=-1) - mu_path        # exclusive
        atten = torch.flip(torch.exp(-depth), dims=[-1])
    else:                                    # beam enters from w=0
        depth = torch.cumsum(mu, dim=-1) - mu
        atten = torch.exp(-depth)
    terma = sino.unsqueeze(-1) * mu * atten  # [N,H,1]*[N,H,W] -> [N,H,W]
    if scatter_h > 0 or scatter_w > 0:
        terma = _gaussian_blur(terma, scatter_h, scatter_w)
    return terma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/data/tomo_data/cache_sino_r8")
    ap.add_argument("--data-path", default="/mnt/data/tomo_data/")
    ap.add_argument("--patient", default="183040")
    ap.add_argument("--reduction-ratio", type=int, default=8)
    ap.add_argument("--mu-water", type=float, default=0.03)
    ap.add_argument("--out", default="/mnt/data/sinogram_generator/prototype_dose_forward.png")
    args = ap.parse_args()

    ds = RTDataset(args.data_path, augmentation=None, use_cache=True,
                   cache_dir=args.cache_dir, reduction_ratio=args.reduction_ratio,
                   debug=args.patient)
    s = ds[0]
    berlingo = s["input"]                    # [2, N_CP, H, W]
    sino = s["target"].float()               # [N_CP, 64]
    ct = berlingo[0].float()                 # [N_CP, H, W]
    real_dose = berlingo[1].float()          # [N_CP, H, W] (rotated planned dose)
    N, H, W = ct.shape
    logging.info("patient=%s  berlingo=%s  sino=%s  (H==leaves: %s)",
                 s["patient_id"], tuple(berlingo.shape), tuple(sino.shape), H == sino.shape[1])

    # ---- CHECK 1: amplitude sensitivity (vs surrogate's 0.66% for 0.5x) ----
    d1 = dose_forward(sino, ct, args.mu_water)
    base = d1.norm().item()
    print("\n=== CHECK 1: amplitude sensitivity ===")
    for a in (0.5, 0.8, 1.0, 1.5, 2.0):
        da = dose_forward(a * sino, ct, args.mu_water)
        rel = (da - d1).norm().item() / base
        print(f"  LOT x{a:<4}: dose rel-change {rel:6.3f}   (expected |a-1| = {abs(a-1):.3f})")
    print("  surrogate g for reference: 0.5x LOT -> only 0.66% change (amplitude-blind).")

    # ---- gradient: non-trivial, finite, dosimetrically weighted ----
    sp = sino.clone().requires_grad_(True)
    gt = dose_forward(sino, ct, args.mu_water).detach()
    L = ((dose_forward(sp, ct, args.mu_water) - gt) ** 2).mean()
    L.backward()
    g = sp.grad
    print("\n=== gradient health ===")
    print(f"  L(GT,GT)={L.item():.3e}  grad finite={bool(torch.isfinite(g).all())}  "
          f"grad nonzero frac={(g.abs()>0).float().mean().item():.3f}")
    # per-leaf dose weight = total deposited energy per unit LOT (dosimetric importance)
    weight = (dose_forward(torch.ones_like(sino), ct, args.mu_water)).sum(-1)  # [N,64]
    openm = sino > 1e-3
    print(f"  per-open-leaf dose weight: mean {weight[openm].mean():.3f}  "
          f"std {weight[openm].std():.3f}  (varies -> loss focuses dosimetrically)")

    # ---- CHECK 2: viz, simplified dose vs CT vs real planned dose (BEV) ----
    fwd = dose_forward(sino, ct, args.mu_water).detach()
    open_per_cp = (sino > 1e-3).float().sum(-1)
    cps = open_per_cp.argsort(descending=True)[:3].tolist()   # 3 most-open CPs
    # correlation (primary-only vs accumulated real dose, so expect partial)
    fm, rm = fwd.flatten(), real_dose.flatten()
    corr = torch.corrcoef(torch.stack([fm, rm]))[0, 1].item()
    print(f"\n=== CHECK 2: BEV shape (corr Forward(GT) vs real planned dose = {corr:.3f}) ===")
    print("  (per-CP primary vs accumulated real dose -> correlation is partial by design)")

    fig, axes = plt.subplots(len(cps), 4, figsize=(16, 4 * len(cps)))
    for r, i in enumerate(cps):
        for c, (img, title, cmap) in enumerate([
            (ct[i], f"CP {i}: CT (rho src)", "gray"),
            (fwd[i], "Forward(GT) simplified dose", "inferno"),
            (real_dose[i], "real planned dose (BEV)", "inferno"),
            (sino[i].unsqueeze(1).repeat(1, 8), "LOT (leaf)", "hot"),
        ]):
            ax = axes[r, c] if len(cps) > 1 else axes[c]
            im = ax.imshow(img.numpy(), cmap=cmap, aspect="auto")
            ax.set_title(title, fontsize=9); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"Differentiable simplified dose forward (BEV)  patient {s['patient_id']}", y=1.0)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
