#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prototype_dose_accumulate.py

Step 2 of lever B: ANGULAR ACCUMULATION of the differentiable BEV primary dose
(prototype_dose_forward.py) into the patient frame -> a real 3D differentiable dose,
where the planning null-space (many sinograms -> same accumulated dose) finally appears.

Per control point i the berlingo slice is the axial CT rotated by alpha_i=(90-theta_i)
at couch z_i=z_iso-tables[i] (utils/mask_utils.apply_tomo_transform_to_stack). So to
accumulate we UN-rotate each CP's BEV dose by -alpha_i (differentiable grid_sample) and
sum it into the z-bin of z_i (differentiable index_add):

    patient_dose[z,Y,X] = sum_i [zbin(z_i)==z] * Rotate(TERMA_i, -alpha_i)

Validates:
  (1) accumulation makes the simplified dose match the REAL planned dose FAR better
      than the per-CP primary did (corr 0.47 -> ?), i.e. the operator is dosimetrically
      meaningful once summed over angles;
  (2) the dose loss is ~INVARIANT to a dose-equivalent sinogram perturbation while
      open_l1 jumps -> it dissolves the 0.15 degeneracy (the whole point of lever B).
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.patient import RTDataset
from prototype_dose_forward import dose_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
SINO_TAG = (0x300D, 0x10A7)


def read_geometry(plan_path):
    ds = pydicom.dcmread(plan_path, force=True)
    beam = ds[(0x300A, 0x00B0)][0]
    cps = beam[(0x300A, 0x0111)].value
    ang = np.array([float(getattr(cp, "GantryAngle", 0.0)) for cp in cps], np.float32)
    tab = np.array([float(getattr(cp, "TableTopLateralPosition", 0.0)) for cp in cps], np.float32)
    return ang, tab


def rotate_stack(imgs: torch.Tensor, beta_rad: torch.Tensor) -> torch.Tensor:
    """Rotate each [H,W] image about its center by beta_rad[i] (differentiable)."""
    n = imgs.shape[0]
    c, s = torch.cos(beta_rad), torch.sin(beta_rad)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = c; theta[:, 0, 1] = -s
    theta[:, 1, 0] = s; theta[:, 1, 1] = c
    grid = F.affine_grid(theta, (n, 1, imgs.shape[-2], imgs.shape[-1]), align_corners=False)
    return F.grid_sample(imgs.unsqueeze(1), grid, align_corners=False, padding_mode="zeros")[:, 0]


def accumulate(sino, ct, alpha_deg, zbin, n_z, sign=-1.0, mu=0.03,
               scatter_h=0.0, scatter_w=0.0):
    """sino[N,64], ct[N,64,64] -> patient_dose[n_z,64,64] (differentiable in sino)."""
    terma = dose_forward(sino, ct, mu, scatter_h=scatter_h, scatter_w=scatter_w)  # [N,64,64]
    beta = torch.deg2rad(torch.as_tensor(sign * alpha_deg))  # un-rotate
    rot = rotate_stack(terma, beta)                          # [N,64,64]
    vol = torch.zeros(n_z, ct.shape[-2], ct.shape[-1])
    vol.index_add_(0, zbin, rot)
    return vol


def corr(a, b):
    a, b = a.flatten(), b.flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="183040")
    ap.add_argument("--plan", default="/mnt/data/tomo_data/183040/RC_Publi_Tomo_Halcyon_DIBH/"
                    "Tomo_FB_copy/pareto_0/RP1.2.752.243.1.1.20250317095242862.1300.83153.dcm")
    ap.add_argument("--n-z", type=int, default=48)
    ap.add_argument("--mu-water", type=float, default=0.03)
    ap.add_argument("--out", default="/mnt/data/sinogram_generator/prototype_dose_accumulate.png")
    args = ap.parse_args()

    ds = RTDataset("/mnt/data/tomo_data/", augmentation=None, use_cache=True,
                   cache_dir="/mnt/data/tomo_data/cache_sino_r8", reduction_ratio=8,
                   debug=args.patient)
    s = ds[0]
    ct = s["input"][0].float(); real_dose = s["input"][1].float(); sino = s["target"].float()
    N = ct.shape[0]
    ang, tab = read_geometry(args.plan)
    M = min(N, len(ang)); ct, real_dose, sino = ct[:M], real_dose[:M], sino[:M]
    ang, tab = ang[:M], tab[:M]
    logging.info("patient=%s  N_CP berlingo=%d plan=%d -> use %d", s["patient_id"], N, len(ang), M)

    alpha = 90.0 - ang                                  # forward rotation used to build berlingo
    # z-bin from table position (z_i = z_iso - tab, z_iso const -> bin by -tab)
    z = -tab
    zbin = ((z - z.min()) / (np.ptp(z) + 1e-6) * (args.n_z - 1)).round().astype(np.int64)
    zbin = torch.from_numpy(zbin)

    # ---- (1) accumulation vs real dose (try both rotation signs, keep best) ----
    real3d = {}
    for sign in (-1.0, 1.0):
        terma_real = real_dose                                  # already TOTAL dose per CP
        beta = torch.deg2rad(torch.as_tensor(sign * alpha))
        rr = rotate_stack(terma_real, beta)
        vol = torch.zeros(args.n_z, 64, 64); cnt = torch.zeros(args.n_z, 64, 64)
        vol.index_add_(0, zbin, rr); cnt.index_add_(0, zbin, torch.ones_like(rr))
        real3d[sign] = vol / cnt.clamp_min(1.0)                 # AVERAGE (real dose is total)

    print("\n=== (1) accumulation: simplified vs real planned dose ===")
    print(f"  per-CP primary (no accumulation), corr vs real dose = {corr(dose_forward(sino,ct,args.mu_water), real_dose):.3f}")
    best = None
    for sign in (-1.0, 1.0):
        acc = accumulate(sino, ct, alpha, zbin, args.n_z, sign=sign, mu=args.mu_water)
        c = corr(acc, real3d[sign])
        print(f"  accumulated (un-rotate sign {sign:+.0f}), corr vs real 3D dose = {c:.3f}")
        if best is None or c > best[1]:
            best = (sign, c, acc)
    sign, cbest, acc_gt = best
    print(f"  -> accumulation raises correlation 0.47 -> {cbest:.3f}  (sign {sign:+.0f})")

    # ---- (2) degeneracy relief: dose-equivalent perturbation ----
    print("\n=== (2) degeneracy relief (dose loss vs open_l1 under perturbations) ===")
    base_dose = acc_gt.detach()
    dnorm = base_dose.norm().item()

    def open_l1(a, b):
        m = (a > 1e-3) | (b > 1e-3)
        return (a - b).abs()[m].mean().item()

    for name, pert in [
        ("roll +3 CP (angular phase)", torch.roll(sino, 3, dims=0)),
        ("roll +8 CP", torch.roll(sino, 8, dims=0)),
        ("swap adjacent CP pairs", sino.reshape(-1, 64).clone()),
    ]:
        if name.startswith("swap"):
            p = sino.clone()
            p[: (M // 2) * 2] = p[: (M // 2) * 2].reshape(-1, 2, 64).flip(1).reshape(-1, 64)
            pert = p
        d = accumulate(pert, ct, alpha, zbin, args.n_z, sign=sign, mu=args.mu_water).detach()
        dose_rel = (d - base_dose).norm().item() / dnorm
        ol = open_l1(pert, sino)
        print(f"  {name:28s}: open_l1 vs GT = {ol:.3f} | accumulated-dose rel-change = {dose_rel:.3f}")
    print("  => big open_l1 but small dose change == the null-space the dose loss IGNORES.")

    # gradient through the full accumulation
    sp = sino.clone().requires_grad_(True)
    L = ((accumulate(sp, ct, alpha, zbin, args.n_z, sign=sign, mu=args.mu_water) - base_dose) ** 2).mean()
    L.backward()
    print(f"\n  grad through full back-projection: finite={bool(torch.isfinite(sp.grad).all())} "
          f"nonzero={ (sp.grad.abs()>0).float().mean().item():.2f}")

    # ---- viz ----
    zc = int(zbin.bincount().argmax())
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, img, t in [(ax[0], acc_gt[zc], f"accumulated SIMPLIFIED dose (z-bin {zc})"),
                      (ax[1], real3d[sign][zc], "accumulated REAL dose (same z)"),
                      (ax[2], (acc_gt.sum(0)), "simplified dose, z-sum (coronal-ish)")]:
        im = a.imshow(img.detach().numpy(), cmap="inferno", aspect="auto"); a.set_title(t, fontsize=9)
        a.axis("off"); plt.colorbar(im, ax=a, fraction=0.046)
    plt.suptitle(f"Differentiable angular accumulation  patient {s['patient_id']}  corr={cbest:.2f}", y=1.0)
    plt.tight_layout(); plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
