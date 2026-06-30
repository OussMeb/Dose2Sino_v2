#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why the dose sits near the edge in the berlingo: the slices are rotated about the
ISOCENTER (center of the 64x64 grid), while the breast target is ~12 cm lateral to
it. Show several control points: the off-iso target traces a circle around the
centered cross -> it reaches the edges at the extremes (= the sinogram sinusoid).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.patient import RTDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/tomo_data")
    ap.add_argument("--patient", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="berlingo_sweep")
    args = ap.parse_args()

    ds = RTDataset(args.root, use_cache=False, debug=args.patient)
    s = ds[0]
    ct = s['input'][0].numpy()    # [N_CP,64,64]
    dose = s['input'][1].numpy()  # [N_CP,64,64]
    full = ds._load_full_sample(0)
    angles = np.asarray(full['plan_data']['angles'], dtype=float)

    # control points where the dose is actually present (target in the slice)
    active = np.argwhere(dose.reshape(dose.shape[0], -1).max(1) > 0.05).ravel()
    sel = active[np.linspace(0, len(active) - 1, args.n).round().astype(int)]

    fig, ax = plt.subplots(1, args.n, figsize=(3.0 * args.n, 3.6), constrained_layout=True)
    for a, i in zip(ax, sel):
        a.imshow(ct[i], cmap='gray', vmin=0, vmax=1)
        a.imshow(np.ma.masked_less(dose[i], 0.05), cmap='jet', alpha=0.6, vmin=0, vmax=dose.max())
        # isocenter = center of rotation = grid center
        a.axhline(32, color='cyan', lw=0.6, alpha=0.7)
        a.axvline(32, color='cyan', lw=0.6, alpha=0.7)
        a.plot(32, 32, '+', color='cyan', ms=10)
        a.set_title(f"CP {i}  gantry {angles[i]:.0f}°", fontsize=10)
        a.set_xlabel("ray axis W (collapsed)")
        a.set_ylabel("leaf axis H")
        a.set_xlim(0, 64); a.set_ylim(64, 0)
    fig.suptitle(f"Patient {args.patient}  —  berlingo dose slices (cyan + = isocenter / rotation center)",
                 fontsize=12)
    out = f"{args.out}_{args.patient}.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}  (iso at grid center; off-iso breast sweeps the edges)")


if __name__ == "__main__":
    main()
