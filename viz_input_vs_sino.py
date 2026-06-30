#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show the model input (CT + dose berlingo) next to the target sinogram.

The berlingo is [2, N_CP, H(leaf=64), W(ray=64)]. To compare in the sinogram's
own [N_CP, 64] frame we collapse the ray axis W (CT: mean silhouette, dose: max
= "dose anywhere along the ray"), which is exactly the reduction the model does.
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
    ap.add_argument("--out", default="input_vs_sino")
    args = ap.parse_args()

    ds = RTDataset(args.root, use_cache=False, debug=args.patient)
    if len(ds) == 0:
        raise SystemExit(f"No samples for patient {args.patient}")
    s = ds[0]
    x = s['input'].numpy()          # [2, N_CP, 64, 64]
    sino = s['target'].numpy()      # [N_CP, 64]
    ct, dose = x[0], x[1]           # each [N_CP, 64, 64]

    ct_proj = ct.mean(axis=2)       # collapse ray -> [N_CP, 64] anatomy silhouette
    dose_proj = dose.max(axis=2)    # collapse ray -> [N_CP, 64] dose presence
    ncp = sino.shape[0]

    fig, ax = plt.subplots(1, 4, figsize=(17, 6), constrained_layout=True)
    kw = dict(aspect='auto', origin='upper', extent=[0, 64, ncp, 0])

    ax[0].imshow(ct_proj, cmap='gray', **kw)
    ax[0].set_title("CT input\n(ray-projected)")
    ax[1].imshow(dose_proj, cmap='magma', **kw)
    ax[1].set_title("Dose input\n(ray-projected, max)")
    ax[2].imshow(sino, cmap='viridis', **kw)
    ax[2].set_title("Sinogram (target)\nLOT [N_CP, 64]")

    # overlay: sinogram opening contour on top of the dose projection
    ax[3].imshow(dose_proj, cmap='gray', **kw)
    ax[3].contour(np.linspace(0, 64, 64), np.linspace(0, ncp, ncp),
                  (sino > 0.05).astype(float), levels=[0.5], colors='red', linewidths=0.6)
    ax[3].set_title("Dose input + sinogram\nopening (red)")

    for a in ax:
        a.set_xlabel("leaf (0-63)")
        a.set_ylabel("control point")
    fig.suptitle(f"Patient {args.patient}  —  input (CT+dose) vs sinogram  (same [N_CP, 64] frame)",
                 fontsize=13)
    out = f"{args.out}_{args.patient}.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}  | input {x.shape}  sino {sino.shape}")


if __name__ == "__main__":
    main()
