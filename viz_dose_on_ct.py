#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show the dose registered on the CT (no berlingo) — axial / coronal / sagittal
views through the dose hotspot, dose overlaid on the CT.
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
    ap.add_argument("--out", default="dose_on_ct")
    args = ap.parse_args()

    ds = RTDataset(args.root, use_cache=False, debug=args.patient)
    full = ds._load_full_sample(0)
    ct = full['ct_volume'].numpy().squeeze()                 # [Z,Y,X]
    dose = full['dose_data']['dose_grid'].numpy().squeeze()  # [Z,Y,X] on CT grid
    dmax = float(dose.max())
    print(f"CT {ct.shape}  dose {dose.shape}  dose_max={dmax:.2f} Gy")

    # hotspot voxel
    zc, yc, xc = np.unravel_index(np.argmax(dose), dose.shape)
    print(f"hotspot (z,y,x)=({zc},{yc},{xc})")

    dmask = lambda d: np.ma.masked_less(d, 0.15 * dmax)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)

    # axial
    ax[0].imshow(ct[zc], cmap='gray', vmin=-1000, vmax=1000)
    im = ax[0].imshow(dmask(dose[zc]), cmap='jet', alpha=0.5, vmin=0, vmax=dmax)
    ax[0].set_title(f"axial  z={zc}")

    # coronal (fixed y)
    ax[1].imshow(ct[:, yc, :], cmap='gray', vmin=-1000, vmax=1000, aspect='auto')
    ax[1].imshow(dmask(dose[:, yc, :]), cmap='jet', alpha=0.5, vmin=0, vmax=dmax, aspect='auto')
    ax[1].set_title(f"coronal  y={yc}")

    # sagittal (fixed x)
    ax[2].imshow(ct[:, :, xc], cmap='gray', vmin=-1000, vmax=1000, aspect='auto')
    ax[2].imshow(dmask(dose[:, :, xc]), cmap='jet', alpha=0.5, vmin=0, vmax=dmax, aspect='auto')
    ax[2].set_title(f"sagittal  x={xc}")

    for a in ax:
        a.axis('off')
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("dose (Gy)")
    fig.suptitle(f"Patient {args.patient}  —  RT dose registered on CT (≥15% isodose)", fontsize=13)
    out = f"{args.out}_{args.patient}.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
