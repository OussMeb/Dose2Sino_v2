#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the dose correctly registered into the CT frame?

Loads CT volume + dose (already resampled onto the CT grid by _load_rt_dose),
plus the PTV rasterized on the same grid, and overlays them on axial slices.
If the registration is right, the high-dose region must sit on the patient and
hug the PTV.
"""
import argparse
import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.patient import RTDataset
from check_ptv_concordance import rasterize_roi_on_ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/tomo_data")
    ap.add_argument("--patient", required=True)
    ap.add_argument("--roi", nargs="+", default=["PTVs", "PTV sein", "PTV CMI"])
    ap.add_argument("--out", default="dose_ct_reg")
    args = ap.parse_args()

    ds = RTDataset(args.root, use_cache=False, debug=args.patient)
    full = ds._load_full_sample(0)
    info = ds.samples[0]

    ct = full['ct_volume'].numpy().squeeze()              # [Z,Y,X]
    dose = full['dose_data']['dose_grid'].numpy().squeeze()  # [Z,Y,X] on CT grid
    print(f"CT {ct.shape}  dose {dose.shape}  dose_max={dose.max():.2f}")

    ct_metas = [pydicom.dcmread(str(f), stop_before_pixels=True) for f in info['ct_files']]
    ct_metas.sort(key=lambda m: float(m.ImagePositionPatient[2]))
    rs = pydicom.dcmread(str(info['rs_file']))
    ptv = rasterize_roi_on_ct(rs, args.roi, ct_metas)     # [Z,Y,X]

    # pick the 3 axial slices with the most dose
    dose_per_z = dose.reshape(dose.shape[0], -1).sum(axis=1)
    zsel = np.argsort(dose_per_z)[-3:][::-1]

    dmax = float(dose.max())
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for a, z in zip(ax, zsel):
        a.imshow(ct[z], cmap='gray', vmin=-1000, vmax=1000)
        a.imshow(np.ma.masked_less(dose[z], 0.2 * dmax), cmap='jet',
                 alpha=0.45, vmin=0, vmax=dmax)
        if ptv[z].any():
            a.contour(ptv[z], levels=[0.5], colors='lime', linewidths=1.2)
        a.set_title(f"z={z}  (dose jet, PTV green)")
        a.axis('off')
    fig.suptitle(f"Patient {args.patient}  —  dose & PTV registered on CT (axial)", fontsize=13)
    out = f"{args.out}_{args.patient}.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
