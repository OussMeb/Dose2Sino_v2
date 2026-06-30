#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the concordance between the LOT sinogram opening and the PTV position.

Pipeline: take the PTV contour from the RS DICOM, rasterize it on the CT grid,
run it through the SAME crop+rotate transform as the berlingo, project it along
the ray axis (W, last in-plane axis) with a MAX, and correlate the resulting
[N_CP, 64] "PTV projection" against the real plan sinogram.

We run it twice:
  --y-crop full   : old behaviour (Y kept full extent -> anisotropic grid)
  --y-crop field  : the fix (Y cropped to 40 cm -> isotropic grid)

so the effect of the symmetric-crop fix on concordance is directly measurable.
"""
import argparse
import numpy as np
import torch
import pydicom
import cv2
from pathlib import Path
from torch.nn import functional as F

from utils.patient import RTDataset
from utils.mask_utils import apply_tomo_transform_to_stack


def rasterize_roi_on_ct(rs, roi_names, ct_metas):
    """Build a [Z,Y,X] uint8 PTV mask on the CT grid (slices sorted asc by z)."""
    # CT geometry
    origin = np.array([float(v) for v in ct_metas[0].ImagePositionPatient])
    ps_row = float(ct_metas[0].PixelSpacing[0])   # y spacing
    ps_col = float(ct_metas[0].PixelSpacing[1])   # x spacing
    ny, nx = int(ct_metas[0].Rows), int(ct_metas[0].Columns)
    z_positions = np.array([float(m.ImagePositionPatient[2]) for m in ct_metas])
    nz = len(ct_metas)
    vol = np.zeros((nz, ny, nx), dtype=np.uint8)

    # name -> ROI number
    num_by_name = {r.ROIName: int(r.ROINumber) for r in rs.StructureSetROISequence}
    wanted = set()
    for nm in roi_names:
        if nm in num_by_name:
            wanted.add(num_by_name[nm])
    if not wanted:
        raise SystemExit(f"None of {roi_names} found. Have: {list(num_by_name)}")

    for roi_c in rs.ROIContourSequence:
        if int(roi_c.ReferencedROINumber) not in wanted:
            continue
        for c in getattr(roi_c, "ContourSequence", []):
            pts = np.array(c.ContourData, dtype=np.float64).reshape(-1, 3)
            zc = pts[:, 2].mean()
            zi = int(np.argmin(np.abs(z_positions - zc)))
            col = (pts[:, 0] - origin[0]) / ps_col
            row = (pts[:, 1] - origin[1]) / ps_row
            poly = np.stack([col, row], axis=1).round().astype(np.int32)
            cv2.fillPoly(vol[zi], [poly], 1)
    return vol


def ptv_projection(ptv_zyx, full, y_crop, ps_y, ps_x):
    """Replicate patient.py crop+transform on the PTV, return [N_CP,64] projection."""
    TARGET_HW = 64
    Z, Yf, Xf = ptv_zyx.shape
    ox = float(full['ct_origin_x']); oy = float(full['ct_origin_y']); oz = float(full['ct_origin_z'])
    cps = full['plan_data']['cps']
    x_iso = float(cps[0].IsocenterPosition[0])
    y_iso = float(cps[0].IsocenterPosition[1])
    z_iso = float(cps[0].IsocenterPosition[2])

    ptv = torch.from_numpy(ptv_zyx).float()[None, None]  # [1,1,Z,Y,X]

    # --- X crop to 40 cm around iso (always) ---
    tw = int(round(400.0 / ps_x))
    iso_vox_x = int(round((x_iso - ox) / ps_x))
    w_start = max(0, iso_vox_x - tw // 2); w_end = w_start + tw
    if w_end > Xf:
        w_end = Xf; w_start = max(0, w_end - tw)
    ptv = ptv[:, :, :, :, w_start:w_end]
    crop_x_mm = (w_end - w_start) * ps_x
    new_ox = ox + w_start * ps_x

    # --- Y crop ---
    if y_crop == "field":
        thh = int(round(400.0 / ps_y))
        iso_vox_y = int(round((y_iso - oy) / ps_y))
        h_start = max(0, iso_vox_y - thh // 2); h_end = h_start + thh
        if h_end > Yf:
            h_end = Yf; h_start = max(0, h_end - thh)
        ptv = ptv[:, :, :, h_start:h_end, :]
        crop_y_mm = (h_end - h_start) * ps_y
        new_oy = oy + h_start * ps_y
        spacing_y = crop_y_mm / TARGET_HW
    else:  # full extent (old)
        new_oy = oy
        spacing_y = (Yf * ps_y) / TARGET_HW

    ptv_r = F.interpolate(ptv, size=(Z, TARGET_HW, TARGET_HW), mode='nearest').squeeze()

    spacing_zyx = (full['ct_dz'], spacing_y, crop_x_mm / TARGET_HW)
    origin_zyx = (oz, new_oy, new_ox)

    berlingo = apply_tomo_transform_to_stack(
        mask_zyx=(ptv_r.numpy() > 0.5).astype(np.float32),
        angles=full['plan_data']['angles'],
        tables=full['plan_data']['tables'],
        x_iso=x_iso, y_iso=y_iso, z_iso=z_iso,
        spacing_zyx=spacing_zyx, origin_zyx=origin_zyx,
        is_label=True,
    )  # [N_CP, 64(H/leaf), 64(W/ray)]

    pred = berlingo.max(axis=2)  # project along ray axis -> [N_CP, 64]
    return pred


def corr(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def centroid_corr(pred, sino):
    """Per-control-point leaf-centroid correlation (robust to scale/threshold)."""
    leaves = np.arange(pred.shape[1])
    def cen(m):
        w = m.clip(0, None)
        s = w.sum(axis=1)
        c = (w * leaves[None]).sum(axis=1) / np.where(s > 0, s, 1)
        return c, s > 0
    cp, mp = cen(pred); cs, ms = cen(sino)
    m = mp & ms
    if m.sum() < 3:
        return float('nan')
    return float(np.corrcoef(cp[m], cs[m])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/tomo_data")
    ap.add_argument("--patient", required=True)
    ap.add_argument("--roi", nargs="+", default=["PTVs", "PTV sein", "PTV CMI"])
    ap.add_argument("--out", default="ptv_concordance")
    args = ap.parse_args()

    ds = RTDataset(args.root, use_cache=False, debug=args.patient)
    if len(ds) == 0:
        raise SystemExit(f"No samples for patient {args.patient}")
    full = ds._load_full_sample(0)
    info = ds.samples[0]
    ps_y = float(full['pixel_spacing'][0]); ps_x = float(full['pixel_spacing'][1])

    ct_metas = [pydicom.dcmread(str(f), stop_before_pixels=True) for f in info['ct_files']]
    ct_metas.sort(key=lambda m: float(m.ImagePositionPatient[2]))
    rs = pydicom.dcmread(str(info['rs_file']))

    ptv = rasterize_roi_on_ct(rs, args.roi, ct_metas)
    print(f"PTV voxels: {int(ptv.sum())}  grid {ptv.shape}  ps=({ps_y:.3f},{ps_x:.3f})")

    sino = np.asarray(full['plan_data']['sino'], dtype=np.float32)
    print(f"sinogram {sino.shape}  mean={sino.mean():.4f}  open%={(sino>0).mean()*100:.1f}")

    # Gantry-angle buckets: leaf axis == patient-Y near theta 90/270 (fix matters),
    # == patient-X near theta 0/180 (X always cropped -> fix irrelevant).
    angles = np.asarray(full['plan_data']['angles'], dtype=np.float64)
    th = np.deg2rad(angles)
    leafY = np.abs(np.sin(th)) > 0.7071   # theta within 45 deg of 90/270
    leafX = ~leafY

    def leaf_spread(m):
        leaves = np.arange(m.shape[1]); out = []
        for row in m:
            w = row.clip(0, None); s = w.sum()
            if s <= 0:
                out.append(np.nan); continue
            mu = (w * leaves).sum() / s
            out.append(np.sqrt((w * (leaves - mu) ** 2).sum() / s))
        return np.array(out)

    sino_spread = leaf_spread((sino > 0).astype(np.float32))

    results = {}
    preds = {}
    for mode in ("full", "field"):
        pred = ptv_projection(ptv, full, mode, ps_y, ps_x)
        preds[mode] = pred
        c = corr(pred, (sino > 0).astype(np.float32))
        cc = centroid_corr(pred, sino)
        ccY = centroid_corr(pred[leafY], sino[leafY])
        ccX = centroid_corr(pred[leafX], sino[leafX])
        # leaf-extent ratio pred/sino on the Y-leaf bucket (where the fix acts)
        ps = leaf_spread(pred); m = leafY & np.isfinite(ps) & np.isfinite(sino_spread)
        spread_ratio = float(np.nanmedian(ps[m] / sino_spread[m])) if m.sum() else float('nan')
        results[mode] = (c, cc)
        print(f"[y-crop={mode:5s}] pixel-corr={c:+.3f}  centroid-corr all={cc:+.3f} "
              f"leafY={ccY:+.3f} leafX={ccX:+.3f}  spread(pred/sino,leafY)={spread_ratio:.2f}")

    # Visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(16, 5))
        ax[0].imshow(sino, aspect='auto', cmap='magma'); ax[0].set_title("sinogram (target)")
        ax[1].imshow((sino > 0), aspect='auto', cmap='gray'); ax[1].set_title("sino open mask")
        ax[2].imshow(preds['full'], aspect='auto', cmap='gray')
        ax[2].set_title(f"PTV proj Y=full\ncorr={results['full'][0]:+.2f} cen={results['full'][1]:+.2f}")
        ax[3].imshow(preds['field'], aspect='auto', cmap='gray')
        ax[3].set_title(f"PTV proj Y=field (fix)\ncorr={results['field'][0]:+.2f} cen={results['field'][1]:+.2f}")
        for a in ax:
            a.set_xlabel("leaf"); a.set_ylabel("control point")
        fig.suptitle(f"Patient {args.patient}  -  sinogram opening vs PTV projection")
        fig.tight_layout()
        out = f"{args.out}_{args.patient}.png"
        fig.savefig(out, dpi=110)
        print(f"Saved {out}")
    except Exception as e:
        print(f"viz skipped: {e}")


if __name__ == "__main__":
    main()
