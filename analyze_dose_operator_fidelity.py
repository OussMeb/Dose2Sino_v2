#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_dose_operator_fidelity.py

The dose-consistency loss (lever B) is only as trustworthy as the simplified dose
operator's fidelity vs the real planned dose. On 183040/217035 it is 0.92-0.94 but
on 187591 only 0.55 -> fidelity is PATIENT-DEPENDENT. Before committing to a full
training run, characterise the DISTRIBUTION of operator fidelity across patients:
- mostly >0.9 with a small low tail -> usable (down-weight/exclude low-fidelity cases);
- many ~0.6 -> the operator is not ready and the loss would be unreliable.

Per patient (one pareto): corr( accumulated simplified dose , real planned dose ),
with the tuned physical operator (field_z=2.5, scatter_xy=1.0).
"""
import logging
import numpy as np
import torch
from utils.patient import RTDataset
from utils.dose_operator import dose_forward, accumulate, build_zbin, read_geometry, find_plan

logging.basicConfig(level=logging.WARNING)


def corr(a, b):
    a, b = a.flatten(), b.flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main(n_patients=25, n_z=48):
    from pathlib import Path
    pats = sorted({f.name.split("_")[0] + ("_DIBH" if "_DIBH" in f.name else "")
                   for f in Path("/mnt/data/tomo_data/cache_sino_r8").glob("*.pt.gz")})
    pats = pats[:n_patients]
    rows = []
    for pat in pats:
        try:
            ds = RTDataset("/mnt/data/tomo_data/", augmentation=None, use_cache=True,
                           cache_dir="/mnt/data/tomo_data/cache_sino_r8", reduction_ratio=8, debug=pat)
            s = ds[0]
            ct = s["input"][0].float(); real = s["input"][1].float(); sino = s["target"].float()
            N = ct.shape[0]
            plan = find_plan("/mnt/data/tomo_data/", s["patient_id"], s["pareto_index"])
            ang, tab = read_geometry(plan)
            alpha = torch.tensor(90.0 - ang[:N]); zbin = build_zbin(tab[:N], n_z)
            real3d = accumulate(real, alpha, zbin, n_z, sign=1.0, reduce="mean")
            sim = accumulate(dose_forward(sino, ct, 0.03), alpha, zbin, n_z, sign=1.0,
                             reduce="sum", field_z=2.5, scatter_xy=1.0)
            c = corr(sim, real3d)
            rows.append((pat, c))
            print(f"{pat:14s} corr {c:.3f}")
        except Exception as e:
            print(f"{pat:14s} SKIP ({type(e).__name__}: {e})")
    cs = np.array([c for _, c in rows])
    print(f"\n=== fidelity over {len(cs)} patients ===")
    print(f"  mean {cs.mean():.3f}  median {np.median(cs):.3f}  min {cs.min():.3f}  max {cs.max():.3f}")
    for thr in (0.9, 0.85, 0.8, 0.7):
        print(f"  fraction >= {thr}: {(cs >= thr).mean():.2f}")


if __name__ == "__main__":
    main()
