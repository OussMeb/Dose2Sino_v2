#!/usr/bin/env python3
"""
How DEGENERATE is anatomy -> LOT sinogram? The key question behind "should we get
more data". For each patient that has >=2 Pareto plans (SAME anatomy/CT, different
optimizer tradeoff), measure how much the GT sinograms differ across plans. If that
spread is ~the model's 0.15 floor, then the floor is largely an irreducible
(aleatoric / optimizer-null-space) ceiling that MORE same-distribution patients
cannot lower -- you'd need either richer conditioning or a dose-equivalent target.
If the spread is small, the floor is model/coverage-limited and more data helps.
"""
from pathlib import Path
import numpy as np
import pydicom
from collections import defaultdict

SINO_TAG = (0x300D, 0x10A7)
DATA = Path("/mnt/data/tomo_data")


def extract_sino(p):
    ds = pydicom.dcmread(p, force=True)
    try:
        beam = ds[(0x300A, 0x00B0)][0]
        cps = beam[(0x300A, 0x0111)].value
    except Exception:
        return None
    rows = []
    for cp in cps:
        if SINO_TAG not in cp:
            continue
        v = cp[SINO_TAG].value
        if isinstance(v, (bytes, bytearray)):
            r = [float(x) for x in v.decode(errors="ignore").split("\\") if x.strip() != ""]
            if len(r) == 64:
                rows.append(r)
    return np.asarray(rows, np.float32) if len(rows) >= 16 else None


def open_l1(a, b):
    """Mean |a-b| on leaves open in EITHER plan (the metric the model is judged on)."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = (a > 1e-3) | (b > 1e-3)
    return float(np.abs(a - b)[m].mean()) if m.any() else np.nan


# group plans by patient (top-level folder under DATA)
plans_by_pat = defaultdict(list)
for p in sorted(DATA.glob("*/**/pareto_*/RP*.dcm")):
    pat = p.relative_to(DATA).parts[0]
    plans_by_pat[pat].append(p)

cross, within_open_frac = [], []
n_multi = 0
for pat, plans in plans_by_pat.items():
    if len(plans) < 2:
        continue
    sinos = [s for s in (extract_sino(p) for p in plans) if s is not None]
    if len(sinos) < 2:
        continue
    n_multi += 1
    for i in range(len(sinos)):
        for j in range(i + 1, len(sinos)):
            cross.append(open_l1(sinos[i], sinos[j]))
    within_open_frac.append(np.mean([(s > 1e-3).mean() for s in sinos]))

cross = np.array([c for c in cross if np.isfinite(c)])
print(f"patients with >=2 Pareto plans: {n_multi}")
print(f"cross-plan pairs compared: {len(cross)}")
print(f"\nCROSS-PLAN open_l1 (same anatomy, different plan):")
print(f"  mean {cross.mean():.3f}  median {np.median(cross):.3f}  "
      f"[p10 {np.percentile(cross,10):.3f} .. p90 {np.percentile(cross,90):.3f}]")
print(f"\nFor reference: model val open_l1 floor ~0.153 (vs GT of ONE chosen plan).")
print(f"mean open fraction per sinogram: {np.mean(within_open_frac):.3f}")
