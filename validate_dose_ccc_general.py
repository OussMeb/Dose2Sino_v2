#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generalization CCC check on any val patient: dose-model predicted sinogram vs the GT
plan, both through DoseCUDA's real CCC on the same grid. Confirms the 0.991 result."""
import sys, time, tempfile, argparse, numpy as np, torch, pydicom
from pathlib import Path
from DoseCUDA import TomoDoseGrid, TomoPlan
from utils.patient import RTDataset
from models.sinogram_2p5d import DosePrediction2p5D
from utils.dose_operator import find_plan

CKPT_DEFAULT = "checkpoints/20260626_200610_2p5d_dose/best_model_new_session_session_0_.pth"
SINO_TAG = (0x300D, 0x10A7)

ap = argparse.ArgumentParser()
ap.add_argument("--patient", required=True)
ap.add_argument("--ckpt", default=CKPT_DEFAULT, help="Model checkpoint to evaluate")
args = ap.parse_args()
pat = args.patient
CKPT = args.ckpt
dev = "cuda"

# 1) predict sinogram with the dose-trained model (base32)
ds = RTDataset("/mnt/data/tomo_data/", augmentation=None, use_cache=True,
               cache_dir="/mnt/data/tomo_data/cache_sino_r8", reduction_ratio=8, debug=pat)
s = ds[0]; inp = s["input"].unsqueeze(0).to(dev)
m = DosePrediction2p5D(base_filters=32, in_channel=2, n_leaves=64, reduce_h=False).to(dev)
m.load_state_dict(torch.load(CKPT, map_location=dev)["model_state_dict"]); m.eval()
with torch.no_grad():
    pred = torch.sigmoid(m(inp)[:, 0, :, :, 0])[0].cpu().numpy()
print(f"{pat}: pred sino {pred.shape} mean {pred.mean():.4f}", flush=True)

plan_path = find_plan("/mnt/data/tomo_data/", s["patient_id"], s["pareto_index"])
ct_dir = str(Path(plan_path).parent.parent)            # Tomo_* folder holding CT*.dcm

# 2) injected (predicted) plan
dsp = pydicom.dcmread(plan_path, force=True)
cps = dsp[(0x300A, 0x00B0)][0][(0x300A, 0x0111)].value
n = min(len(cps), pred.shape[0])
for i in range(n):
    cps[i][SINO_TAG].value = "\\".join(f"{v:.7g}" for v in pred[i]).encode()
tmp = tempfile.NamedTemporaryFile(suffix="_PRED.dcm", delete=False, dir="/mnt/data/sinogram_generator")
dsp.save_as(tmp.name)
iso = np.array([float(v) for v in cps[0][(0x300A, 0x012C)].value])

# 3) one CT/grid, two CCC computes (GT plan, predicted plan)
dose = TomoDoseGrid()
dose.loadCTDCM(ct_dir); dose.resampleCTfromSpacing(2.5)
dose.setDoseROI(bbox_min_mm=[iso[0]-180, iso[1]-180, iso[2]-130],
                bbox_max_mm=[iso[0]+180, iso[1]+180, iso[2]+130])
pg = TomoPlan("Tomo"); pg.readPlanDicom(plan_path, n_sub_cps=3)
t = time.time(); dose.computeTomoPlan(pg, gpu_id=0); dgt = dose.dose.copy()
print(f"  GT CCC {time.time()-t:.0f}s max {dgt.max():.3f}", flush=True)
pp = TomoPlan("Tomo"); pp.readPlanDicom(tmp.name, n_sub_cps=3)
t = time.time(); dose.computeTomoPlan(pp, gpu_id=0); dp = dose.dose.copy()
print(f"  PRED CCC {time.time()-t:.0f}s max {dp.max():.3f}", flush=True)
Path(tmp.name).unlink(missing_ok=True)

# 4) compare
msk = dgt > 0.1 * dgt.max()
def corr(a, b, mk=None):
    if mk is not None: a, b = a[mk], b[mk]
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
gd = np.abs(dp - dgt)
print(f"=== {pat}: corr full {corr(dp,dgt):.4f} | corr region {corr(dp,dgt,msk):.4f} | "
      f"mean region pred {dp[msk].mean():.3f} gt {dgt[msk].mean():.3f} | "
      f"maxhot pred {dp.max():.2f} gt {dgt.max():.2f} | |d| {gd[msk].mean():.3f}Gy "
      f"({100*gd[msk].mean()/dgt[msk].mean():.1f}%)", flush=True)
