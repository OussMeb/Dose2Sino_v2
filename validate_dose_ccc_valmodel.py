#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_dose_ccc.py — the GOLD validation: does the PREDICTED sinogram deliver the
right REAL dose, judged by DoseCUDA's collapsed-cone engine (not the 0.91 ray-tracer)?

Pipeline: model checkpoint -> predicted sinogram -> inject into a copy of the plan
DICOM (Tomo LOT tag 300D,10A7) -> DoseCUDA CCC -> compare to the GT plan's CCC dose
(same ROI/grid). ~14 min/plan on the M6000, so this is an OFFLINE judge for a few
cases, not a training loss.
"""
import sys, time, shutil, tempfile, numpy as np, torch, pydicom
from pathlib import Path
from DoseCUDA import TomoDoseGrid, TomoPlan
from utils.patient import RTDataset
from models.sinogram_2p5d import DosePrediction2p5D

D = "/mnt/data/tomo_data/183040/RC_Publi_Tomo_Halcyon_DIBH/Tomo_FB_copy"
PLAN = D + "/pareto_0/RP1.2.752.243.1.1.20250317095242862.1300.83153.dcm"
CKPT = "checkpoints/20260626_200610_2p5d_dose/best_model_new_session_session_0_.pth"
SINO_TAG = (0x300D, 0x10A7)

# 1) predicted sinogram from the overfit checkpoint
dev = "cuda"
ds = RTDataset("/mnt/data/tomo_data/", augmentation=None, use_cache=True,
               cache_dir="/mnt/data/tomo_data/cache_sino_r8", reduction_ratio=8, debug="183040")
s = ds[0]; inp = s["input"].unsqueeze(0).to(dev)
m = DosePrediction2p5D(base_filters=32, in_channel=2, n_leaves=64, reduce_h=False).to(dev)
m.load_state_dict(torch.load(CKPT, map_location=dev)["model_state_dict"]); m.eval()
with torch.no_grad():
    pred = torch.sigmoid(m(inp)[:, 0, :, :, 0])[0].cpu().numpy()   # [N_CP,64] in [0,1]
print(f"pred sino {pred.shape} mean {pred.mean():.4f} (GT mean ~0.1145)", flush=True)

# 2) inject predicted LOT into a copy of the plan DICOM
ds_plan = pydicom.dcmread(PLAN, force=True)
cps = ds_plan[(0x300A, 0x00B0)][0][(0x300A, 0x0111)].value
assert len(cps) == pred.shape[0], f"CP count {len(cps)} vs pred {pred.shape[0]}"
for i, cp in enumerate(cps):
    cp[SINO_TAG].value = "\\".join(f"{v:.7g}" for v in pred[i]).encode()
tmp = tempfile.NamedTemporaryFile(suffix="_PRED.dcm", delete=False, dir="/mnt/data/sinogram_generator")
ds_plan.save_as(tmp.name); print(f"injected plan -> {tmp.name}", flush=True)

# 3) CCC dose of the predicted plan (same ROI/grid as the GT run)
iso = np.array([float(v) for v in cps[0][(0x300A, 0x012C)].value])
dose = TomoDoseGrid(); plan = TomoPlan("Tomo")
dose.loadCTDCM(D); dose.resampleCTfromSpacing(2.5)
plan.readPlanDicom(tmp.name, n_sub_cps=3)
dose.setDoseROI(bbox_min_mm=[iso[0]-180, iso[1]-180, iso[2]-130],
                bbox_max_mm=[iso[0]+180, iso[1]+180, iso[2]+130])
print("computeTomoPlan (pred)...", flush=True); t = time.time()
dose.computeTomoPlan(plan, gpu_id=0)
print(f"  TIME {time.time()-t:.1f}s  max {dose.dose.max():.4f} Gy", flush=True)
dpred = dose.dose

# 4) compare to GT CCC dose
dgt = np.load("/mnt/data/sinogram_generator/ccc_gt_183040.npy")
if dpred.shape != dgt.shape:
    print(f"  shape mismatch pred {dpred.shape} vs gt {dgt.shape}; cropping to min", flush=True)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(dpred.shape, dgt.shape))
    dpred, dgt = dpred[sl], dgt[sl]
m_ = dgt > 0.1 * dgt.max()
def corr(a, b, msk=None):
    if msk is not None: a, b = a[msk], b[msk]
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
print("\n=== REAL CCC DOSE: predicted sinogram vs GT sinogram ===")
print(f"  corr full           {corr(dpred, dgt):.4f}")
print(f"  corr in dose-region {corr(dpred, dgt, m_):.4f}")
print(f"  max dose  pred {dpred.max():.3f} Gy   gt {dgt.max():.3f} Gy")
print(f"  mean(region) pred {dpred[m_].mean():.3f}   gt {dgt[m_].mean():.3f} Gy")
gd = np.abs(dpred - dgt)
print(f"  mean |Δdose| in region: {gd[m_].mean():.4f} Gy ({100*gd[m_].mean()/dgt[m_].mean():.1f}% of mean)")
np.save("/mnt/data/sinogram_generator/ccc_pred_183040_dosemodel.npy", dose.dose)
print("saved ccc_pred_183040_dosemodel.npy", flush=True)
