#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"Shape right, values off" — is it sigmoid saturation/compression (removing sigmoid
would help) or conditional-mean regression-to-the-mean (removing it would NOT help)?

Loads a trained checkpoint, runs on a few val patients, and reports on OPEN leaves:
  - bias mean(pred-target), dynamic-range ratio std(pred)/std(target)
  - calibration curve (binned target -> mean pred): regression-to-mean if high
    targets are under-predicted and low ones over-predicted
  - sigmoid regime: logit distribution + fraction of preds saturated near 0/1
"""
import argparse, re
import numpy as np
import torch
from utils.patient import RTDataset, get_patient_based_splits
from models.sinogram_2p5d import DosePrediction2p5D


def load_model(ckpt, base, device):
    m = DosePrediction2p5D(base_filters=base, in_channel=2, n_leaves=64,
                           slice_chunk=256, reduce_h=False).to(device).eval()
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    m.load_state_dict(sd)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--cache", default="/mnt/data/tomo_data/cache_sino_r8")
    ap.add_argument("--n-patients", type=int, default=3)
    ap.add_argument("--tag", default="ckpt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RTDataset("/mnt/data/tomo_data", use_cache=True, cache_dir=args.cache,
                   reduction_ratio=8, target_hw=64)
    _, val_idx, _ = get_patient_based_splits(ds, 0.7, 0.15, 0.15, seed=42)
    # one sample per distinct val patient, up to n-patients
    seen, idxs = set(), []
    for i in val_idx:
        pid = ds.samples[i]["patient_id"]
        if pid not in seen:
            seen.add(pid); idxs.append(i)
        if len(idxs) >= args.n_patients: break

    m = load_model(args.ckpt, args.base, device)
    P, T, L = [], [], []
    with torch.no_grad():
        for i in idxs:
            s = ds[i]
            x = s["input"].unsqueeze(0).to(device)       # [1,2,N_CP,64,64]
            logits = m(x).squeeze().cpu()                # [N_CP,64]
            tgt = s["target"].float()                    # [N_CP,64]
            P.append(torch.sigmoid(logits).numpy().ravel())
            T.append(tgt.numpy().ravel()); L.append(logits.numpy().ravel())
    p = np.concatenate(P); t = np.concatenate(T); lg = np.concatenate(L)

    openm = t > 0.05
    po, to, lo = p[openm], t[openm], lg[openm]
    print(f"\n===== {args.tag} (base{args.base}, {len(idxs)} val patients) =====")
    print(f"open leaves: {openm.sum()} / {openm.size} ({100*openm.mean():.1f}%)")
    print(f"OPEN  target mean={to.mean():.3f} std={to.std():.3f}")
    print(f"OPEN  pred   mean={po.mean():.3f} std={po.std():.3f}   "
          f"-> bias={np.mean(po-to):+.3f}  range-ratio std(pred)/std(tgt)={po.std()/to.std():.2f}")
    print(f"OPEN  |pred-tgt| (open_l1) = {np.mean(np.abs(po-to)):.4f}")
    # saturation
    print(f"pred near 0 (<0.02): {100*np.mean(p<0.02):.1f}%   near 1 (>0.98): {100*np.mean(p>0.98):.1f}%")
    print(f"OPEN  logits: mean={lo.mean():+.2f} std={lo.std():.2f}  "
          f"|logit|>4 (saturated): {100*np.mean(np.abs(lo)>4):.1f}%   in [-2,2] (linear): {100*np.mean(np.abs(lo)<2):.1f}%")
    # calibration: binned target -> mean pred
    print("calibration (target bin -> mean pred [n]):")
    edges = [0.05,0.1,0.2,0.3,0.4,0.5,0.7,1.01]
    for a,b in zip(edges[:-1],edges[1:]):
        msk=(to>=a)&(to<b)
        if msk.sum()>20:
            print(f"  tgt[{a:.2f},{b:.2f}): mean_pred={po[msk].mean():.3f}  (n={msk.sum()})")


if __name__ == "__main__":
    main()
