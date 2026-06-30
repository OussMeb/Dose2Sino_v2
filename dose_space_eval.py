#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is open_l1 ~0.158 dosimetrically good?  Evaluate the sinogram predictor in DOSE space
using the learned forward surrogate g.

Key metric (isolates the sinogram-prediction's dosimetric impact, cancelling g's own
bias since both sinograms pass through g):
    dose_impact = MAE( g(CT, sino_pred) , g(CT, sino_GT) )
Context:
    g_err  = MAE( g(CT, sino_GT) , real_dose )      # surrogate's own error
    total  = MAE( g(CT, sino_pred), real_dose )
All as fraction of the per-patient dose peak.
"""
import argparse
import numpy as np
import torch
from utils.patient import RTDataset, get_patient_based_splits
from models.sinogram_2p5d import DosePrediction2p5D
from train_surrogate import DoseSurrogate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sino-ckpt", default="checkpoints/20260620_121840_2p5d/best_model_new_session_session_0_.pth")
    ap.add_argument("--sino-base", type=int, default=24)
    ap.add_argument("--surrogate", default="checkpoints/surrogate_dose.pth")
    ap.add_argument("--n-patients", type=int, default=6)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RTDataset("/mnt/data/tomo_data", use_cache=True, cache_dir="/mnt/data/tomo_data/cache_sino_r8",
                   reduction_ratio=8, target_hw=64)
    _, va, _ = get_patient_based_splits(ds, 0.7, 0.15, 0.15, seed=42)
    seen, idxs = set(), []
    for i in va:
        pid = ds.samples[i]["patient_id"]
        if pid not in seen:
            seen.add(pid); idxs.append(i)
        if len(idxs) >= args.n_patients: break

    # sinogram predictor
    fm = DosePrediction2p5D(base_filters=args.sino_base, in_channel=2, n_leaves=64,
                            slice_chunk=256, reduce_h=False).to(dev).eval()
    sd = torch.load(args.sino_ckpt, map_location=dev, weights_only=False)
    fm.load_state_dict(sd.get("model_state_dict", sd))
    # surrogate
    cs = torch.load(args.surrogate, map_location=dev, weights_only=False)
    g = DoseSurrogate(base=cs.get("base", 16)).to(dev).eval()
    g.load_state_dict(cs["model_state_dict"])

    def gdose(ct, sino):
        sb = sino.unsqueeze(-1).expand(-1, -1, ct.shape[-1])
        x = torch.stack([ct, sb], 0).unsqueeze(0)
        return g(x).squeeze()

    print(f"{'patient':>10} | dose_impact | g_err | total  (all % of peak)   open_l1")
    imp, ge, to, ol = [], [], [], []
    with torch.no_grad():
        for i in idxs:
            s = ds[i]
            ct = s["input"][0].to(dev)                 # [N_CP,64,64]
            real = s["input"][1].to(dev)               # real dose berlingo
            sino_gt = s["target"].float().to(dev)      # [N_CP,64]
            logits = fm(s["input"].unsqueeze(0).to(dev)).squeeze()
            sino_pred = torch.sigmoid(logits)          # [N_CP,64]

            d_pred = gdose(ct, sino_pred)
            d_gt = gdose(ct, sino_gt)
            peak = real.max().item() + 1e-6
            impact = (d_pred - d_gt).abs().mean().item() / peak
            gerr = (d_gt - real).abs().mean().item() / peak
            total = (d_pred - real).abs().mean().item() / peak
            o = (sino_pred - sino_gt).abs()[sino_gt > 0.05].mean().item()
            imp.append(impact); ge.append(gerr); to.append(total); ol.append(o)
            print(f"{ds.samples[i]['patient_id']:>10} |   {impact*100:5.2f}%    | {gerr*100:5.2f}% | {total*100:5.2f}%      {o:.4f}")
    print("-"*72)
    print(f"{'MEAN':>10} |   {np.mean(imp)*100:5.2f}%    | {np.mean(ge)*100:5.2f}% | {np.mean(to)*100:5.2f}%      {np.mean(ol):.4f}")
    print(f"\ndose_impact = extra dose error from PREDICTING the sinogram (vs using GT), via g.")
    print(f"If dose_impact is small (<~ g_err), the open_l1 {np.mean(ol):.3f} is dosimetrically benign.")


if __name__ == "__main__":
    main()
