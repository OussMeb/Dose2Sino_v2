#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0 — forward dose surrogate  g : (CT berlingo, sinogram) -> dose berlingo.

Learns the WELL-POSED forward physics (LOT -> dose) from the 1062 cached
(sinogram, dose) pairs, as a stand-in for the TPS dose engine. Go/no-go gate for
the dose-space evaluation + forward-consistency loss: if g reproduces the dose
well on held-out patients, the whole dose-space direction is viable.

Representation (reuses cache_sino_r8, target_hw=64):
  - input  x = [CT_berlingo, sino_broadcast]  -> [B,2,N_CP,64,64]
           sino[i,j] (leaf j LOT at CP i) is broadcast along the ray axis W.
  - target y = dose_berlingo                   -> [B,1,N_CP,64,64]
Dose accumulates over neighbouring projections (helical) -> conv along N_CP gives
the cross-projection receptive field.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u train_surrogate.py --epochs 6
"""
import argparse, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils.patient import RTDataset, get_patient_based_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def cbr(ci, co, stride=1, norm=True):
    # NOTE: InstanceNorm3d normalizes per-instance feature scale -> makes the net
    # AMPLITUDE-INVARIANT to the input. For a dose surrogate (dose ~ linear in LOT)
    # that is wrong: it caused g(0.5xLOT)~=g(LOT). norm=False keeps scale info.
    layers = [nn.Conv3d(ci, co, 3, stride=stride, padding=1)]
    if norm: layers.append(nn.InstanceNorm3d(co))
    layers += [nn.LeakyReLU(0.1, True), nn.Conv3d(co, co, 3, padding=1)]
    if norm: layers.append(nn.InstanceNorm3d(co))
    layers.append(nn.LeakyReLU(0.1, True))
    return nn.Sequential(*layers)


class DoseSurrogate(nn.Module):
    """Compact 3D U-Net, stride-2 on all axes (dose is smooth along N_CP)."""
    def __init__(self, base=16, use_norm=True):
        super().__init__()
        n = use_norm
        self.e0 = cbr(2, base, stride=2, norm=n)        # 1/2
        self.e1 = cbr(base, base*2, stride=2, norm=n)   # 1/4
        self.e2 = cbr(base*2, base*4, stride=2, norm=n) # 1/8
        self.bott = cbr(base*4, base*4, norm=n)
        self.d2 = cbr(base*4 + base*4, base*2, norm=n)
        self.d1 = cbr(base*2 + base*2, base, norm=n)
        self.d0 = cbr(base + base, base, norm=n)
        self.out = nn.Conv3d(base, 1, 1)

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode='trilinear', align_corners=False)

    def forward(self, x):
        in_size = x.shape[2:]
        s0 = self.e0(x); s1 = self.e1(s0); s2 = self.e2(s1)
        b = self.bott(s2)
        d2 = self.d2(torch.cat([self._up(b, s1), self._up(s2, s1)], 1))
        d1 = self.d1(torch.cat([self._up(d2, s0), self._up(s1, s0)], 1))
        d0 = self.d0(torch.cat([self._up(d1, s0), s0], 1))
        y = self.out(d0)
        return F.interpolate(y, size=in_size, mode='trilinear', align_corners=False)


def make_xy(sample, device):
    inp = sample["input"]                       # [2,N_CP,64,64] (ct, dose)
    ct = inp[0:1]                               # [1,N_CP,64,64]
    dose = inp[1:2]                             # [1,N_CP,64,64]  (target)
    sino = sample["target"].float()            # [N_CP,64]
    sino_b = sino.unsqueeze(-1).expand(-1, -1, ct.shape[-1])  # [N_CP,64,64] broadcast on W
    x = torch.stack([ct[0], sino_b], 0).unsqueeze(0)          # [1,2,N_CP,64,64]
    y = dose.unsqueeze(0)                                      # [1,1,N_CP,64,64]
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, ds, idxs, device):
    model.eval(); maes=[]; rels=[]
    for i in idxs:
        x, y = make_xy(ds[i], device)
        p = model(x)
        mae = (p - y).abs().mean().item()
        rng = (y.max() - y.min()).item() + 1e-6
        maes.append(mae); rels.append(mae / rng)
    return float(np.mean(maes)), float(np.mean(rels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--cache", default="/mnt/data/tomo_data/cache_sino_r8")
    ap.add_argument("--save", default="checkpoints/surrogate_dose.pth")
    ap.add_argument("--amp-aug", type=float, default=0.0,
                    help="If >0, per-sample scale (sino,dose) by a~U[1-amp_aug,1+amp_aug] "
                         "to force amplitude sensitivity (dose is ~linear in fluence).")
    ap.add_argument("--no-norm", action="store_true",
                    help="Drop InstanceNorm (it makes the net amplitude-invariant).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RTDataset("/mnt/data/tomo_data", use_cache=True, cache_dir=args.cache,
                   reduction_ratio=8, target_hw=64)
    tr, va, _ = get_patient_based_splits(ds, 0.7, 0.15, 0.15, seed=42)
    logging.info("surrogate: train %d / val %d samples, base=%d", len(tr), len(va), args.base)

    model = DoseSurrogate(base=args.base, use_norm=not args.no_norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr/50)
    logging.info("params=%d", sum(p.numel() for p in model.parameters()))

    rng = np.random.default_rng(0)
    for ep in range(1, args.epochs+1):
        model.train(); order = rng.permutation(tr); tot=0.0
        for k, i in enumerate(order):
            x, y = make_xy(ds[int(i)], device)
            if args.amp_aug > 0:
                a = float(rng.uniform(1.0 - args.amp_aug, 1.0 + args.amp_aug))
                x = x.clone(); x[:, 1:2] *= a; y = y * a   # scale ONLY the sino channel + dose
            opt.zero_grad(set_to_none=True)
            loss = F.l1_loss(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item()
            if k % 100 == 0:
                logging.info("ep%d %d/%d train_L1 %.5f", ep, k, len(order), loss.item())
        sched.step()
        vmae, vrel = evaluate(model, ds, va, device)
        logging.info("=== epoch %d | train_L1 %.5f | VAL dose MAE %.5f (%.1f%% of range) ===",
                     ep, tot/len(order), vmae, 100*vrel)
        torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "base": args.base,
                    "use_norm": not args.no_norm}, args.save)
    logging.info("saved %s", args.save)


if __name__ == "__main__":
    main()
