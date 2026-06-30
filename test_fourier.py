#!/usr/bin/env python3
"""Unit tests for the Fourier additions: SpectralRefine1D (FNO) + AngularSpectralLoss."""
import sys
sys.path.insert(0, "/mnt/data/sinogram_generator")
import torch
from models.sinogram_2p5d import SpectralRefine1D, DosePrediction2p5D
from utils.losses import AngularSpectralLoss

torch.manual_seed(0)
ok = True

def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

# --- SpectralRefine1D ---
B, N, L = 2, 300, 64
fno = SpectralRefine1D(n_leaves=L, modes=64, hidden=32)
plane = torch.randn(B, N, L, requires_grad=True)
out = fno(plane)
check("fno output shape preserved", out.shape == plane.shape)
# identity at init (proj zero-init -> residual zero)
check("fno is identity at init", torch.allclose(out, plane, atol=1e-6))

# after perturbing proj, it must produce a non-trivial GLOBAL change:
# a single-CP spike in the input should affect FAR-AWAY CPs (global receptive field),
# which a local conv cannot do.
with torch.no_grad():
    fno.proj.weight.normal_(0, 0.1)
    fno.proj.bias.zero_()
spike = torch.zeros(1, N, L)
spike[0, N // 2, :] = 1.0
resp = (fno(spike) - spike)[0].abs().mean(dim=1)   # [N] response magnitude per CP
far = resp[:10].mean().item()                       # CPs far from the spike
check("fno has global reach (far CPs respond)", far > 1e-6)

# gradient flows to spectral weights once proj != 0
loss = (fno(plane) ** 2).mean()
loss.backward()
gwr = fno.wr.grad
check("fno spectral weights get gradient", gwr is not None and gwr.abs().sum() > 0)

# --- AngularSpectralLoss ---
asl = AngularSpectralLoss(modes=32, weight=1.0)
tgt = torch.rand(B, N, L)
# perfect match -> ~0 (logits = logit(target))
logit_perfect = torch.logit(tgt.clamp(1e-4, 1 - 1e-4))
lp = asl(logit_perfect, tgt).item()
check("spectral loss ~0 when pred matches target", lp < 1e-3)
# mismatched -> strictly positive
lm = asl(torch.zeros(B, N, L), tgt).item()
check("spectral loss > 0 on mismatch", lm > lp)
# differentiable (realistic non-constant prediction)
z = (torch.randn(B, N, L) * 0.5).requires_grad_(True)
asl(z, tgt).backward()
check("spectral loss differentiable", z.grad is not None and z.grad.abs().sum() > 0
      and torch.isfinite(z.grad).all())

# --- end-to-end model with fno refine, real berlingo-ish shape ---
for mode in ["conv", "fno", "both"]:
    m = DosePrediction2p5D(base_filters=8, refine=True, refine_mode=mode,
                           refine_channels=16, fno_modes=32, reduce_h=False)
    x = torch.randn(1, 2, 40, 64, 64)          # [B,2,N_CP,H,W]
    y = m(x)
    check(f"model refine_mode={mode} forward shape", tuple(y.shape) == (1, 1, 40, 64, 1))
    y.sum().backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    check(f"model refine_mode={mode} backward has grad", has_grad)

print("\n=== ALL PASS ===" if ok else "\n=== FAILURES ===")
sys.exit(0 if ok else 1)
