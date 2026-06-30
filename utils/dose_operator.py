#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/dose_operator.py

Differentiable simplified dose forward for the dose-consistency loss (lever B).

The berlingo already gives CT rotated to each gantry angle, so the ray axis W is
beam depth. We compute a beam's-eye-view primary dose (TERMA) per control point and
ANGULARLY ACCUMULATE it (un-rotate by -(90-theta), bin by couch z) into a 3D patient
dose. Everything is differentiable and LINEAR in the sinogram, so -- unlike the
learned CNN surrogate -- it is exactly amplitude-faithful by construction.

Validated in prototype_dose_forward.py / prototype_dose_accumulate.py:
amplitude 0.5x LOT -> 0.500 dose (vs surrogate 0.66%); accumulated dose corr 0.88
vs the real planned dose. See EXPERIMENTS.md "Lever B prototype".
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pydicom

SINO_TAG = (0x300D, 0x10A7)

# ---- Real Tomo 6MV CCC kernel ----
_KERNEL_CSV_DEFAULT = "/mnt/data/DoseCUDA/DoseCUDA/lookuptables/photons/Tomo/6MV/kernel.csv"
_kernel3d_cache: dict = {}


def build_ccc_kernel(dz_cm: float, dxy_cm: float,
                     rmax_cm: float = 5.0,
                     kernel_csv: str = _KERNEL_CSV_DEFAULT,
                     device=None) -> "torch.Tensor | None":
    """Build isotropic Tomo 6MV CCC scatter kernel [1,1,nz,nxy,nxy] from the CSV.

    Uses double-exponential collapsed-cone model; angle-weighted to give the
    isotropic (solid-angle-averaged) kernel h_iso(r). Cached by (dz, dxy, rmax).
    Build time ~0.1s (vectorised numpy). Returns None if the CSV is unavailable.
    """
    key = (round(dz_cm, 3), round(dxy_cm, 3), round(rmax_cm, 1))
    if key in _kernel3d_cache:
        t = _kernel3d_cache[key]
        return t.to(device) if device is not None else t

    try:
        cones = np.loadtxt(kernel_csv, delimiter=",", skiprows=1)
    except (FileNotFoundError, OSError):
        import warnings
        warnings.warn(f"CCC kernel CSV not found: {kernel_csv} — Dmax penalty disabled")
        return None

    th = np.deg2rad(cones[:, 0])
    w = np.sin(th); w_sum = w.sum()
    Am, am, Bm, bm = cones[:, 1], cones[:, 2], cones[:, 3], cones[:, 4]

    nz = int(rmax_cm / dz_cm)
    nxy = int(rmax_cm / dxy_cm)
    zc = np.arange(-nz, nz + 1, dtype=np.float64) * dz_cm
    yc = np.arange(-nxy, nxy + 1, dtype=np.float64) * dxy_cm
    ZZ, YY, XX = np.meshgrid(zc, yc, yc, indexing="ij")
    R = np.sqrt(ZZ ** 2 + YY ** 2 + XX ** 2)

    K = np.zeros_like(R)
    for i in range(len(w)):
        K += w[i] * (Am[i] * np.exp(-am[i] * R) + Bm[i] * np.exp(-bm[i] * R))
    K = (K / w_sum).astype(np.float32)

    t = torch.tensor(K, dtype=torch.float32)
    t = (t / t.sum())[None, None]           # [1, 1, 2nz+1, 2nxy+1, 2nxy+1]
    _kernel3d_cache[key] = t
    return t.to(device) if device is not None else t


def apply_ccc_kernel(vol: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply CCC scatter kernel to accumulated TERMA volume [Z,H,W] -> [Z,H,W].

    Differentiable w.r.t. vol. Run in fp32 (kernel is fp32).
    """
    pad = [s // 2 for s in kernel.shape[2:]]
    return F.conv3d(vol[None, None].float(), kernel.float(), padding=pad)[0, 0]


def ct_to_rho(ct_norm: torch.Tensor) -> torch.Tensor:
    """Normalized CT in [0,1] (HU = ct*4095 - 1024) -> relative electron density.
    air(HU=-1000)->0, water(0)->1, bone(~1000)->2."""
    hu = ct_norm * 4095.0 - 1024.0
    return torch.clamp((hu + 1000.0) / 1000.0, min=0.0)


def _gaussian_blur(x: torch.Tensor, sigma_h: float, sigma_w: float) -> torch.Tensor:
    """Separable Gaussian blur on [N,H,W] (differentiable) ~ lateral scatter."""
    def kern(sig):
        r = max(1, int(3 * sig))
        t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
        k = torch.exp(-(t ** 2) / (2 * sig * sig))
        return (k / k.sum()), r
    y = x.unsqueeze(1)
    if sigma_h > 0:
        kh, rh = kern(sigma_h); y = F.conv2d(y, kh.view(1, 1, -1, 1), padding=(rh, 0))
    if sigma_w > 0:
        kw, rw = kern(sigma_w); y = F.conv2d(y, kw.view(1, 1, 1, -1), padding=(0, rw))
    return y[:, 0]


def dose_forward(sino: torch.Tensor, ct: torch.Tensor, mu_water: float = 0.03,
                 entry: str = "w0", scatter_h: float = 0.0, scatter_w: float = 0.0) -> torch.Tensor:
    """sino [N,64], ct [N,H,W] (H=64 leaves) -> BEV primary dose [N,H,W].
    Differentiable, linear in sino."""
    rho = ct_to_rho(ct)
    mu = mu_water * rho
    if entry == "wmax":
        mu_path = torch.flip(mu, dims=[-1])
        depth = torch.cumsum(mu_path, dim=-1) - mu_path
        atten = torch.flip(torch.exp(-depth), dims=[-1])
    else:
        depth = torch.cumsum(mu, dim=-1) - mu
        atten = torch.exp(-depth)
    terma = sino.unsqueeze(-1) * mu * atten
    if scatter_h > 0 or scatter_w > 0:
        terma = _gaussian_blur(terma, scatter_h, scatter_w)
    return terma


def rotate_stack(imgs: torch.Tensor, beta_rad: torch.Tensor) -> torch.Tensor:
    """Rotate each [H,W] image about its center by beta_rad[i] (differentiable)."""
    n = imgs.shape[0]
    c, s = torch.cos(beta_rad), torch.sin(beta_rad)
    theta = torch.zeros(n, 2, 3, device=imgs.device, dtype=imgs.dtype)
    theta[:, 0, 0] = c; theta[:, 0, 1] = -s
    theta[:, 1, 0] = s; theta[:, 1, 1] = c
    grid = F.affine_grid(theta, (n, 1, imgs.shape[-2], imgs.shape[-1]), align_corners=False)
    return F.grid_sample(imgs.unsqueeze(1), grid, align_corners=False, padding_mode="zeros")[:, 0]


def build_zbin(tables: np.ndarray, n_z: int, device=None) -> torch.Tensor:
    """Couch positions -> z-bin index per CP (z_i = z_iso - table, bin by -table)."""
    z = -np.asarray(tables, dtype=np.float32)
    idx = ((z - z.min()) / (np.ptp(z) + 1e-6) * (n_z - 1)).round().astype(np.int64)
    return torch.as_tensor(idx, device=device)


def _blur3d(v: torch.Tensor, sz: float, sxy: float) -> torch.Tensor:
    """Separable 3D Gaussian on [Z,H,W]: sz along Z (beam field width in z), sxy along
    H,W (lateral scatter). Physical, not cosmetic: it injects the real jaw width that a
    per-CP single-slice accumulation omits -> the CORRECT dose null-space."""
    x = v[None, None]
    for ax, sig in [(2, sz), (3, sxy), (4, sxy)]:
        if sig <= 0:
            continue
        r = max(1, int(3 * sig))
        t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
        k = torch.exp(-(t ** 2) / (2 * sig * sig)); k = k / k.sum()
        shape = [1, 1, 1, 1, 1]; shape[ax] = -1
        pad = [0, 0, 0]; pad[ax - 2] = r
        x = F.conv3d(x, k.view(shape), padding=tuple(pad))
    return x[0, 0]


def accumulate(bev: torch.Tensor, alpha_deg: torch.Tensor, zbin: torch.Tensor,
               n_z: int, sign: float = 1.0, reduce: str = "sum",
               field_z: float = 0.0, scatter_xy: float = 0.0) -> torch.Tensor:
    """Un-rotate each CP BEV slice by -(90-theta) and bin into z -> [n_z,H,W].
    reduce='sum' for primary dose (superposition); 'mean' for the real TOTAL dose
    (already accumulated, so average the duplicate views per z-bin).
    field_z/scatter_xy add the jaw field width / lateral scatter to the PRIMARY side
    (do NOT apply to the real dose, which already has them). Raises operator fidelity
    vs real dose 0.88 -> ~0.94 (EXPERIMENTS 'Lever B / improve operator')."""
    beta = torch.deg2rad(sign * alpha_deg.to(bev.dtype))
    rot = rotate_stack(bev, beta)
    H, W = bev.shape[-2], bev.shape[-1]
    vol = torch.zeros(n_z, H, W, device=bev.device, dtype=bev.dtype)
    vol.index_add_(0, zbin, rot)
    if reduce == "mean":
        cnt = torch.zeros(n_z, H, W, device=bev.device, dtype=bev.dtype)
        cnt.index_add_(0, zbin, torch.ones_like(rot))
        vol = vol / cnt.clamp_min(1.0)
    if field_z > 0 or scatter_xy > 0:
        vol = _blur3d(vol, field_z, scatter_xy)
    return vol


def read_geometry(plan_path: str):
    """RTPLAN -> (gantry_angles[N], table_positions[N]) float32."""
    ds = pydicom.dcmread(plan_path, force=True)
    beam = ds[(0x300A, 0x00B0)][0]
    cps = beam[(0x300A, 0x0111)].value
    ang = np.array([float(getattr(cp, "GantryAngle", 0.0)) for cp in cps], np.float32)
    tab = np.array([float(getattr(cp, "TableTopLateralPosition", 0.0)) for cp in cps], np.float32)
    return ang, tab


def find_plan(data_path: str, patient_id: str, pareto_index) -> str:
    """Locate the RP*.dcm for a patient/pareto (FB or DIBH)."""
    root = Path(data_path)
    pid = str(patient_id)
    pats = [pid]
    if pid.endswith("_DIBH"):
        pats = [pid]
    for sub in ("Tomo_FB_copy", "Tomo_DIBH"):
        hits = list(root.glob(f"{pid}/**/{sub}/pareto_{int(pareto_index)}/RP*.dcm"))
        if hits:
            return str(hits[0])
    hits = list(root.glob(f"{pid}/**/pareto_{int(pareto_index)}/RP*.dcm"))
    if not hits:
        raise FileNotFoundError(f"no RP plan for {pid} pareto {pareto_index} under {root}")
    return str(hits[0])
