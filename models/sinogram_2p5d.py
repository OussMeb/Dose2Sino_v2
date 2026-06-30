#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/sinogram_2p5d.py

2.5D slice-wise sinogram predictor (architecture rethink).

Rationale (see CLAUDE.md): the 3D V-Net family is hard to optimize here (deeper/
wider/fancier variants stall or collapse; it overfits one sample only to
open_l1 ~0.11). The task is intrinsically per-control-point: berlingo slice i is
the CT/dose rotated to gantry angle i, and sinogram row i is the leaf opening at
control point i. So we process EACH slice [2, H, W] with a shared, well-
conditioned 2D CNN -> a [64] leaf row, and stack over N_CP. 2D CNNs train
easily, use far less memory (so they scale), and respect the structure.

Anisotropy: keep the leaf axis (H) near full resolution and downsample only the
ray axis (W); collapse the ray by MAX projection ("open the leaf if any voxel
along the ray is PTV"); reduce H -> 64 leaves by adaptive max pooling (sharp).

Interface matches DosePredictionAttentionInAgg:
    forward([B, 2, N_CP, H, W]) -> [B, 1, N_CP, 64, 1]  (raw logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Res2D(nn.Module):
    """Residual 2D block (Conv-IN-ReLU x2 + skip). 2D residuals are well-behaved
    (unlike the 3D residual on the huge ratio-3 volume, which stalled)."""

    def __init__(self, cin: int, cout: int, stride=(1, 1)):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.n1 = nn.InstanceNorm2d(cout, affine=True)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = nn.InstanceNorm2d(cout, affine=True)
        self.act = nn.ReLU(inplace=True)
        same = (cin == cout) and (tuple(stride) == (1, 1) or stride == 1)
        self.sc = nn.Identity() if same else nn.Conv2d(cin, cout, 1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = self.sc(x)
        y = self.act(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return self.act(y + idt)


class SpectralRefine1D(nn.Module):
    """Residual 1D Fourier (FNO) refinement along the control-point / gantry-angle
    axis of the [B, N_CP, 64] output plane.

    Motivation (measured, not assumed): GT LOT sinograms have real low-frequency
    structure along N_CP -- ~10x more power in the lowest angular frequencies than
    a CP-shuffled null (distributions do not overlap), ~50% of angular power in the
    lowest ~9% of bins, on a ~1300-long axis. The per-slice encoder predicts every
    control point INDEPENDENTLY and the conv `refine_head` only couples +/-1 CP, so
    neither sees this GLOBAL, slowly-varying angular coupling. An FNO layer mixes
    ALL control points in O(N log N) via a learned filter on the lowest `modes`
    frequencies (Li et al. 2020), giving the missing global angular receptive field.

    The 64 leaves are the channel dim (standard FNO channel mixing) so a structure
    that sweeps across leaves with gantry angle -- the sinusoid that names the
    sinogram -- can be tracked. Kept as a *residual* with a zero-init output proj,
    so it is exact identity at init (never disrupts the coarse prediction, same
    safety as the conv head) and the sparse HF tail stays in the real-domain path.
    """

    def __init__(self, n_leaves: int = 64, modes: int = 64, hidden: int = 64):
        super().__init__()
        self.modes = int(modes)
        self.lift = nn.Conv1d(n_leaves, hidden, 1)     # 64 leaves -> hidden channels
        self.wpoint = nn.Conv1d(hidden, hidden, 1)     # real-space residual path
        self.proj = nn.Conv1d(hidden, n_leaves, 1)     # back to 64 leaves (zero-init)
        # complex spectral weights over the lowest `modes` frequencies, mixing
        # channels per mode: [modes, hidden(out), hidden(in)].
        scale = hidden ** -0.5
        self.wr = nn.Parameter(torch.randn(self.modes, hidden, hidden) * scale)
        self.wi = nn.Parameter(torch.randn(self.modes, hidden, hidden) * scale)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, plane: torch.Tensor) -> torch.Tensor:  # [B, N_CP, 64]
        x = plane.transpose(1, 2)                 # [B, 64, N_CP]
        h = self.lift(x)                          # [B, H, N_CP]
        n = h.shape[-1]
        # cuFFT has no half support on pre-SM_53 GPUs (the M6000 is SM_52), so run
        # the spectral path in float32 even under AMP autocast.
        with torch.autocast(device_type=h.device.type, enabled=False):
            hf32 = h.float()
            Hf = torch.fft.rfft(hf32, dim=-1)     # [B, H, F]  (F = n//2 + 1)
            m = min(self.modes, Hf.shape[-1])
            # per low mode k: out[b,o,k] = sum_i Hf[b,i,k] * w[k,o,i], complex.
            # einsum has no complex support on this torch build, so do the complex
            # product via real/imag parts: (Hr+iHi)(wr+iwi).
            Hr, Hi = Hf[..., :m].real, Hf[..., :m].imag           # [B, H, m]
            wr, wi = self.wr[:m], self.wi[:m]                     # [m, H, H]
            out_r = (torch.einsum("bik,koi->bok", Hr, wr)
                     - torch.einsum("bik,koi->bok", Hi, wi))
            out_i = (torch.einsum("bik,koi->bok", Hr, wi)
                     + torch.einsum("bik,koi->bok", Hi, wr))
            outf = torch.zeros_like(Hf)
            outf[..., :m] = torch.complex(out_r, out_i)
            spec = torch.fft.irfft(outf, n=n, dim=-1)             # [B, H, N_CP]
        spec = spec.to(h.dtype)
        h = spec + self.wpoint(h)
        out = self.proj(torch.relu(h))            # [B, 64, N_CP]  (zero at init)
        return plane + out.transpose(1, 2)        # residual on the output plane


class Sinogram2p5D(nn.Module):
    """Shared 2D encoder over control-point slices -> 64 leaf logits per slice."""

    def __init__(self, in_channels: int = 2, base: int = 24, n_leaves: int = 64,
                 slice_chunk: int = 0, reduce_h: bool = True,
                 refine: bool = False, refine_channels: int = 32,
                 refine_mode: str = "conv", fno_modes: int = 64):
        super().__init__()
        self.n_leaves = int(n_leaves)
        self.refine = bool(refine)
        # slice_chunk>0 processes control points in chunks (memory control); the
        # graph is kept (cat), so it only caps peak activation if used with
        # checkpointing -- here it mainly bounds intermediate buffers.
        self.slice_chunk = int(slice_chunk)

        # reduce_h controls whether the stem downsamples the LEAF axis (H).
        #   reduce_h=True  (ratio=3, H~170): stem H/2 -> 85, then AdaptiveMaxPool
        #     85->64 (overlapping windows -> adjacent leaves correlated; this is
        #     the ~0.11 floor source).
        #   reduce_h=False (ratio=8, H~64): keep H, so the readout pool 64->64 is
        #     ~1:1 (one leaf <-> one H row, no overlap -> independent leaves).
        # In both cases ONLY the ray axis W is downsampled by the encoder.
        h_stride = 2 if reduce_h else 1
        self.stem = Res2D(in_channels, base, stride=(h_stride, 2))   # H/h_stride, W/2
        self.e1 = Res2D(base, base * 2, stride=(1, 2))          # W/4
        self.e2 = Res2D(base * 2, base * 4, stride=(1, 2))      # W/8
        self.e3 = Res2D(base * 4, base * 4, stride=(1, 2))      # W/16
        self.head = nn.Sequential(
            Res2D(base * 4, base * 2),
            nn.Conv2d(base * 2, base, 1),
            nn.ReLU(inplace=True),
        )
        # Leaf readout = H-ALIGNED max-pool (keeps each leaf tied to its H region
        # -> localization) + PER-LEAF independent weights (each leaf has its own
        # linear over the base channels -> breaks the shared-readout correlation
        # that floored every model at ~0.11). Bounded by the pooled features, so
        # it does not collapse to all-zero like a global Linear, and unlike a
        # global max-pool it keeps leaf localization.
        self.to_leaf = nn.AdaptiveMaxPool2d((self.n_leaves, 1))
        self.leaf_w = nn.Parameter(torch.randn(self.n_leaves, base) * (base ** -0.5))
        self.leaf_b = nn.Parameter(torch.zeros(self.n_leaves))

        # Optional 2D refinement head on the [N_CP, 64] OUTPUT plane. The per-slice
        # encoder above predicts every (CP, leaf) value INDEPENDENTLY -> it cannot
        # use the strong local structure of a real sinogram (smoothness along the
        # gantry/CP axis, correlation between neighbouring leaves). This small 2D
        # CNN runs over the coarse logits as a [B,1,N_CP,64] image and adds a
        # learned residual, so each value is refined from its CP x leaf neighbours.
        # Cheap: the plane is ~N_CP*64, negligible vs the per-slice encoder.
        # The last conv is zero-initialised -> identity at start (residual=0), so
        # it never disrupts a good coarse prediction early in training.
        # refine_mode selects how the [N_CP, 64] output plane is coupled:
        #   "conv" -> local 2D CNN (sees +/-1-2 CP/leaf; the original head).
        #   "fno"  -> 1D Fourier op along N_CP (GLOBAL angular reach; see
        #             SpectralRefine1D). "both" stacks conv after fno.
        self.refine_mode = str(refine_mode)
        if self.refine:
            if self.refine_mode in ("conv", "both"):
                c = int(refine_channels)
                self.refine_head = nn.Sequential(
                    nn.Conv2d(1, c, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(c, 1, 3, padding=1),
                )
                nn.init.zeros_(self.refine_head[-1].weight)
                nn.init.zeros_(self.refine_head[-1].bias)
            if self.refine_mode in ("fno", "both"):
                self.fno_head = SpectralRefine1D(
                    n_leaves=self.n_leaves, modes=int(fno_modes),
                    hidden=int(refine_channels),
                )

    def _encode_slices(self, s: torch.Tensor) -> torch.Tensor:
        # s: [N, in_channels, H, W] -> logits [N, n_leaves]
        f = self.stem(s)
        f = self.e3(self.e2(self.e1(f)))
        f = self.head(f)                       # [N, base, h, w]
        f = self.to_leaf(f)[:, :, :, 0]        # [N, base, n_leaves]  (H-aligned)
        # per-leaf independent linear: out[n,j] = sum_c f[n,c,j]*W[j,c] + b[j]
        return torch.einsum("ncj,jc->nj", f, self.leaf_w) + self.leaf_b  # [N, n_leaves]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        s = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)  # [B*N_CP, 2, H, W]

        if self.slice_chunk and s.shape[0] > self.slice_chunk:
            # Process control-point slices in chunks with gradient checkpointing,
            # so PEAK memory is bounded by one chunk regardless of N_CP (patients
            # vary; the biggest ones OOM otherwise). Checkpointing recomputes the
            # forward during backward (~1.3x compute) instead of storing all
            # activations.
            outs = []
            for i in range(0, s.shape[0], self.slice_chunk):
                chunk = s[i:i + self.slice_chunk]
                if self.training:
                    outs.append(checkpoint(self._encode_slices, chunk, use_reentrant=False))
                else:
                    outs.append(self._encode_slices(chunk))
            logits = torch.cat(outs, dim=0)
        else:
            logits = self._encode_slices(s)

        logits = logits.view(b, d, self.n_leaves)              # [B, N_CP, 64] coarse

        if self.refine:
            if self.refine_mode in ("fno", "both"):
                # GLOBAL angular coupling along N_CP via a 1D Fourier op (residual).
                logits = self.fno_head(logits)                 # [B, N_CP, 64]
            if self.refine_mode in ("conv", "both"):
                # Local CP x leaf coupling on the output plane via a residual 2D CNN.
                r = logits.unsqueeze(1)                        # [B, 1, N_CP, 64]
                logits = logits + self.refine_head(r).squeeze(1)

        return logits.unsqueeze(1).unsqueeze(-1)  # [B, 1, N_CP, n_leaves, 1]


class DosePrediction2p5D(nn.Module):
    """Drop-in wrapper matching DosePredictionAttentionInAgg's interface."""

    def __init__(self, base_filters: int = 24, in_channel: int = 2,
                 n_leaves: int = 64, slice_chunk: int = 0, reduce_h: bool = True,
                 refine: bool = False, refine_channels: int = 32,
                 refine_mode: str = "conv", fno_modes: int = 64, **_ignored):
        super().__init__()
        self.net = Sinogram2p5D(
            in_channels=in_channel, base=base_filters,
            n_leaves=n_leaves, slice_chunk=slice_chunk, reduce_h=reduce_h,
            refine=refine, refine_channels=refine_channels,
            refine_mode=refine_mode, fno_modes=fno_modes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
