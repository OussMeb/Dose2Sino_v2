#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/augmentation.py

Augmentation for the LOT-sinogram task, applied to an *already-cached berlingo*
sample dict:

    sample['input']  : [2, N_CP, H, W]  (ch0 = CT, ch1 = dose)
    sample['target'] : [N_CP, 64]       (leaf-open fraction per control point)

IMPORTANT — geometry constraints. Each control-point slice of the berlingo is
already rotated to its gantry angle (`apply_tomo_transform_to_stack`), and the
target is the projection of the PTV along the ray (W) axis. So only the symmetry
transforms that the projection actually respects are valid here:

  * Leaf-axis (H) flip  -> must also flip the 64-leaf axis of the target. This is
    the top<->bottom symmetry of the MLC leaf bank. Geometrically exact.
  * Ray-axis (W) flip   -> INPUT ONLY, target unchanged. The model collapses W
    with a max/LogSumExp projection ("open if PTV anywhere along the ray"), which
    is invariant to ray order; flipping W teaches exactly that inductive bias.
  * CT intensity jitter -> small multiplicative scale + additive noise on the CT
    channel only. Does not touch geometry. Dose drives the LOT more directly, so
    it is left untouched to avoid changing the target semantics.

NOT included (would corrupt input<->target alignment on a cached berlingo):
in-plane rotation, in-plane zoom, control-point roll — each slice's rotation is
tied to an absolute gantry angle, so re-rotating/rolling desyncs it from the
sinogram.
"""

from __future__ import annotations

import torch


class RTDataAugmentation:
    """Geometry-safe augmentation for cached berlingo + sinogram samples."""

    def __init__(
        self,
        flip_leaf_prob: float = 0.5,
        flip_ray_prob: float = 0.5,
        ct_jitter_prob: float = 0.3,
        ct_jitter_std: float = 0.05,
        ct_scale_range: float = 0.05,
    ):
        self.flip_leaf_prob = flip_leaf_prob
        self.flip_ray_prob = flip_ray_prob
        self.ct_jitter_prob = ct_jitter_prob
        self.ct_jitter_std = ct_jitter_std
        self.ct_scale_range = ct_scale_range

    def __call__(self, sample: dict) -> dict:
        x = sample["input"]       # [2, N_CP, H, W]
        y = sample["target"]      # [N_CP, 64]

        # Leaf-axis (H) flip: reverse H of the input AND the 64-leaf axis of target.
        if torch.rand(()) < self.flip_leaf_prob:
            x = torch.flip(x, dims=[2])   # H is axis 2 of [2, N_CP, H, W]
            y = torch.flip(y, dims=[1])   # leaf axis is axis 1 of [N_CP, 64]

        # Ray-axis (W) flip: input only; projection along W is order-invariant.
        if torch.rand(()) < self.flip_ray_prob:
            x = torch.flip(x, dims=[3])   # W is axis 3

        # CT intensity jitter on channel 0 only.
        if torch.rand(()) < self.ct_jitter_prob:
            x = x.clone()
            scale = 1.0 + (torch.rand(()) * 2 - 1) * self.ct_scale_range
            noise = torch.randn_like(x[0]) * self.ct_jitter_std
            x[0] = x[0] * scale + noise

        sample = dict(sample)
        sample["input"] = x.contiguous()
        sample["target"] = y.contiguous()
        return sample
