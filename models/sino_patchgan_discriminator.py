#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/sino_patchgan_discriminator.py

2D conditional PatchGAN discriminator for LOT sinograms.

Input:
    condition_maps: [B, C_cond, N_CP, 64]
    sinogram:       [B, 1,      N_CP, 64]

D input:
    concat(condition_maps, sinogram) -> [B, C_cond + 1, N_CP, 64]

Output:
    patch logits: [B, 1, h, w]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SinoPatchGANDiscriminator(nn.Module):
    """Lightweight 2D PatchGAN for conditional sinogram realism."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        max_channels: int = 256,
    ):
        super().__init__()

        def block(
            in_ch: int,
            out_ch: int,
            stride: tuple[int, int],
            normalize: bool = True,
        ) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=not normalize,
                )
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        c1 = base_channels
        c2 = min(base_channels * 2, max_channels)
        c3 = min(base_channels * 4, max_channels)
        c4 = min(base_channels * 8, max_channels)

        self.net = nn.Sequential(
            *block(in_channels, c1, stride=(2, 2), normalize=False),
            *block(c1, c2, stride=(2, 2), normalize=True),
            *block(c2, c3, stride=(2, 2), normalize=True),
            *block(c3, c4, stride=(1, 1), normalize=True),
            nn.Conv2d(c4, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, condition_maps: torch.Tensor, sinogram: torch.Tensor) -> torch.Tensor:
        if condition_maps.ndim != 4:
            raise ValueError(f"condition_maps must be [B,C,N_CP,64], got {tuple(condition_maps.shape)}")
        if sinogram.ndim != 4:
            raise ValueError(f"sinogram must be [B,1,N_CP,64], got {tuple(sinogram.shape)}")
        if condition_maps.shape[0] != sinogram.shape[0]:
            raise ValueError("Batch size mismatch between condition_maps and sinogram.")
        if condition_maps.shape[2:] != sinogram.shape[2:]:
            raise ValueError(
                f"Spatial mismatch: condition={tuple(condition_maps.shape)}, "
                f"sinogram={tuple(sinogram.shape)}"
            )

        return self.net(torch.cat([condition_maps, sinogram], dim=1))
