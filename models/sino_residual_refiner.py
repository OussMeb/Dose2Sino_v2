#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/sino_residual_refiner.py

Small bounded residual refiner for LOT sinograms.

Goal:
    Preserve the supervised generator global prediction and learn only a local correction.

Input:
    condition_maps: [B, C_cond, N_CP, 64]
    baseline_sino:  [B, 1,      N_CP, 64]

Output:
    {
        "delta":   bounded correction in [-delta_scale, +delta_scale],
        "refined": clamp(baseline_sino + delta, 0, 1),
        "raw_delta": unbounded raw correction
    }
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock2D(nn.Module):
    """Small residual Conv2d block with InstanceNorm2d."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DownBlock2D(nn.Module):
    """Downsample both CP and leaf axes."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualConvBlock2D(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock2D(nn.Module):
    """Upsample to skip resolution then fuse."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualConvBlock2D(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class SinoResidualRefiner2D(nn.Module):
    """
    Bounded residual U-Net refiner.

    The final convolution is zero-initialized, so the initial output is exactly:
        refined = baseline_sino
    """

    def __init__(
        self,
        condition_channels: int = 2,
        base_channels: int = 32,
        delta_scale: float = 0.10,
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)
        in_channels = int(condition_channels) + 1

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualConvBlock2D(base_channels),
        )

        self.down1 = DownBlock2D(base_channels, base_channels * 2)
        self.down2 = DownBlock2D(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock2D(base_channels * 4, base_channels * 4)

        self.bottleneck = nn.Sequential(
            ResidualConvBlock2D(base_channels * 4),
            ResidualConvBlock2D(base_channels * 4),
        )

        self.up2 = UpBlock2D(base_channels * 4, base_channels * 4, base_channels * 4)
        self.up1 = UpBlock2D(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up0 = UpBlock2D(base_channels * 2, base_channels, base_channels)

        self.out = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, condition_maps: torch.Tensor, baseline_sino: torch.Tensor) -> dict[str, torch.Tensor]:
        if condition_maps.ndim != 4:
            raise ValueError(f"condition_maps must be [B,C,N_CP,64], got {tuple(condition_maps.shape)}")
        if baseline_sino.ndim != 4:
            raise ValueError(f"baseline_sino must be [B,1,N_CP,64], got {tuple(baseline_sino.shape)}")
        if baseline_sino.shape[1] != 1:
            raise ValueError(f"baseline_sino must have one channel, got {baseline_sino.shape[1]}")
        if condition_maps.shape[0] != baseline_sino.shape[0]:
            raise ValueError("Batch size mismatch between condition_maps and baseline_sino.")
        if condition_maps.shape[-2:] != baseline_sino.shape[-2:]:
            raise ValueError(
                f"Spatial mismatch: condition={tuple(condition_maps.shape)}, "
                f"baseline={tuple(baseline_sino.shape)}"
            )

        x0 = self.stem(torch.cat([condition_maps, baseline_sino], dim=1))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        b = self.bottleneck(x3)

        y2 = self.up2(b, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)

        raw_delta = self.out(y0)
        delta = self.delta_scale * torch.tanh(raw_delta)
        refined = torch.clamp(baseline_sino + delta, 0.0, 1.0)

        return {
            "raw_delta": raw_delta,
            "delta": delta,
            "refined": refined,
        }
