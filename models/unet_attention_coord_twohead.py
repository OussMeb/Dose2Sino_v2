#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/unet_attention_coord_twohead.py

Phase 1 candidate:
- 5-channel input:
    0 CT berlingo
    1 dose berlingo
    2 normalized control-point index
    3 normalized leaf index
    4 normalized table position
- InstanceNorm3d
- lightweight control-point attention
- learned Conv3D aggregation over detector-x
- two-head sinogram prediction:
    open_logits  -> open/closed probability
    value_logits -> opening amplitude
    pred_prob = sigmoid(open_logits) * sigmoid(value_logits)

Input:
    [B, 5, N_CP, 64, 64]

Output dict:
    {
        "open_logits":  [B, 1, N_CP, 64, 1],
        "value_logits": [B, 1, N_CP, 64, 1],
        "pred_prob":    [B, 1, N_CP, 64, 1],
    }
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two Conv3d layers with InstanceNorm3d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x


class DownSampling(nn.Module):
    """Preserve control-point axis, downsample detector dimensions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 2, 2),
            stride=(1, 2, 2),
            padding=(1, 0, 0),
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class UpSampling(nn.Module):
    """Preserve control-point axis, upsample detector dimensions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.up(x)))


class ControlPointAttention3D(nn.Module):
    """
    Lightweight attention along N_CP.

    Starts neutral because gamma is initialized at 0.
    """

    def __init__(self, channels: int, kernel_size: int = 15, reduction: int = 4):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("attention_kernel_size must be odd.")

        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=hidden, out_channels=channels, kernel_size=kernel_size, padding=kernel_size // 2),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(3, 4))          # [B, C, N_CP]
        gate = torch.tanh(self.net(pooled))  # [B, C, N_CP]
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + self.gamma * gate)


class VNetAttentionCoordTwoHead(nn.Module):
    """Shared VNet encoder-decoder with two sinogram heads."""

    def __init__(
        self,
        in_channels: int = 5,
        base_filters: int = 8,
        attention_kernel_size: int = 15,
        detector_width: int = 64,
    ):
        super().__init__()
        self.detector_width = int(detector_width)

        self.enc1 = ConvBlock(in_channels, base_filters)
        self.attn1 = ControlPointAttention3D(base_filters, attention_kernel_size)
        self.down1 = DownSampling(base_filters, base_filters * 2)

        self.enc2 = ConvBlock(base_filters * 2, base_filters * 2)
        self.attn2 = ControlPointAttention3D(base_filters * 2, attention_kernel_size)
        self.down2 = DownSampling(base_filters * 2, base_filters * 4)

        self.enc3 = ConvBlock(base_filters * 4, base_filters * 4)
        self.attn3 = ControlPointAttention3D(base_filters * 4, attention_kernel_size)
        self.down3 = DownSampling(base_filters * 4, base_filters * 8)

        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 8)
        self.attn_bottleneck = ControlPointAttention3D(base_filters * 8, attention_kernel_size)

        self.up3 = UpSampling(base_filters * 8, base_filters * 4)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4)
        self.attn_dec3 = ControlPointAttention3D(base_filters * 4, attention_kernel_size)

        self.up2 = UpSampling(base_filters * 4, base_filters * 2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)
        self.attn_dec2 = ControlPointAttention3D(base_filters * 2, attention_kernel_size)

        self.up1 = UpSampling(base_filters * 2, base_filters)
        self.dec1 = ConvBlock(base_filters * 2, base_filters)
        self.attn_dec1 = ControlPointAttention3D(base_filters, attention_kernel_size)

        self.open_head = nn.Conv3d(base_filters, 1, kernel_size=1)
        self.value_head = nn.Conv3d(base_filters, 1, kernel_size=1)

        self.open_aggregate = nn.Conv3d(1, 1, kernel_size=(1, 1, self.detector_width))
        self.value_aggregate = nn.Conv3d(1, 1, kernel_size=(1, 1, self.detector_width))

    def _check_width(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.detector_width:
            raise ValueError(
                "Detector width mismatch before aggregation: "
                f"got W={x.shape[-1]}, expected W={self.detector_width}. "
                "Use REDUCTION_RATIO=8 or cached 64x64 tensors."
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x1 = self.attn1(self.enc1(x))
        x2 = self.attn2(self.enc2(self.down1(x1)))
        x3 = self.attn3(self.enc3(self.down2(x2)))

        b = self.attn_bottleneck(self.bottleneck(self.down3(x3)))

        d3 = self.up3(b)
        if d3.size()[2:] != x3.size()[2:]:
            d3 = F.interpolate(d3, size=x3.size()[2:], mode="trilinear", align_corners=False)
        d3 = self.attn_dec3(self.dec3(torch.cat([d3, x3], dim=1)))
        del x3

        d2 = self.up2(d3)
        if d2.size()[2:] != x2.size()[2:]:
            d2 = F.interpolate(d2, size=x2.size()[2:], mode="trilinear", align_corners=False)
        d2 = self.attn_dec2(self.dec2(torch.cat([d2, x2], dim=1)))
        del x2

        d1 = self.up1(d2)
        if d1.size()[2:] != x1.size()[2:]:
            d1 = F.interpolate(d1, size=x1.size()[2:], mode="trilinear", align_corners=False)
        d1 = self.attn_dec1(self.dec1(torch.cat([d1, x1], dim=1)))
        del x1

        self._check_width(d1)

        open_logits_3d = self.open_head(d1)
        value_logits_3d = self.value_head(d1)

        open_logits = self.open_aggregate(open_logits_3d)
        value_logits = self.value_aggregate(value_logits_3d)

        pred_prob = torch.sigmoid(open_logits) * torch.sigmoid(value_logits)

        return {
            "open_logits": open_logits,
            "value_logits": value_logits,
            "pred_prob": pred_prob,
        }


class DosePredictionAttentionCoordTwoHead(nn.Module):
    """CT+dose+coordinate channels -> two-head LOT sinogram prediction."""

    def __init__(
        self,
        base_filters: int = 8,
        in_channel: int = 5,
        attention_kernel_size: int = 15,
        detector_width: int = 64,
    ):
        super().__init__()
        self.vnet = VNetAttentionCoordTwoHead(
            in_channels=in_channel,
            base_filters=base_filters,
            attention_kernel_size=attention_kernel_size,
            detector_width=detector_width,
        )

    def forward(self, input: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.vnet(input)
