#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/unet_attention_in_agg.py

VNet-style sinogram predictor with:
- InstanceNorm3d for batch-size-1 stability
- lightweight control-point attention
- learned Conv3D aggregation over detector-x

Input:
    [B, 2, N_CP, 64, 64]

Output:
    [B, 1, N_CP, 64, 1] raw logits

The output is intentionally raw logits because SinogramLoss uses
BCEWithLogitsLoss. Apply torch.sigmoid(output) for visualization/metrics.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two Conv3d layers with InstanceNorm3d and ReLU.

    NB: a residual/skip variant was tried to improve optimization conditioning.
    It let a big (bf=32) model train at ratio-8, but it STALLED the bf=8 ratio-3
    deployment config (loss stuck at init) -- residual + InstanceNorm on the
    large ratio-3 volume gives a bad landscape. So the plain block is kept.
    """

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
    """Anisotropic downsampling: preserve N_CP, downsample detector dimensions."""

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
    """Anisotropic upsampling: preserve N_CP, upsample detector dimensions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.upsample(x)))


class ControlPointAttention3D(nn.Module):
    """
    Lightweight attention along the control-point axis.

    It pools detector dimensions, computes a per-channel/per-control-point gate,
    then applies it as a residual modulation. The trainable gamma starts at zero,
    so the block is exactly neutral at initialization.
    """

    def __init__(self, channels: int, kernel_size: int = 15, reduction: int = 4):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("attention kernel_size must be odd.")

        hidden = max(channels // reduction, 4)
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=kernel_size, padding=padding),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N_CP, H, W]
        pooled = x.mean(dim=(3, 4))          # [B, C, N_CP]
        gate = torch.tanh(self.net(pooled))  # [-1, 1], [B, C, N_CP]
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + self.gamma * gate)


class ConvBlock2D(nn.Module):
    """Two Conv2d + ReLU for the [N_CP, leaf] refinement plane.

    NO normalization: InstanceNorm/BatchNorm force each channel toward zero-mean
    unit-variance over the plane, which fights the ~82%-zero (sparse) sinogram
    output and causes the model to over-paint. A shallow plain conv block trains
    fine here and can represent a near-constant (mostly closed) map.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Refine2DUNet(nn.Module):
    """
    Small 2D U-Net operating on the projected sinogram plane [B, C, N_CP, leaf].

    This is the "refine" half of project-then-2D-refine: the 3D encoder builds
    ray features, a projection collapses the beam axis, and THIS net models the
    2D structure of the sinogram (banded leaf patterns over control points) that
    the single-shot reduction heads could not represent.
    """

    def __init__(self, in_channels: int, base: int = 16, out_channels: int = 1):
        super().__init__()
        self.enc1 = ConvBlock2D(in_channels, base)
        self.down1 = nn.Conv2d(base, base * 2, 2, stride=2)
        self.enc2 = ConvBlock2D(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 2, stride=2)
        self.bottleneck = ConvBlock2D(base * 4, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock2D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock2D(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)
        # NB: do NOT init the bias strongly negative. A "mostly-closed" init
        # (sigmoid ~0.02) saturates the sigmoid where d(sigmoid)/d(logit) ~ 0.02,
        # starving the gradient and stalling optimization. Start near 0 (sigmoid
        # 0.5, max gradient) and let the FP penalty pull closed leaves down.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        b = self.bottleneck(self.down2(x2))

        u2 = self.up2(b)
        if u2.shape[-2:] != x2.shape[-2:]:
            u2 = F.interpolate(u2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-2:] != x1.shape[-2:]:
            u1 = F.interpolate(u1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))
        return self.out(d1)


class VnetAttentionInAgg(nn.Module):
    """
    VNet encoder-decoder with InstanceNorm, CP attention, and learned aggregation.

    Output is raw logits with shape [B, 1, N_CP, 64, 1].
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_filters: int = 16,
        attention_kernel_size: int = 15,
        detector_width: int = 64,
        n_leaves: int = 64,
    ):
        super().__init__()
        self.detector_width = int(detector_width)
        self.n_leaves = int(n_leaves)

        self.encoder_block1 = ConvBlock(in_channels, base_filters)
        self.attn1 = ControlPointAttention3D(base_filters, kernel_size=attention_kernel_size)
        self.down1 = DownSampling(base_filters, base_filters * 2)

        self.encoder_block2 = ConvBlock(base_filters * 2, base_filters * 2)
        self.attn2 = ControlPointAttention3D(base_filters * 2, kernel_size=attention_kernel_size)
        self.down2 = DownSampling(base_filters * 2, base_filters * 4)

        self.encoder_block3 = ConvBlock(base_filters * 4, base_filters * 4)
        self.attn3 = ControlPointAttention3D(base_filters * 4, kernel_size=attention_kernel_size)
        self.down3 = DownSampling(base_filters * 4, base_filters * 8)

        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 8)
        self.attn_bottleneck = ControlPointAttention3D(
            base_filters * 8,
            kernel_size=attention_kernel_size,
        )

        self.up3 = UpSampling(base_filters * 8, base_filters * 4)
        self.decoder_block3 = ConvBlock(base_filters * 8, base_filters * 4)
        self.attn_dec3 = ControlPointAttention3D(base_filters * 4, kernel_size=attention_kernel_size)

        self.up2 = UpSampling(base_filters * 4, base_filters * 2)
        self.decoder_block2 = ConvBlock(base_filters * 4, base_filters * 2)
        self.attn_dec2 = ControlPointAttention3D(base_filters * 2, kernel_size=attention_kernel_size)

        self.up1 = UpSampling(base_filters * 2, base_filters)
        self.decoder_block1 = ConvBlock(base_filters * 2, base_filters)
        self.attn_dec1 = ControlPointAttention3D(base_filters, kernel_size=attention_kernel_size)

        # Learned reduction of the leaf axis (H) onto the fixed MLC leaf count.
        # A strided conv coarsens H locally, then AdaptiveMaxPool snaps it to
        # exactly n_leaves. Max (not average) preserves the sharp, sparse
        # open-leaf lines that trilinear interpolation used to smear.
        # Operates on the feature-rich map (base_filters channels) before output.
        # Strided coarsening of the leaf axis (H), then AdaptiveMaxPool to
        # exactly n_leaves. (Assumes H >= 2*n_leaves, i.e. REDUCTION_RATIO<=~5;
        # fine for the ratio-3 deployment.)
        self.leaf_reduce = nn.Sequential(
            nn.Conv3d(base_filters, base_filters, (1, 5, 1), stride=(1, 2, 1), padding=(0, 2, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_filters, base_filters, (1, 3, 1), padding=(0, 1, 0)),
            nn.ReLU(inplace=True),
        )
        # MAX pooling on the leaf axis: preserves the sharp open-leaf peaks.
        # Empirically max reaches a much lower overfit floor than average
        # (open_l1 ~0.11 vs ~0.21) -- avg blurs adjacent leaves so the 64
        # distinct leaf values cannot be sharply represented.
        self.leaf_pool = nn.AdaptiveMaxPool3d((None, self.n_leaves, None))

        # Simple, robust head (the only one that trains reliably). The 2D-refine
        # and soft-max-projection heads all collapsed/stalled; this linear
        # aggregate reaches the best floor.
        self.output = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        # Ray axis (W) -> detector_width, then a learned linear conv collapses it
        # to 1 (the projection along the beam).
        self.ray_pool = nn.AdaptiveAvgPool3d((None, None, self.detector_width))
        self.aggregate = nn.Conv3d(
            out_channels, out_channels, kernel_size=(1, 1, self.detector_width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.attn1(self.encoder_block1(x))
        x2 = self.attn2(self.encoder_block2(self.down1(x1)))
        x3 = self.attn3(self.encoder_block3(self.down2(x2)))

        bottleneck = self.attn_bottleneck(self.bottleneck(self.down3(x3)))

        d3 = self.up3(bottleneck)
        if d3.size()[2:] != x3.size()[2:]:
            d3 = F.interpolate(d3, size=x3.size()[2:], mode="trilinear", align_corners=False)
        d3 = self.attn_dec3(self.decoder_block3(torch.cat([d3, x3], dim=1)))
        del x3

        d2 = self.up2(d3)
        if d2.size()[2:] != x2.size()[2:]:
            d2 = F.interpolate(d2, size=x2.size()[2:], mode="trilinear", align_corners=False)
        d2 = self.attn_dec2(self.decoder_block2(torch.cat([d2, x2], dim=1)))
        del x2

        d1 = self.up1(d2)
        if d1.size()[2:] != x1.size()[2:]:
            d1 = F.interpolate(d1, size=x1.size()[2:], mode="trilinear", align_corners=False)
        d1 = self.attn_dec1(self.decoder_block1(torch.cat([d1, x1], dim=1)))
        del x1

        # Leaf axis (H) -> n_leaves via the learned strided-conv head.
        # Assumes H >= n_leaves, which holds for REDUCTION_RATIO <= ~5.
        d1 = self.leaf_pool(self.leaf_reduce(d1))   # [B, F, N_CP, n_leaves, W]

        output_3d = F.relu(self.output(d1))
        output_3d = self.ray_pool(output_3d)        # W -> detector_width
        return self.aggregate(output_3d)            # -> [B, out, N_CP, n_leaves, 1]


class DosePredictionAttentionInAgg(nn.Module):
    """CT+dose berlingo -> LOT sinogram logits."""

    def __init__(
        self,
        base_filters: int = 16,
        in_channel: int = 2,
        attention_kernel_size: int = 15,
        detector_width: int = 64,
        n_leaves: int = 64,
    ):
        super().__init__()
        self.vnet = VnetAttentionInAgg(
            in_channels=in_channel,
            out_channels=1,
            base_filters=base_filters,
            attention_kernel_size=attention_kernel_size,
            detector_width=detector_width,
            n_leaves=n_leaves,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.vnet(input)
