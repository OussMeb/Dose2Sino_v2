# models/unet_attention.py
# -*- coding: utf-8 -*-
"""
Attention-augmented VNet for CT+dose berlingo -> LOT sinogram prediction.

This file is intentionally separate from models/unet.py so the baseline model
remains untouched during architecture sanity checks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPChannelAttention(nn.Module):
    """
    Lightweight attention over the control-point axis.

    Input:
        x: [B, C, N_CP, H, W]

    The module pools detector dimensions H/W, learns a channel-wise attention
    signal over N_CP, then applies it residually. Gamma starts at zero so the
    model is initialized exactly like the baseline path.
    """

    def __init__(self, channels: int, kernel_size: int = 15) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length CP attention.")

        padding = kernel_size // 2

        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.activation = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = x.mean(dim=(-1, -2))
        attention = self.activation(self.pointwise(self.depthwise(context)))
        attention = attention.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + self.gamma * attention)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = True,
        attention_kernel_size: int = 15,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention = (
            CPChannelAttention(out_channels, kernel_size=attention_kernel_size)
            if use_attention
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.attention(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 2, 2),
            stride=(1, 2, 2),
            padding=(1, 0, 0),
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class UpSampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.upsample(x)))


class AttentionVNet(nn.Module):
    """
    Baseline VNet + lightweight CP/channel attention.

    Input:
        [B, in_channels, N_CP, 64, 64]

    Output:
        [B, 1, N_CP, 64, 1]

    The detector dimensions are downsampled/upsampled like the baseline model.
    The CP dimension is kept fixed, and attention gates learn CP-aware feature
    recalibration without full self-attention memory cost.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_filters: int = 16,
        attention_kernel_size: int = 15,
    ) -> None:
        super().__init__()

        self.encoder_block1 = ConvBlock(
            in_channels,
            base_filters,
            use_attention=True,
            attention_kernel_size=attention_kernel_size,
        )
        self.down1 = DownSampling(base_filters, base_filters * 2)

        self.encoder_block2 = ConvBlock(
            base_filters * 2,
            base_filters * 2,
            use_attention=True,
            attention_kernel_size=attention_kernel_size,
        )
        self.down2 = DownSampling(base_filters * 2, base_filters * 4)

        self.encoder_block3 = ConvBlock(
            base_filters * 4,
            base_filters * 4,
            use_attention=True,
            attention_kernel_size=attention_kernel_size,
        )
        self.down3 = DownSampling(base_filters * 4, base_filters * 8)

        self.bottleneck = ConvBlock(
            base_filters * 8,
            base_filters * 8,
            use_attention=True,
            attention_kernel_size=attention_kernel_size,
        )

        self.up3 = UpSampling(base_filters * 8, base_filters * 4)
        self.decoder_block3 = ConvBlock(
            base_filters * 8,
            base_filters * 4,
            use_attention=True,
            attention_kernel_size=attention_kernel_size,
        )

        self.up2 = UpSampling(base_filters * 4, base_filters * 2)
        self.decoder_block2 = ConvBlock(
            base_filters * 4,
            base_filters * 2,
            use_attention=True,
            attention_kernel_size=attention_kernel_size,
        )

        self.up1 = UpSampling(base_filters * 2, base_filters)
        self.decoder_block1 = ConvBlock(
            base_filters * 2,
            base_filters,
            use_attention=True,
            attention_kernel_size=attention_kernel_size,
        )

        self.output = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder_block1(x)
        x2 = self.encoder_block2(self.down1(x1))
        x3 = self.encoder_block3(self.down2(x2))

        bottleneck = self.bottleneck(self.down3(x3))

        d3 = self.up3(bottleneck)
        if d3.size()[2:] != x3.size()[2:]:
            d3 = F.interpolate(
                d3,
                size=x3.size()[2:],
                mode="trilinear",
                align_corners=False,
            )
        d3 = self.decoder_block3(torch.cat([d3, x3], dim=1))
        del x3

        d2 = self.up2(d3)
        if d2.size()[2:] != x2.size()[2:]:
            d2 = F.interpolate(
                d2,
                size=x2.size()[2:],
                mode="trilinear",
                align_corners=False,
            )
        d2 = self.decoder_block2(torch.cat([d2, x2], dim=1))
        del x2

        d1 = self.up1(d2)
        if d1.size()[2:] != x1.size()[2:]:
            d1 = F.interpolate(
                d1,
                size=x1.size()[2:],
                mode="trilinear",
                align_corners=False,
            )
        d1 = self.decoder_block1(torch.cat([d1, x1], dim=1))
        del x1

        output_3d = F.relu(self.output(d1))
        return torch.sum(output_3d, dim=4, keepdim=True)


class DosePredictionAttention(nn.Module):
    """
    Drop-in replacement for baseline DosePrediction.

    Keeps the same input/output contract:
        input:  [B, 2, N_CP, 64, 64]
        output: [B, 1, N_CP, 64, 1]
    """

    def __init__(
        self,
        base_filters: int = 16,
        in_channel: int = 2,
        attention_kernel_size: int = 15,
    ) -> None:
        super().__init__()
        self.vnet = AttentionVNet(
            in_channels=in_channel,
            out_channels=1,
            base_filters=base_filters,
            attention_kernel_size=attention_kernel_size,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.vnet(input_tensor)
