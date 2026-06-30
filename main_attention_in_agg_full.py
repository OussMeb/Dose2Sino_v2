#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_attention_in_agg_full.py

Full-dataset training entry for:
- Attention
- InstanceNorm3d
- learned Conv3D aggregation
- SinogramLoss on raw logits

Place next to main.py and run from project root.
"""

from __future__ import annotations

from datetime import datetime
import logging
import os
import torch

from models.unet_attention_in_agg import DosePredictionAttentionInAgg
from utils.config import Config
from utils.losses import SinogramLoss
from utils.trainer_supervised_logits import TrainerSupervisedLogits


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
torch.manual_seed(42)

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    config = Config()

    config.DATA_PATH = "/mnt/data/tomo_data/"
    config.CACHE_DIR = "/mnt/data/tomo_data/cache_sino"
    config.CHECKPOINT_DIR = "/mnt/data/sinogram_generator/checkpoints"

    # Higher-resolution input (smaller ratio = more data). The model now resamples
    # the in-plane axes down to the canonical 64-leaf x detector_width grid at its
    # output stage, so any REDUCTION_RATIO is supported.
    config.REDUCTION_RATIO = 3

    config.USE_CACHE = True
    config.NUM_WORKERS = 10
    config.BASE_FILTERS = 8

    # Keep full-dataset run controlled.
    config.BATCH_SIZE = 1
    config.USE_MIXED_PRECISION = True
    config.pos_weight = 8.0
    config.fp_weight = 4.0

    # The scheduler/early-stop settings can stay from Config.
    config.JOURNAL = (
        "Full dataset: Attention + InstanceNorm3d + learned Conv3D aggregation "
        "+ SinogramLoss(alpha=0.5,pos_weight=8.0,open-weighted), REDUCTION_RATIO=3, BASE_FILTERS=8."
    )

    model = DosePredictionAttentionInAgg(
        base_filters=config.BASE_FILTERS,
        in_channel=2,
        attention_kernel_size=15,
        detector_width=64,
    )

    loss = SinogramLoss(eps=1e-3, pos_weight=config.pos_weight, fp_weight=config.fp_weight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    trainer = TrainerSupervisedLogits(
        config=config,
        model=model,
        device=device,
        loss_function=loss,
        phase_suffix="attention_in_agg_full",
    )

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    logging.info(
        "Training completed in %.2f hours.",
        (end_time - start_time).total_seconds() / 3600,
    )

    trainer.test()
