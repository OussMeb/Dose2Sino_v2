#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_base32.py  —  LEVER 2 (capacity), methodical A/B vs the geometry-fixed baseline.

ONLY variable changed vs main_2p5d.py: BASE_FILTERS 24 -> 32 (~735k params vs 414k).
(base48 was tried first but ran ~99 s/it / ~22 h/epoch on the M6000 — Maxwell, no
Tensor Cores — so ~2 weeks to plateau, untenable. base32 ~1.8x compute is tractable
at ~3 h/epoch while still testing the "model underfits" hypothesis.)

Same 64² cache, loss, cosine LR, flip-aug, seed/splits. Baseline to beat: val
open_l1 0.158.

    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -u exp_base32.py
"""
from __future__ import annotations

from datetime import datetime
import logging
import os
import torch

from models.sinogram_2p5d import DosePrediction2p5D
from utils.config import Config
from utils.losses import SinogramLoss
from utils.trainer_supervised_logits import TrainerSupervisedLogits


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
torch.manual_seed(42)

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    config = Config()
    config.DATA_PATH = "/mnt/data/tomo_data/"
    config.CACHE_DIR = "/mnt/data/tomo_data/cache_sino_r8"   # TARGET_HW=64 (geometry-fixed)
    config.CHECKPOINT_DIR = "/mnt/data/sinogram_generator/checkpoints"

    config.REDUCTION_RATIO = 8
    config.USE_CACHE = True
    config.NUM_WORKERS = 10
    config.TARGET_HW = 64

    config.BATCH_SIZE = 1
    config.BASE_FILTERS = 32         # <-- LEVER 2: capacity 24 -> 32
    config.USE_MIXED_PRECISION = True

    config.LEARNING_RATE = 1e-3
    config.LR_SCHEDULE = "cosine"
    config.COSINE_T_MAX_EPOCHS = 20
    config.MIN_LR = 1e-5
    config.AUTO_STOP = 15

    config.pos_weight = 8.0
    config.fp_weight = 4.0

    config.USE_AUGMENTATION = True
    config.FLIP_PROB = 0.5
    config.ZOOM_PROB = 0.0
    config.WEIGHT_DECAY = 0.0

    config.JOURNAL = (
        "LEVER 2 / capacity. Geometry-fixed pipeline (symmetric 40cm crop + z_iso-tables). "
        "Only change vs baseline: BASE_FILTERS 24->32 (base48 too slow on M6000). "
        "Baseline val open_l1 0.158."
    )

    model = DosePrediction2p5D(
        base_filters=config.BASE_FILTERS,
        in_channel=2,
        n_leaves=64,
        slice_chunk=256,
        reduce_h=False,    # TARGET_HW=64: keep H -> 1:1 leaf readout
    )

    loss = SinogramLoss(eps=1e-3, pos_weight=config.pos_weight, fp_weight=config.fp_weight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    trainer = TrainerSupervisedLogits(
        config=config, model=model, device=device,
        loss_function=loss, phase_suffix="2p5d_base32",
    )

    start = datetime.now()
    trainer.train()
    logging.info("Training completed in %.2f hours.",
                 (datetime.now() - start).total_seconds() / 3600)
    trainer.test()
