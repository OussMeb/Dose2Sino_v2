#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_hw128.py  —  LEVER 1 (input resolution), methodical A/B vs the geometry-fixed baseline.

Variable changed vs main_2p5d.py: TARGET_HW 64 -> 128 (the berlingo in-plane size:
H=leaf, W=ray). Doubles the dose/CT sampling the model sees. reduce_h flips to True
so the stem halves H (128->64) and the per-leaf-independent 1:1 readout regime is
preserved (pooling 64->64); this coupling is intrinsic to the resolution change.

Tests the "floor = lossy low-res dose input" hypothesis: if a 128 berlingo lowers
val open_l1 below 0.158, the inverse was input-resolution-limited; if not, the
floor is data/generalization, not input resolution.

Needs a DEDICATED cache (cache key encodes neither ratio nor target_hw):
    generate_cache.py --target-hw 128 --cache-dir /mnt/data/tomo_data/cache_sino_hw128 --force

Baseline to beat: val open_l1 0.158 (TARGET_HW=64). NOTE: ~4x the compute/memory
of the 64 baseline (which was ~2 h/epoch on the M6000) -> expect a multi-day run.

    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -u exp_hw128.py
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
    config.CACHE_DIR = "/mnt/data/tomo_data/cache_sino_hw128"   # <-- dedicated 128 cache
    config.CHECKPOINT_DIR = "/mnt/data/sinogram_generator/checkpoints"

    config.REDUCTION_RATIO = 8
    config.USE_CACHE = True
    config.NUM_WORKERS = 10
    config.TARGET_HW = 128           # <-- LEVER 1: input resolution 64 -> 128

    config.BATCH_SIZE = 1
    config.BASE_FILTERS = 24
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
        "LEVER 1 / input resolution. Geometry-fixed pipeline. TARGET_HW 64->128 "
        "(reduce_h=True: stem 128->64 keeps 1:1 leaf readout). Baseline val open_l1 0.158."
    )

    model = DosePrediction2p5D(
        base_filters=config.BASE_FILTERS,
        in_channel=2,
        n_leaves=64,
        slice_chunk=256,
        reduce_h=True,     # H=128 -> stem H/2 -> 64 -> 1:1 per-leaf readout
    )

    loss = SinogramLoss(eps=1e-3, pos_weight=config.pos_weight, fp_weight=config.fp_weight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    trainer = TrainerSupervisedLogits(
        config=config, model=model, device=device,
        loss_function=loss, phase_suffix="2p5d_hw128",
    )

    start = datetime.now()
    trainer.train()
    logging.info("Training completed in %.2f hours.",
                 (datetime.now() - start).total_seconds() / 3600)
    trainer.test()
