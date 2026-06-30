#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_fourier.py  —  LEVER 3 (Fourier coupling along the gantry/control-point axis).

Hypothesis: the 2.5D model predicts every control point INDEPENDENTLY and the conv
refine head couples only +/-1-2 CP, so it cannot see the GLOBAL low-frequency angular
structure that GT sinograms provably have (measured: 10x more low-freq power along
N_CP than a CP-shuffled null; ~50% of angular power in the lowest ~9% of bins;
N_CP~1300). Adding a 1D Fourier (FNO) refine along N_CP + an angular-spectral loss
should improve GENERALIZATION (the ~0.153/0.26 val regime), which a global low-freq
prior helps but a single-sample overfit cannot show.

A/B vs the LEVER 2 baseline (base32, val open_l1 ~0.153). Changes vs exp_base32.py:
  - refine=True, refine_mode="both" (local conv + global FNO on the [N_CP,64] plane)
  - loss = SinogramSpectralLoss (SinogramLoss + AngularSpectralLoss, weight 0.5)
Everything else identical (base32, r8/64^2 cache, cosine LR, flip-aug, seed/splits).
NOTE: "both+spectral" bundles conv-refine + FNO + spectral loss; if it beats 0.153,
a follow-up A/B should isolate which component carries the gain.

    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -u exp_fourier.py
"""
from __future__ import annotations

from datetime import datetime
import logging
import os
import torch

from models.sinogram_2p5d import DosePrediction2p5D
from utils.config import Config
from utils.losses import SinogramSpectralLoss
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
    config.BASE_FILTERS = 32         # match LEVER 2 baseline (the 0.153 reference)
    config.USE_MIXED_PRECISION = True

    config.LEARNING_RATE = 1e-3
    config.LR_SCHEDULE = "cosine"
    config.COSINE_T_MAX_EPOCHS = 20
    config.MIN_LR = 1e-5
    config.AUTO_STOP = 15

    config.pos_weight = 8.0
    config.fp_weight = 4.0
    config.spectral_weight = 0.5
    config.spectral_modes = 32
    config.fno_modes = 64

    config.USE_AUGMENTATION = True
    config.FLIP_PROB = 0.5
    config.ZOOM_PROB = 0.0
    config.WEIGHT_DECAY = 0.0

    config.JOURNAL = (
        "LEVER 3 / Fourier angular coupling. Geometry-fixed pipeline, base32 baseline "
        "(val open_l1 0.153). Changes: refine_mode=both (conv+FNO along N_CP) + "
        "SinogramSpectralLoss (spectral_weight 0.5, modes 32). Tests whether global "
        "low-freq gantry-angle structure improves generalization."
    )

    model = DosePrediction2p5D(
        base_filters=config.BASE_FILTERS,
        in_channel=2,
        n_leaves=64,
        slice_chunk=256,
        reduce_h=False,    # TARGET_HW=64: keep H -> 1:1 leaf readout
        refine=True,
        refine_mode="both",
        fno_modes=config.fno_modes,
    )

    loss = SinogramSpectralLoss(
        eps=1e-3, pos_weight=config.pos_weight, fp_weight=config.fp_weight,
        spectral_weight=config.spectral_weight, spectral_modes=config.spectral_modes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    trainer = TrainerSupervisedLogits(
        config=config, model=model, device=device,
        loss_function=loss, phase_suffix="2p5d_fourier",
    )

    start = datetime.now()
    trainer.train()
    logging.info("Training completed in %.2f hours.",
                 (datetime.now() - start).total_seconds() / 3600)
    trainer.test()
