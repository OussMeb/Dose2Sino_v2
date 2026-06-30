#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_2p5d.py

Full-dataset training entry for the 2.5D slice-wise model
(models/sinogram_2p5d.py), with the corrected SinogramLoss (Charbonnier + soft
false-positive penalty, no BCE).

Why 2.5D (see CLAUDE.md): the 3D V-Net stalls/collapses and overfits one sample
only to open_l1 ~0.11. The 2.5D model trains reliably, ~15x faster, and overfits
a single sample to ~0 (open_l1 0.013) once the per-leaf independent readout is
used. This is the first real training run with a sound model + loss.

Run from project root:
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -u main_2p5d.py
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
    config.CACHE_DIR = "/mnt/data/tomo_data/cache_sino_r8"   # ratio-8 cache (H~64, W~64)
    config.CHECKPOINT_DIR = "/mnt/data/sinogram_generator/checkpoints"

    config.REDUCTION_RATIO = 8
    config.USE_CACHE = True
    config.NUM_WORKERS = 10

    config.BATCH_SIZE = 1            # B=1: the model unfolds N_CP slices into the batch axis
    config.BASE_FILTERS = 24         # 2.5D base channels
    config.USE_MIXED_PRECISION = True

    config.LEARNING_RATE = 1e-3

    # Cosine LR over a horizon scaled to TRAINING (not the sanity's fast 900-step
    # rate). Anneals 1e-3 -> 1e-5 gradually over COSINE_T_MAX_EPOCHS epochs, stepped
    # per epoch. Cosine polished fine details in the sanity; the old ReduceLROnPlateau
    # barely decayed before early-stop. Keep a generous early-stop as a safety net.
    config.LR_SCHEDULE = "cosine"
    config.COSINE_T_MAX_EPOCHS = 20
    config.MIN_LR = 1e-5
    config.AUTO_STOP = 15            # reach ~T_max epochs (cosine done) before stopping

    config.pos_weight = 8.0
    config.fp_weight = 4.0

    # Flip augmentation as a generalization regularizer (the open problem is the
    # ~0.31 val plateau from few patients). Leaf-flip is a VALID mirror equivariance
    # (mirror the BEV anatomy H + mirror the target 64-leaf axis = a mirror-image
    # patient); ray-flip is input-only (projection is ray-order invariant). The
    # earlier "flip hurts" verdict was confounded (old ratio-3 self-reducing arch).
    # Isolate flips: no CT jitter (ZOOM_PROB=0), no weight decay.
    config.USE_AUGMENTATION = True   # COMBINED: rotation-fix + flip-aug (both confirmed-helpful levers)
    config.FLIP_PROB = 0.5           # drives both leaf-axis and ray-axis flips
    config.ZOOM_PROB = 0.0           # CT jitter off -> isolate the flip effect
    config.WEIGHT_DECAY = 0.0
    config.BASE_FILTERS = 24

    config.JOURNAL = (
        "2.5D, REDUCTION_RATIO=8 + reduce_h=False (1:1 leaf readout), BASE_FILTERS=24. "
        "Dataset includes DIBH acquisitions (1062 samples, leakage-safe split) + 40cm "
        "MLC-field W-crop. LOSS = SinogramLoss (Charbonnier+FP) -- the L1-rule loss "
        "COLLAPSED to zero in sanity (vanishing grad near p=0); Charbonnier overfits one "
        "sample to open_l1 ~0.01 / precision ~100%. Physical sinogram rules (min-LOT, "
        "leaf cycles, floor) are enforced at EXPORT (generate_rtplan_attention_in_agg.py), "
        "not in the loss. COSINE LR 1e-3->1e-5 over 20 epochs. pos_weight=8 fp_weight=4. "
        "*** ROTATION-CONVENTION FIX *** apply_tomo_transform now uses (90-theta): the old "
        "theta mis-registered berlingo vs sinogram (centroid corr -0.6 -> +0.88, 3 patients) -- "
        "likely the true cause of the ~0.31 plateau. NO augmentation here to ISOLATE the rotation "
        "fix's effect. Baseline to beat: best val open_l1 0.3105 (mis-registered data)."
    )

    model = DosePrediction2p5D(
        base_filters=config.BASE_FILTERS,
        in_channel=2,
        n_leaves=64,
        slice_chunk=256,   # bound peak memory: process N_CP slices in chunks (checkpointed)
        reduce_h=False,    # ratio-8: keep H~64 -> 1:1 leaf readout (no leaf correlation)
    )

    loss = SinogramLoss(eps=1e-3, pos_weight=config.pos_weight, fp_weight=config.fp_weight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # benchmark=False: the 2.5D model unfolds N_CP into the batch axis, so the
        # batch size varies per patient. With benchmark=True cuDNN re-tunes conv
        # algorithms on every new shape (~10x slowdown); False keeps it stable.
        torch.backends.cudnn.benchmark = False

    trainer = TrainerSupervisedLogits(
        config=config,
        model=model,
        device=device,
        loss_function=loss,
        phase_suffix="2p5d",
    )

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    logging.info(
        "Training completed in %.2f hours.",
        (end_time - start_time).total_seconds() / 3600,
    )

    trainer.test()
