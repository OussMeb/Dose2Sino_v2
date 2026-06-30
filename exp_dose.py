#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_dose.py — LEVER B: train with the differentiable dose-consistency loss.

Hypothesis: open_l1 is saturated at the ~0.15 anatomy->sinogram degeneracy
(EXPERIMENTS "degeneracy floor"); matching the DELIVERED DOSE (not one arbitrary
plan) is the right objective. The loss = scale-invariant simplified-CCC dose match
(utils/dose_operator, ray-tracer fidelity ~0.92, the differentiable GRADIENT) +
sinogram anchor (deliverability) + amplitude/MU term (the dose term is scale-
invariant so absolute level must be pinned separately; fixes the +8% seen in CCC
validation). DoseCUDA (real CCC, ~14 min/plan) is the OFFLINE gold JUDGE, not the loss.

Calibrated operating point (sanity sweep): dose_weight=1, sino_weight=2.0 (open_l1
lands ~0.15 = as close to GT as another valid plan), amp_weight tuned so pred mean
matches GT. Changes vs exp_base32.py: ONLY the loss (same base32 / r8 cache / cosine /
flip-aug / splits). Baseline to beat is judged in DOSE space, NOT open_l1.

Run a short PROBE first (watch the "VAL dose metric (mean L_dose)" line for a few
epochs); commit to the full run only if it improves. ~2-3x slower/step than base32
(the dose accumulation), so ~1 h/epoch.

    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -u exp_dose.py
"""
from __future__ import annotations
from datetime import datetime
import logging, os
import torch

from models.sinogram_2p5d import DosePrediction2p5D
from utils.config import Config
from utils.losses import DoseConsistencyLoss
from utils.trainer_supervised_logits import TrainerSupervisedLogits

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
torch.manual_seed(42)

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    config = Config()
    config.DATA_PATH = "/mnt/data/tomo_data/"
    config.CACHE_DIR = "/mnt/data/tomo_data/cache_sino_r8"
    config.CHECKPOINT_DIR = "/mnt/data/sinogram_generator/checkpoints"

    config.REDUCTION_RATIO = 8
    config.USE_CACHE = True
    config.NUM_WORKERS = 10
    config.TARGET_HW = 64

    config.BATCH_SIZE = 1
    config.BASE_FILTERS = 32
    config.USE_MIXED_PRECISION = True

    config.LEARNING_RATE = 1e-3
    config.LR_SCHEDULE = "cosine"
    config.COSINE_T_MAX_EPOCHS = 20
    config.MIN_LR = 1e-5
    config.AUTO_STOP = 15

    config.pos_weight = 8.0
    config.fp_weight = 4.0
    # dose-loss operating point (sanity-calibrated)
    config.dose_weight = 1.0
    config.sino_weight = 2.0
    config.amp_weight = 5.0
    config.n_z = 48
    # low-fidelity acquisitions (ray-tracer corr < ~0.65, a per-acquisition couch
    # issue): fall back to the sinogram anchor only for these (no dose gradient).
    config.DOSE_EXCLUDE = {"187591", "223696_DIBH", "229221_DIBH"}

    config.USE_AUGMENTATION = True
    config.FLIP_PROB = 0.5
    config.ZOOM_PROB = 0.0
    config.WEIGHT_DECAY = 0.0

    config.JOURNAL = (
        "LEVER B / dose-consistency loss. base32 geometry-fixed pipeline. Loss = "
        "DoseConsistencyLoss(dose_w=1, sino_w=2, amp_w=5). Judge in DOSE space "
        "(VAL mean L_dose; DoseCUDA CCC on a few val patients offline). open_l1 saturated."
    )

    model = DosePrediction2p5D(
        base_filters=config.BASE_FILTERS, in_channel=2, n_leaves=64,
        slice_chunk=256, reduce_h=False,
    )
    loss = DoseConsistencyLoss(
        n_z=config.n_z, dose_weight=config.dose_weight, sino_weight=config.sino_weight,
        amp_weight=config.amp_weight, pos_weight=config.pos_weight, fp_weight=config.fp_weight,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    trainer = TrainerSupervisedLogits(
        config=config, model=model, device=device,
        loss_function=loss, phase_suffix="2p5d_dose",
    )

    start = datetime.now()
    trainer.train()
    logging.info("Training completed in %.2f hours.",
                 (datetime.now() - start).total_seconds() / 3600)
    trainer.test()
