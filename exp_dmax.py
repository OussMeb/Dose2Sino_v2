#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_dmax.py — LEVER C: fine-tune with a Dmax / hotspot penalty.

Hypothesis: the +9-10% Dmax hotspot observed on every val patient via DoseCUDA CCC is
caused by regression smoothing (the model predicts a slightly average sinogram, which
when accumulated concentrates dose spatially). A one-sided Dmax hinge loss, computed
via the real Tomo 6MV CCC scatter kernel (isotropic double-exponential, corr=0.893 vs
real CCC Dmax), provides a differentiable gradient that should push pred_Dmax <= gt_Dmax.

Approach: fine-tune the best dose-trained checkpoint (base32, r8) with:
  DoseConsistencyLoss(dose_weight=1, sino_weight=2, amp_weight=5, dmax_weight=1)
  + the CCC Tomo kernel for Dmax (dz=0.5cm, dxy=0.625cm, rmax=5cm)

Run:
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      .venv/bin/python -u exp_dmax.py

Judge: DoseCUDA CCC on val patients (validate_dose_ccc_general.py). Metric to watch:
  dp.max() / dgt.max() (hotspot ratio) — should drop from ~1.10 toward 1.00.
  Correlation and mean-dose error should stay at ~0.99 / < 3%.
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

DOSE_CKPT = "checkpoints/20260626_200610_2p5d_dose/best_model_new_session_session_0_.pth"

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

    # Lower LR for fine-tuning (start from dose ckpt)
    config.LEARNING_RATE = 2e-4
    config.LR_SCHEDULE = "cosine"
    config.COSINE_T_MAX_EPOCHS = 15
    config.MIN_LR = 1e-5
    config.AUTO_STOP = 10

    config.pos_weight = 8.0
    config.fp_weight = 4.0
    config.dose_weight = 1.0
    config.sino_weight = 2.0
    config.amp_weight = 5.0
    config.dmax_weight = 1.0
    config.n_z = 48
    config.DOSE_EXCLUDE = {"187591", "223696_DIBH", "229221_DIBH"}

    config.USE_AUGMENTATION = True
    config.FLIP_PROB = 0.5
    config.ZOOM_PROB = 0.0
    config.WEIGHT_DECAY = 0.0

    config.JOURNAL = (
        "LEVER C / Dmax penalty. Fine-tune from dose ckpt. Loss = "
        "DoseConsistencyLoss(dose_w=1, sino_w=2, amp_w=5, dmax_w=1). "
        "CCC kernel (dz=0.5, dxy=0.625, rmax=5cm, corr=0.893 vs CCC). "
        "Judge: DoseCUDA CCC hotspot ratio on val patients."
    )

    model = DosePrediction2p5D(
        base_filters=config.BASE_FILTERS, in_channel=2, n_leaves=64,
        slice_chunk=256, reduce_h=False,
    )
    # Load the best dose checkpoint as starting point
    ckpt = torch.load(DOSE_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    logging.info("Fine-tuning from %s (epoch %s)", DOSE_CKPT, ckpt.get("epoch", "?"))

    loss = DoseConsistencyLoss(
        n_z=config.n_z,
        dose_weight=config.dose_weight,
        sino_weight=config.sino_weight,
        amp_weight=config.amp_weight,
        dmax_weight=config.dmax_weight,
        pos_weight=config.pos_weight,
        fp_weight=config.fp_weight,
        dz_cm=0.5,
        dxy_cm=0.625,
        dmax_rmax_cm=5.0,
    )
    logging.info("DoseConsistencyLoss dmax_weight=%.1f kernel=%s",
                 loss.dmax_weight, tuple(loss._ccc_kernel.shape) if loss._ccc_kernel is not None else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    trainer = TrainerSupervisedLogits(
        config=config, model=model, device=device,
        loss_function=loss, phase_suffix="2p5d_dmax",
    )

    start = datetime.now()
    trainer.train()
    logging.info("Training completed in %.2f hours.",
                 (datetime.now() - start).total_seconds() / 3600)
    trainer.test()
