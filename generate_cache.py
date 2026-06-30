#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_cache.py

Precompute the on-disk sample cache for the whole dataset, in parallel.

Each RTDataset.__getitem__ writes its processed sample to
    <cache_dir>/<patient_id>_<pareto_index>.pt.gz
as a side effect (see utils/patient.py). So building the cache is just a matter
of iterating over every index once with use_cache=True. We use a DataLoader with
many workers to do that in parallel; a lightweight collate_fn keeps only the
patient id so the heavy tensors are never serialized back to the main process.

NOTE: the cache key is (patient_id, pareto_index) and does NOT encode
REDUCTION_RATIO. A cache built at one ratio will be silently reused at another.
Use a ratio-specific --cache-dir (the default) or pass --force to wipe first.

Examples:
    # Mirror main_attention_in_agg_full.py (ratio=3), 10 workers
    python generate_cache.py --reduction-ratio 3 --num-workers 10

    # Custom paths, wipe any stale cache first
    python generate_cache.py --data-path /mnt/data/tomo_data/ \
        --cache-dir /mnt/data/tomo_data/cache_sino_r3 --reduction-ratio 3 --force
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import Config
from utils.patient import RTDataset


def _id_only_collate(batch):
    """Return just the patient ids; the cache write already happened in the worker."""
    return [(s["patient_id"], s["pareto_index"]) for s in batch]


def parse_args() -> argparse.Namespace:
    config = Config()
    parser = argparse.ArgumentParser(description="Precompute the dataset cache in parallel.")
    parser.add_argument("--data-path", default="/mnt/data/tomo_data/",
                        help="Dataset root (default: %(default)s).")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache output dir. Default: <data-path>/cache_sino_r<ratio>.")
    parser.add_argument("--reduction-ratio", type=int, default=3,
                        help="In-plane downsampling ratio (default: %(default)s).")
    parser.add_argument("--num-workers", type=int, default=10,
                        help="DataLoader workers (default: %(default)s).")
    parser.add_argument("--max-dose", type=float, default=config.MAX_DOSE,
                        help="Dose normalization (default: %(default)s).")
    parser.add_argument("--force", action="store_true",
                        help="Delete the cache dir before generating (avoids stale ratio reuse).")
    parser.add_argument("--target-hw", type=int, default=64,
                        help="In-plane berlingo size H=leaf,W=ray (default: %(default)s). "
                             "NOT encoded in the cache key -> use a distinct --cache-dir per size.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir \
        else Path(args.data_path) / f"cache_sino_r{args.reduction_ratio}"

    if args.force and cache_dir.exists():
        logging.warning("Wiping existing cache dir %s", cache_dir)
        shutil.rmtree(cache_dir)

    logging.info("Building cache in %s (ratio=%d, workers=%d)",
                 cache_dir, args.reduction_ratio, args.num_workers)

    dataset = RTDataset(
        args.data_path,
        augmentation=None,
        max_dose=args.max_dose,
        use_cache=True,
        cache_dir=str(cache_dir),
        reduction_ratio=args.reduction_ratio,
        target_hw=args.target_hw,
    )

    total = len(dataset)
    if total == 0:
        logging.error("No samples found under %s — nothing to cache.", args.data_path)
        return

    logging.info("Found %d samples.", total)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=False,
        collate_fn=_id_only_collate,
    )

    failures = 0
    for batch in tqdm(loader, total=total, desc="Caching", unit="sample"):
        if not batch:
            failures += 1

    cached = len(list(cache_dir.glob("*.pt.gz")))
    logging.info("Done. %d/%d cache files present in %s (%d empty batches).",
                 cached, total, cache_dir, failures)


if __name__ == "__main__":
    main()
