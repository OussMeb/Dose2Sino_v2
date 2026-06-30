#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O utilities for saving arrays, images, and files.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from .json_utils import normalize_name


def _ensure_parent_dir(path: str | Path) -> None:
    """Ensure parent directory exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_array_npy(path_npy: str | Path, x: np.ndarray, dtype: str = "float32") -> None:
    """Save array as NPY (float32 recommended)."""
    _ensure_parent_dir(path_npy)
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    np.save(str(path_npy), arr)


def save_array_npz(path_npz: str | Path, **arrays: np.ndarray) -> None:
    """Save compressed NPZ: np.savez_compressed("file.npz", key=array, ...)"""
    _ensure_parent_dir(path_npz)
    np.savez_compressed(str(path_npz), **arrays)


def save_png(out: Path | str, arr: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> None:
    """
    Save PNG in grayscale WITHOUT inversion:
      - 0 -> black
      - 0.5 -> gray
      - 1 -> white
    """
    a = arr.astype(np.float32)
    a = np.clip(a, vmin, vmax)
    plt.imsave(str(out), a, cmap="gray", vmin=vmin, vmax=vmax)


def save_array_auto(
    base_path_no_ext: str | Path,
    x01: np.ndarray,
    save_format: str = "both",
    npy_dtype: str = "float32",
    save_npz: bool = False,
    npz_key: str = "arr"
) -> None:
    """
    Save array in specified format(s):
      - "png": saves as PNG
      - "npy": saves as NPY
      - "both": saves both formats
      - optionally saves NPZ if save_npz=True
    """
    fmt = (save_format or "png").lower()
    if fmt not in ["png", "npy", "both"]:
        raise ValueError(f"Invalid save_format: {save_format}. Expected: 'png'|'npy'|'both'.")

    base = str(base_path_no_ext)

    if fmt in ["png", "both"]:
        save_png(str(base + ".png"), x01)

    if fmt in ["npy", "both"]:
        save_array_npy(base + ".npy", x01, dtype=npy_dtype)
        if save_npz:
            save_array_npz(base + ".npz", **{npz_key: np.asarray(x01).astype(npy_dtype, copy=False)})


def build_x_from_saved_montages(
    patient_out_dir: Path,
    order: list[str],
    ny: int,
    nx: int,
    dtype: str = "float32",
    allow_missing: bool = True,
) -> np.ndarray:
    """
    Build X array from saved montage files:
      patient_out_dir/berlingo_<name>.npy  (montage 2D: [NY*N_CP, NX])

    Returns:
      X_montage: shape [C, NY*N_CP, NX] float32
        - if montage missing -> channel = 0 (if allow_missing)
        - verifies NX and divisibility by NY
    """
    C = len(order)

    # 1) Determine H (= NY*NCP) and NCP by finding first existing montage
    H = None
    first_path = None
    for name in order:
        key = normalize_name(name)
        p = patient_out_dir / f"berlingo_{key}.npy"
        if p.exists():
            arr = np.load(str(p), mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"Invalid montage {p.name}: ndim={arr.ndim} (expected 2)")
            if arr.shape[1] != nx:
                raise ValueError(f"Invalid montage {p.name}: NX={arr.shape[1]} (expected {nx})")
            H = int(arr.shape[0])
            first_path = p
            break

    if H is None:
        raise FileNotFoundError(f"No .npy montage found in {patient_out_dir} to build X.")

    if (H % ny) != 0:
        raise ValueError(
            f"Montage height H={H} not divisible by NY={ny}. Ex: {first_path.name if first_path else '??'}")

    n_cp = H // ny

    # 2) Allocate X
    X = np.zeros((C, H, nx), dtype=np.float32)

    # 3) Load each channel
    for ch, name in enumerate(order):
        key = normalize_name(name)
        p = patient_out_dir / f"berlingo_{key}.npy"
        if not p.exists():
            if allow_missing:
                logging.info(f"[X-MONTAGE] missing {p.name} -> channel {ch} zeros")
                continue
            raise FileNotFoundError(f"Missing montage: {p}")

        arr = np.load(str(p))
        if arr.ndim != 2:
            raise ValueError(f"Invalid montage {p.name}: ndim={arr.ndim} (expected 2)")
        if arr.shape != (H, nx):
            raise ValueError(f"Montage shape mismatch {p.name}: {arr.shape} expected {(H, nx)}")

        X[ch] = arr.astype(np.float32, copy=False)

    # 4) Final cast
    if dtype is not None:
        X = X.astype(dtype, copy=False)

    logging.info(f"[X-MONTAGE] built from disk: shape={X.shape} (C,H,NX)=({C},{H},{nx}), n_cp={n_cp}, NY={ny}")
    return X

