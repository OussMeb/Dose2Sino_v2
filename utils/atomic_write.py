#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atomic write utilities for multiprocess safety.

All critical file writes must be atomic to prevent corruption in case of crashes
or concurrent access.

Pattern:
    1. Write to temporary file with unique suffix
    2. Atomic rename to final path (os.replace is atomic on POSIX)
    3. Log any failures clearly
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def atomic_save_npy(output_path: Path, array: Any, description: str = ""):
    """
    Atomically save numpy array to .npy file.

    Args:
        output_path: Final path (e.g., X_montage.npy)
        array: Numpy array to save
        description: Optional description for logging

    Raises:
        IOError: If save fails
    """
    import numpy as np

    output_path = Path(output_path)
    pid = os.getpid()

    # Create temp file in same directory (ensures same filesystem)
    # Note: np.save() adds .npy extension automatically, so temp_path must NOT include .npy
    temp_path_no_ext = output_path.parent / f"{output_path.stem}.tmp.{pid}"
    temp_path = Path(str(temp_path_no_ext) + ".npy")  # np.save will add .npy

    try:
        # Write to temp file (np.save adds .npy automatically)
        np.save(temp_path_no_ext, array)

        # Atomic rename
        os.replace(temp_path, output_path)

        logger.debug(f"[ATOMIC_SAVE] Saved {description or output_path.name} ({array.nbytes} bytes)")

    except Exception as e:
        logger.error(f"[ATOMIC_SAVE] Failed to save {description or output_path}: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        raise


def atomic_save_json(output_path: Path, data: dict, description: str = ""):
    """
    Atomically save JSON data to file.

    Args:
        output_path: Final path (e.g., channel_order.json)
        data: Dictionary to save
        description: Optional description for logging

    Raises:
        IOError: If save fails
    """
    output_path = Path(output_path)
    pid = os.getpid()
    temp_path = output_path.parent / f"{output_path.name}.tmp.{pid}"

    try:
        # Write to temp file
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Atomic rename
        os.replace(temp_path, output_path)

        logger.debug(f"[ATOMIC_SAVE] Saved {description or output_path.name}")

    except Exception as e:
        logger.error(f"[ATOMIC_SAVE] Failed to save {description or output_path}: {e}")
        # Clean up temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        raise


def atomic_save_png(output_path: Path, fig, description: str = ""):
    """
    Atomically save matplotlib figure to PNG.

    Args:
        output_path: Final path (e.g., similarity_comparison.png)
        fig: Matplotlib figure object
        description: Optional description for logging

    Raises:
        IOError: If save fails
    """
    output_path = Path(output_path)
    pid = os.getpid()
    temp_path = output_path.parent / f"{output_path.name}.tmp.{pid}"

    try:
        # Save to temp file
        fig.savefig(temp_path, dpi=150, bbox_inches='tight')

        # Atomic rename
        os.replace(temp_path, output_path)

        logger.debug(f"[ATOMIC_SAVE] Saved {description or output_path.name}")

    except Exception as e:
        logger.error(f"[ATOMIC_SAVE] Failed to save {description or output_path}: {e}")
        # Clean up temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        raise


def atomic_move(src: Path, dst: Path, description: str = ""):
    """
    Atomically move file with collision detection.

    If destination already exists, append suffix with timestamp + pid.

    Args:
        src: Source path
        dst: Destination path
        description: Optional description for logging

    Returns:
        Path: Actual destination path (may differ if collision occurred)

    Raises:
        IOError: If move fails
    """
    import time

    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise IOError(f"Source does not exist: {src}")

    try:
        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        # If destination exists, add suffix
        if dst.exists():
            timestamp = int(time.time() * 1000)  # milliseconds
            pid = os.getpid()
            stem = dst.stem
            suffix = dst.suffix
            new_dst = dst.parent / f"{stem}_{timestamp}_{pid}{suffix}"

            logger.warning(f"[ATOMIC_MOVE] Destination exists, using alternate: {new_dst.name}")
            os.replace(src, new_dst)
            return new_dst
        else:
            # Normal case: destination free
            os.replace(src, dst)
            logger.debug(f"[ATOMIC_MOVE] Moved {description or src.name} to {dst.name}")
            return dst

    except Exception as e:
        logger.error(f"[ATOMIC_MOVE] Failed to move {src} to {dst}: {e}")
        raise

