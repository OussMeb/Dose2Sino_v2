#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anti-mix sentinel: Detect directory collisions and ensure output integrity.

At the start of processing, write a sentinel file with patient metadata.
At the end, verify it matches before finalizing outputs.
"""

import os
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def write_patient_sentinel(patient_out_dir: Path, patient_id: str, patient_id_norm: str):
    """
    Write sentinel file at the start of patient processing.

    Args:
        patient_out_dir: Patient output directory
        patient_id: Original patient ID (e.g., "129363_group0")
        patient_id_norm: Normalized patient ID (e.g., "129363_group0")

    Returns:
        dict: Sentinel data written
    """
    patient_out_dir = Path(patient_out_dir)

    sentinel_data = {
        "patient_id": patient_id,
        "patient_id_norm": patient_id_norm,
        "start_time": time.time(),
        "start_time_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
    }

    sentinel_path = patient_out_dir / ".sentinel.json"

    try:
        # Ensure directory exists
        patient_out_dir.mkdir(parents=True, exist_ok=True)

        # Write sentinel
        with open(sentinel_path, 'w', encoding='utf-8') as f:
            json.dump(sentinel_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"[SENTINEL] Written at {sentinel_path}")
        return sentinel_data

    except Exception as e:
        logger.error(f"[SENTINEL] Failed to write sentinel: {e}")
        raise


def verify_patient_sentinel(patient_out_dir: Path, patient_id: str, patient_id_norm: str) -> bool:
    """
    Verify sentinel file matches expected patient.

    Called before finalizing outputs to detect any directory collision.

    Args:
        patient_out_dir: Patient output directory
        patient_id: Expected patient ID
        patient_id_norm: Expected normalized patient ID

    Returns:
        True if sentinel matches, False otherwise

    Raises:
        IOError: If sentinel cannot be read
    """
    patient_out_dir = Path(patient_out_dir)
    sentinel_path = patient_out_dir / ".sentinel.json"

    if not sentinel_path.exists():
        logger.error(f"[SENTINEL] Sentinel not found: {sentinel_path}")
        return False

    try:
        with open(sentinel_path, 'r', encoding='utf-8') as f:
            sentinel_data = json.load(f)

        # Verify critical fields
        sentinel_id = sentinel_data.get("patient_id")
        sentinel_id_norm = sentinel_data.get("patient_id_norm")
        sentinel_pid = sentinel_data.get("pid")

        current_pid = os.getpid()

        # Check if patient ID matches
        if sentinel_id != patient_id:
            logger.error(
                f"[SENTINEL] COLLISION DETECTED: patient_id mismatch\n"
                f"  Expected: {patient_id}\n"
                f"  Found: {sentinel_id}\n"
                f"  Sentinel location: {sentinel_path}"
            )
            return False

        if sentinel_id_norm != patient_id_norm:
            logger.error(
                f"[SENTINEL] COLLISION DETECTED: patient_id_norm mismatch\n"
                f"  Expected: {patient_id_norm}\n"
                f"  Found: {sentinel_id_norm}"
            )
            return False

        # Warn if different process (unusual but possible if process respawned)
        if sentinel_pid != current_pid:
            logger.warning(
                f"[SENTINEL] Different process ID\n"
                f"  Original: {sentinel_pid}\n"
                f"  Current: {current_pid}"
            )

        logger.debug(f"[SENTINEL] Verification passed for {patient_id}")
        return True

    except Exception as e:
        logger.error(f"[SENTINEL] Error verifying sentinel: {e}")
        return False


def get_sentinel_info(patient_out_dir: Path) -> dict | None:
    """
    Get sentinel information for debugging.

    Args:
        patient_out_dir: Patient output directory

    Returns:
        dict: Sentinel data, or None if not found/readable
    """
    sentinel_path = Path(patient_out_dir) / ".sentinel.json"

    if not sentinel_path.exists():
        return None

    try:
        with open(sentinel_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[SENTINEL] Failed to read sentinel: {e}")
        return None

