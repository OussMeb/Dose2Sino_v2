#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing utilities for patient validation and filtering.

Provides functions for:
- Checking patient processing completion status
- Filtering incomplete patients needing reprocessing
- Validating Mosaiq mappings and patient IDs
"""

import json
import logging
from pathlib import Path

import numpy as np


def is_patient_complete(patient_id: str, output_root: Path) -> tuple[bool, str]:
    """
    Check if a patient has been successfully processed.

    Verifies that all required output files exist and have valid data:
    - sino.npy: 2D array
    - X_montage.npy: 3D array
    - similarity_metrics.json: passed validation
    - patient_report.json: status='ok'

    Args:
        patient_id: Patient ID to check
        output_root: Root directory where processed patient data is stored

    Returns:
        Tuple (is_complete: bool, reason: str)
    """
    patient_dir = output_root / patient_id

    if not patient_dir.exists():
        return False, "output_dir missing"

    # Check sino.npy
    sino_path = patient_dir / "sino.npy"
    if not sino_path.exists():
        return False, "sino.npy missing"
    try:
        sino = np.load(sino_path)
        if sino.ndim != 2 or sino.size == 0:
            return False, "sino.npy invalid shape"
    except Exception as e:
        return False, f"sino.npy corrupt: {e}"

    # Check X_montage.npy
    x_montage_path = patient_dir / "X_montage.npy"
    if not x_montage_path.exists():
        return False, "X_montage.npy missing"
    try:
        X = np.load(x_montage_path)
        if X.ndim != 3 or X.size == 0:
            return False, "X_montage.npy invalid shape"
    except Exception as e:
        return False, f"X_montage.npy corrupt: {e}"

    # Check similarity_metrics.json
    similarity_path = patient_dir / "similarity_metrics.json"
    if not similarity_path.exists():
        return False, "similarity_metrics.json missing"
    try:
        with open(similarity_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        if not metrics.get('passed', False):
            composite = metrics.get('composite_score', 'N/A')
            correlation = metrics.get('correlation', 'N/A')
            return False, f"similarity failed (comp={composite}, corr={correlation})"
    except Exception as e:
        return False, f"similarity_metrics.json corrupt: {e}"

    # Check patient_report.json
    report_path = patient_dir / "patient_report.json"
    if not report_path.exists():
        return False, "patient_report.json missing"
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        if report.get('status') != 'ok':
            status = report.get('status', 'unknown')
            return False, f"status={status}"
    except Exception as e:
        return False, f"patient_report.json corrupt: {e}"

    return True, "complete"


def filter_incomplete_patients(patient_ids: list, output_root: Path) -> list:
    """
    Filter out patients that are already successfully processed.

    Returns only patients that need to be (re)processed because they are:
    - Missing output files (sino.npy, X_montage.npy, etc.)
    - Have invalid/corrupted data
    - Failed similarity validation
    - Have status != 'ok' in patient_report.json

    Args:
        patient_ids: List of patient IDs to check
        output_root: Root directory where processed patient data is stored

    Returns:
        List of patient IDs that need to be processed
    """
    # Filter patients
    patients_to_process = []
    patients_already_ok = []

    print("\n" + "="*70)
    print("FILTERING PATIENTS - CHECKING COMPLETION STATUS")
    print("="*70)
    print(f"Scanning {len(patient_ids)} patients...\n")

    for patient_id in patient_ids:
        is_complete, reason = is_patient_complete(patient_id, output_root)

        if is_complete:
            patients_already_ok.append(patient_id)
            print(f"  ✅ {patient_id}: OK (skipping)")
        else:
            patients_to_process.append(patient_id)
            print(f"  ⚠️  {patient_id}: {reason} → will reprocess")

    # Summary
    print("\n" + "="*70)
    print("FILTERING SUMMARY")
    print("="*70)
    print(f"  Already complete (skipping):  {len(patients_already_ok):3d} patients")
    print(f"  Incomplete (will process):    {len(patients_to_process):3d} patients")
    print("="*70)

    if patients_already_ok:
        print(f"\n✅ Skipping {len(patients_already_ok)} already-OK patients:")
        for i, pid in enumerate(patients_already_ok[:10], 1):
            print(f"   {i:2d}. {pid}")
        if len(patients_already_ok) > 10:
            print(f"   ... and {len(patients_already_ok) - 10} more")

    if patients_to_process:
        print(f"\n🔄 Will process {len(patients_to_process)} incomplete patients:")
        for i, pid in enumerate(patients_to_process[:10], 1):
            print(f"   {i:2d}. {pid}")
        if len(patients_to_process) > 10:
            print(f"   ... and {len(patients_to_process) - 10} more")

    return patients_to_process


def filter_valid_patient_ids(unique_patient_ids: list, ptv_mapping_json: Path) -> list:
    """
    Filter patient IDs using Mosaiq mapping if available.

    Args:
        unique_patient_ids: List of unique patient IDs from INPUT_DIR
        ptv_mapping_json: Path to PTV mapping JSON file

    Returns:
        List of valid patient IDs based on mapping
    """
    if not ptv_mapping_json.exists():
        print(f"Mosaiq mapping not found: {ptv_mapping_json}. Using all patients.")
        raise SystemExit("invalid")

    with open(ptv_mapping_json, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    valid_statuses = {"complete", "filled", "complete_multi_dir"}
    valid_ids = set()

    patients = mapping.get("patients", {})
    if isinstance(patients, dict) and patients:
        valid_ids = {
            str(pid) for pid, info in patients.items()
            if (info.get("final_status") in valid_statuses or info.get("status") in valid_statuses)
        }
    else:
        for status in valid_statuses:
            entries = mapping.get(status, [])
            if not isinstance(entries, list):
                continue
            for item in entries:
                if isinstance(item, dict):
                    pid = item.get("patient_id") or item.get("id") or item.get("patient")
                    if pid is not None:
                        valid_ids.add(str(pid))
                else:
                    valid_ids.add(str(item))

    # Filter and report
    valid_patient_ids = [pid for pid in unique_patient_ids if pid in valid_ids]
    invalid_patient_ids = [pid for pid in unique_patient_ids if pid not in valid_ids]

    if invalid_patient_ids:
        print(f"⚠️  Invalid patients (not in mapping): {len(invalid_patient_ids)} patients - {invalid_patient_ids[:10]}{'...' if len(invalid_patient_ids) > 10 else ''}")
    print(f"✅ Patients after Mosaiq filter ({len(valid_patient_ids)}): {valid_patient_ids[:5]}{'...' if len(valid_patient_ids) > 5 else ''}")

    return valid_patient_ids

