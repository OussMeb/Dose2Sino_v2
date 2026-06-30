#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_repair_rtplan_closed_rows.py

Inspect and optionally repair Tomo/Radixact RTPLAN sinogram rows.

Purpose:
    RayStation may reject an AI-generated RTPLAN if rows that must be fully
    closed contain tiny nonzero values from sigmoid prediction.

What this script checks:
    reference RTPLAN row == all zeros
    predicted RTPLAN same row != all zeros

What repair does:
    For every row fully closed in the reference RTPLAN, force the same predicted
    RTPLAN row to exact 64 zeros.

It does NOT modify the original files unless --repair-output is provided.

Usage:

1) Inspect only:
    python inspect_repair_rtplan_closed_rows.py \
      --reference /mnt/data/shared/tomo_data/297768/RC_Publi_Tomo_Halcyon_DIBH/Tomo_FB_copy/pareto_0/RPxxxx.dcm \
      --predicted /home/oussama/Desktop/Project/raystation_exports/297768_all_paretos/297768_pareto_0_AI_SINO.dcm \
      --report /home/oussama/Desktop/Project/raystation_exports/297768_all_paretos/pareto_0_closed_row_report.csv

2) Inspect + repair:
    python inspect_repair_rtplan_closed_rows.py \
      --reference /mnt/data/shared/tomo_data/297768/RC_Publi_Tomo_Halcyon_DIBH/Tomo_FB_copy/pareto_0/RPxxxx.dcm \
      --predicted /home/oussama/Desktop/Project/raystation_exports/297768_all_paretos/297768_pareto_0_AI_SINO.dcm \
      --repair-output /home/oussama/Desktop/Project/raystation_exports/297768_all_paretos/297768_pareto_0_AI_SINO_CLOSED_FIX.dcm \
      --report /home/oussama/Desktop/Project/raystation_exports/297768_all_paretos/pareto_0_closed_row_report.csv

3) Also zero tiny values everywhere:
    add:
      --floor-threshold 0.001

Default behavior:
    Only mandatory fully-closed reference rows are forced to zero.
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.uid import generate_uid


SINO_TAG = (0x300D, 0x10A7)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def decode_sino_value(value: Any) -> np.ndarray | None:
    if isinstance(value, bytes):
        text = value.decode("ascii", errors="ignore")
    elif isinstance(value, bytearray):
        text = bytes(value).decode("ascii", errors="ignore")
    else:
        text = str(value)

    text = text.replace("\x00", "").strip()
    parts = [part.strip() for part in text.split("\\") if part.strip() != ""]

    try:
        row = np.asarray([float(part) for part in parts], dtype=np.float32)
    except ValueError:
        return None

    if row.size != 64:
        return None

    return row


def row_to_tomo_bytes(row: np.ndarray) -> bytes:
    row = np.asarray(row, dtype=np.float32).reshape(-1)

    if row.size != 64:
        raise ValueError(f"Expected 64 leaf values, got {row.size}.")

    parts = []
    for value in row:
        value = float(value)
        parts.append("0" if abs(value) < 1e-8 else f"{value:.7g}")

    raw = "\\".join(parts).encode("ascii")

    if len(raw) % 2 != 0:
        raw += b" "

    return raw


def get_sino_control_points(ds: pydicom.Dataset) -> list[pydicom.Dataset]:
    if not hasattr(ds, "BeamSequence") or len(ds.BeamSequence) == 0:
        raise ValueError("RTPLAN has no BeamSequence.")

    beam = ds.BeamSequence[0]

    if not hasattr(beam, "ControlPointSequence"):
        raise ValueError("First beam has no ControlPointSequence.")

    return [cp for cp in beam.ControlPointSequence if SINO_TAG in cp]


def get_cp_metadata(cp: pydicom.Dataset, cp_order: int) -> dict[str, Any]:
    def read_float(name: str) -> float | None:
        value = getattr(cp, name, None)
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def read_int(name: str, default: int) -> int:
        value = getattr(cp, name, None)
        if value is None:
            return default
        try:
            return int(value)
        except Exception:
            return default

    return {
        "cp_order": cp_order,
        "control_point_index": read_int("ControlPointIndex", cp_order),
        "gantry_angle": read_float("GantryAngle"),
        "table_longitudinal": read_float("TableTopLongitudinalPosition"),
        "table_lateral": read_float("TableTopLateralPosition"),
        "table_vertical": read_float("TableTopVerticalPosition"),
    }


def summarize_row(row: np.ndarray | None, zero_threshold: float) -> dict[str, Any]:
    if row is None:
        return {
            "valid": False,
            "sum": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "nonzero_count": np.nan,
            "all_zero": False,
        }

    abs_row = np.abs(row)
    return {
        "valid": True,
        "sum": float(np.sum(row)),
        "min": float(np.min(row)),
        "max": float(np.max(row)),
        "mean": float(np.mean(row)),
        "nonzero_count": int(np.count_nonzero(abs_row > zero_threshold)),
        "all_zero": bool(np.all(abs_row <= zero_threshold)),
    }


def inspect_plans(
    reference_path: Path,
    predicted_path: Path | None,
    report_path: Path | None,
    zero_threshold: float,
) -> list[dict[str, Any]]:
    ref_ds = pydicom.dcmread(str(reference_path), stop_before_pixels=True)
    ref_cps = get_sino_control_points(ref_ds)

    pred_cps = []
    if predicted_path is not None:
        pred_ds = pydicom.dcmread(str(predicted_path), stop_before_pixels=True)
        pred_cps = get_sino_control_points(pred_ds)

    n = len(ref_cps)

    if predicted_path is not None and len(pred_cps) != len(ref_cps):
        logging.warning(
            "Reference sinogram CP count=%d, predicted sinogram CP count=%d. Inspecting overlap only.",
            len(ref_cps),
            len(pred_cps),
        )
        n = min(len(ref_cps), len(pred_cps))

    rows: list[dict[str, Any]] = []

    for idx in range(n):
        ref_cp = ref_cps[idx]
        ref_row = decode_sino_value(ref_cp[SINO_TAG].value)
        ref_summary = summarize_row(ref_row, zero_threshold)

        row = {
            **get_cp_metadata(ref_cp, idx),
            "ref_valid": ref_summary["valid"],
            "ref_sum": ref_summary["sum"],
            "ref_min": ref_summary["min"],
            "ref_max": ref_summary["max"],
            "ref_mean": ref_summary["mean"],
            "ref_nonzero_count": ref_summary["nonzero_count"],
            "ref_all_zero": ref_summary["all_zero"],
        }

        if predicted_path is not None:
            pred_cp = pred_cps[idx]
            pred_row = decode_sino_value(pred_cp[SINO_TAG].value)
            pred_summary = summarize_row(pred_row, zero_threshold)

            row.update({
                "pred_valid": pred_summary["valid"],
                "pred_sum": pred_summary["sum"],
                "pred_min": pred_summary["min"],
                "pred_max": pred_summary["max"],
                "pred_mean": pred_summary["mean"],
                "pred_nonzero_count": pred_summary["nonzero_count"],
                "pred_all_zero": pred_summary["all_zero"],
                "required_closed_row_violation": bool(
                    ref_summary["all_zero"] and not pred_summary["all_zero"]
                ),
            })

        rows.append(row)

    if report_path is not None:
        write_report(report_path, rows)

    print_summary(rows, predicted_path is not None)
    return rows


def write_report(report_path: Path, rows: list[dict[str, Any]]) -> None:
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []

    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Saved CSV report: %s", report_path)


def print_summary(rows: list[dict[str, Any]], has_prediction: bool) -> None:
    ref_zero = sum(1 for row in rows if bool(row.get("ref_all_zero", False)))
    logging.info("Reference rows inspected: %d", len(rows))
    logging.info("Reference fully-closed rows: %d", ref_zero)

    if has_prediction:
        violations = [row for row in rows if bool(row.get("required_closed_row_violation", False))]
        logging.info("Predicted required-closed-row violations: %d", len(violations))

        if violations:
            preview = [int(row["cp_order"]) for row in violations[:30]]
            logging.warning("First violating CP rows: %s", preview)


def repair_predicted_plan(
    reference_path: Path,
    predicted_path: Path,
    output_path: Path,
    zero_threshold: float,
    floor_threshold: float,
) -> None:
    ref_ds = pydicom.dcmread(str(reference_path), stop_before_pixels=True)
    pred_ds = copy.deepcopy(pydicom.dcmread(str(predicted_path)))

    ref_cps = get_sino_control_points(ref_ds)
    pred_cps = get_sino_control_points(pred_ds)

    n = min(len(ref_cps), len(pred_cps))

    if len(ref_cps) != len(pred_cps):
        logging.warning(
            "Reference CP count=%d, predicted CP count=%d. Repairing overlap only.",
            len(ref_cps),
            len(pred_cps),
        )

    forced_zero_rows = 0
    floored_values = 0

    for idx in range(n):
        ref_row = decode_sino_value(ref_cps[idx][SINO_TAG].value)
        pred_row = decode_sino_value(pred_cps[idx][SINO_TAG].value)

        if ref_row is None or pred_row is None:
            logging.warning("Skipping invalid row at CP order %d.", idx)
            continue

        if np.all(np.abs(ref_row) <= zero_threshold):
            repaired = np.zeros((64,), dtype=np.float32)
            forced_zero_rows += 1
        else:
            repaired = np.asarray(pred_row, dtype=np.float32).copy()
            repaired = np.clip(repaired, 0.0, 1.0)

            if floor_threshold > 0.0:
                mask = repaired < floor_threshold
                floored_values += int(np.count_nonzero(mask))
                repaired[mask] = 0.0

        pred_cps[idx][SINO_TAG].value = row_to_tomo_bytes(repaired)

    now = datetime.now()
    pred_ds.SOPInstanceUID = generate_uid()
    pred_ds.SeriesInstanceUID = generate_uid()
    pred_ds.InstanceCreationDate = now.strftime("%Y%m%d")
    pred_ds.InstanceCreationTime = now.strftime("%H%M%S")
    pred_ds.SeriesDate = now.strftime("%Y%m%d")
    pred_ds.SeriesTime = now.strftime("%H%M%S")

    old_label = str(getattr(pred_ds, "RTPlanLabel", "PLAN"))
    old_name = str(getattr(pred_ds, "RTPlanName", "PLAN"))
    pred_ds.RTPlanLabel = clean_dicom_text(f"{old_label}_FIX", 16)
    pred_ds.RTPlanName = clean_dicom_text(f"{old_name} CLOSED_ROW_FIX", 64)

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_ds.save_as(str(output_path), write_like_original=False)

    logging.info("Saved repaired RTPLAN: %s", output_path)
    logging.info("Forced fully-closed reference rows to exact zero: %d", forced_zero_rows)
    logging.info("Floored tiny predicted values globally: %d", floored_values)


def clean_dicom_text(value: str, max_len: int) -> str:
    return value.replace("\n", " ").replace("\r", " ").strip()[:max_len]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and optionally repair Tomo/Radixact RTPLAN required closed MLC rows."
    )

    parser.add_argument("--reference", type=Path, required=True, help="Original reference RP*.dcm.")
    parser.add_argument("--predicted", type=Path, default=None, help="AI-generated predicted RP*.dcm.")
    parser.add_argument("--report", type=Path, default=None, help="Optional CSV report path.")
    parser.add_argument("--repair-output", type=Path, default=None, help="Optional repaired RTPLAN output.")

    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=1e-8,
        help="Threshold for considering a row fully closed. Default: 1e-8.",
    )
    parser.add_argument(
        "--floor-threshold",
        type=float,
        default=0.0,
        help="Optional global floor: predicted values below this become zero. Default: 0.0.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference = args.reference.expanduser().resolve()
    predicted = args.predicted.expanduser().resolve() if args.predicted is not None else None

    if not reference.exists():
        raise FileNotFoundError(f"Reference RTPLAN not found: {reference}")

    if predicted is not None and not predicted.exists():
        raise FileNotFoundError(f"Predicted RTPLAN not found: {predicted}")

    inspect_plans(
        reference_path=reference,
        predicted_path=predicted,
        report_path=args.report,
        zero_threshold=args.zero_threshold,
    )

    if args.repair_output is not None:
        if predicted is None:
            raise ValueError("--repair-output requires --predicted.")

        repair_predicted_plan(
            reference_path=reference,
            predicted_path=predicted,
            output_path=args.repair_output,
            zero_threshold=args.zero_threshold,
            floor_threshold=args.floor_threshold,
        )

        logging.info("Re-checking repaired RTPLAN.")
        inspect_plans(
            reference_path=reference,
            predicted_path=args.repair_output.expanduser().resolve(),
            report_path=None,
            zero_threshold=args.zero_threshold,
        )


if __name__ == "__main__":
    main()
