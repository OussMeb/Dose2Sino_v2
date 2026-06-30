#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_rtplan_technical_diff.py

Compare a reference Tomo/Radixact RTPLAN with an AI-generated RTPLAN.

Goal:
    Identify technical parameters that differ between the GT RTPLAN and the
    predicted RTPLAN, especially parameters that can make RayStation reject
    the file.

The injector SHOULD only change:
    - Tomo LOT sinogram private tag (300D,10A7)
    - UIDs / dates / labels / description

Everything else should remain identical:
    - FractionGroupSequence
    - BeamSequence
    - beam meterset / dose / machine metadata
    - ControlPointSequence count
    - CP index
    - CumulativeMetersetWeight
    - gantry/couch/table positions
    - BeamLimitingDevicePositionSequence
    - private timing/control tags except the sinogram tag

Usage:
    python inspect_rtplan_technical_diff.py \
      --reference /path/to/original/RPxxxx.dcm \
      --predicted /path/to/AI_SINO_CLOSED_FIX.dcm \
      --output-dir /path/to/technical_diff_pareto_0

Outputs:
    summary.json
    top_level_diff.csv
    fraction_group_diff.csv
    beam_diff.csv
    control_point_diff.csv
    beam_limiting_device_position_diff.csv
    private_control_point_diff.csv
    sinogram_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.datadict import keyword_for_tag


SINO_TAG = (0x300D, 0x10A7)

EXPECTED_ALLOWED_DIFF_KEYWORDS = {
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "InstanceCreationDate",
    "InstanceCreationTime",
    "SeriesDate",
    "SeriesTime",
    "RTPlanLabel",
    "RTPlanName",
    "PlanDescription",
}

TECHNICAL_TOP_LEVEL_KEYWORDS = [
    "SOPClassUID",
    "Modality",
    "PatientID",
    "PatientName",
    "StudyInstanceUID",
    "FrameOfReferenceUID",
    "StudyDate",
    "StudyTime",
    "RTPlanDate",
    "RTPlanTime",
    "RTPlanGeometry",
    "PrescriptionDescription",
    "ApprovalStatus",
]

TECHNICAL_FRACTION_GROUP_KEYWORDS = [
    "FractionGroupNumber",
    "NumberOfFractionsPlanned",
    "NumberOfBeams",
    "NumberOfBrachyApplicationSetups",
]

TECHNICAL_REFERENCED_BEAM_KEYWORDS = [
    "ReferencedBeamNumber",
    "BeamMeterset",
    "BeamDose",
    "BeamDosePointDepth",
    "BeamDosePointEquivalentDepth",
    "BeamDosePointSSD",
]

TECHNICAL_BEAM_KEYWORDS = [
    "BeamNumber",
    "BeamName",
    "BeamType",
    "RadiationType",
    "TreatmentDeliveryType",
    "TreatmentMachineName",
    "Manufacturer",
    "InstitutionName",
    "PrimaryDosimeterUnit",
    "SourceAxisDistance",
    "FinalCumulativeMetersetWeight",
    "NumberOfControlPoints",
    "NumberOfWedges",
    "NumberOfCompensators",
    "NumberOfBoli",
    "NumberOfBlocks",
    "NumberOfRangeShifters",
    "NumberOfLateralSpreadingDevices",
    "NumberOfRangeModulators",
]

TECHNICAL_CP_KEYWORDS = [
    "ControlPointIndex",
    "CumulativeMetersetWeight",
    "NominalBeamEnergy",
    "DoseRateSet",
    "GantryAngle",
    "GantryRotationDirection",
    "BeamLimitingDeviceAngle",
    "BeamLimitingDeviceRotationDirection",
    "PatientSupportAngle",
    "PatientSupportRotationDirection",
    "TableTopEccentricAngle",
    "TableTopEccentricRotationDirection",
    "TableTopVerticalPosition",
    "TableTopLongitudinalPosition",
    "TableTopLateralPosition",
    "IsocenterPosition",
    "SourceToSurfaceDistance",
    "TableTopPitchAngle",
    "TableTopPitchRotationDirection",
    "TableTopRollAngle",
    "TableTopRollRotationDirection",
]


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def tag_name(tag: pydicom.tag.BaseTag) -> str:
    keyword = keyword_for_tag(tag)
    return keyword if keyword else str(tag)


def normalize_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"

    if isinstance(value, bytearray):
        return f"<bytes:{len(value)}>"

    if isinstance(value, pydicom.multival.MultiValue):
        return [normalize_value(v) for v in value]

    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]

    if hasattr(value, "original_string"):
        try:
            return float(value)
        except Exception:
            return str(value)

    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    return str(value)


def values_equal(ref_value: Any, pred_value: Any, atol: float) -> bool:
    ref_norm = normalize_value(ref_value)
    pred_norm = normalize_value(pred_value)

    if ref_norm == pred_norm:
        return True

    try:
        ref_arr = np.asarray(ref_norm, dtype=float)
        pred_arr = np.asarray(pred_norm, dtype=float)
        return bool(np.allclose(ref_arr, pred_arr, atol=atol, rtol=0.0))
    except Exception:
        return False


def value_to_string(value: Any) -> str:
    normalized = normalize_value(value)
    if isinstance(normalized, (dict, list, tuple)):
        return json.dumps(normalized, ensure_ascii=False)
    return str(normalized)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted(set().union(*(row.keys() for row in rows)))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_sequence(ds: pydicom.Dataset, keyword: str) -> list[pydicom.Dataset]:
    seq = getattr(ds, keyword, None)
    if seq is None:
        return []
    return list(seq)


def get_first_beam(ds: pydicom.Dataset) -> pydicom.Dataset:
    beams = get_sequence(ds, "BeamSequence")
    if not beams:
        raise ValueError("RTPLAN has no BeamSequence.")
    return beams[0]


def get_control_points(beam: pydicom.Dataset) -> list[pydicom.Dataset]:
    cps = get_sequence(beam, "ControlPointSequence")
    if not cps:
        raise ValueError("Beam has no ControlPointSequence.")
    return cps


def compare_keywords(
    ref_obj: pydicom.Dataset,
    pred_obj: pydicom.Dataset,
    keywords: list[str],
    scope: str,
    index: int | None,
    atol: float,
    allowed_keywords: set[str] | None = None,
) -> list[dict[str, Any]]:
    allowed_keywords = allowed_keywords or set()
    rows = []

    for keyword in keywords:
        ref_value = getattr(ref_obj, keyword, None)
        pred_value = getattr(pred_obj, keyword, None)

        if not values_equal(ref_value, pred_value, atol):
            rows.append({
                "scope": scope,
                "index": index,
                "keyword": keyword,
                "reference_value": value_to_string(ref_value),
                "predicted_value": value_to_string(pred_value),
                "allowed_difference": keyword in allowed_keywords,
            })

    return rows


def compare_all_non_sequence_elements(
    ref_obj: pydicom.Dataset,
    pred_obj: pydicom.Dataset,
    scope: str,
    index: int | None,
    atol: float,
    allowed_keywords: set[str] | None = None,
    excluded_tags: set[pydicom.tag.BaseTag] | None = None,
) -> list[dict[str, Any]]:
    allowed_keywords = allowed_keywords or set()
    excluded_tags = excluded_tags or set()

    rows = []

    ref_map = {elem.tag: elem for elem in ref_obj if elem.VR != "SQ" and elem.tag not in excluded_tags}
    pred_map = {elem.tag: elem for elem in pred_obj if elem.VR != "SQ" and elem.tag not in excluded_tags}

    for tag in sorted(set(ref_map) | set(pred_map)):
        ref_elem = ref_map.get(tag)
        pred_elem = pred_map.get(tag)

        ref_value = ref_elem.value if ref_elem is not None else None
        pred_value = pred_elem.value if pred_elem is not None else None

        if not values_equal(ref_value, pred_value, atol):
            keyword = tag_name(tag)
            rows.append({
                "scope": scope,
                "index": index,
                "tag": str(tag),
                "keyword": keyword,
                "vr_reference": ref_elem.VR if ref_elem is not None else None,
                "vr_predicted": pred_elem.VR if pred_elem is not None else None,
                "reference_value": value_to_string(ref_value),
                "predicted_value": value_to_string(pred_value),
                "allowed_difference": keyword in allowed_keywords,
            })

    return rows


def compare_top_level(ref_ds: pydicom.Dataset, pred_ds: pydicom.Dataset, atol: float) -> list[dict[str, Any]]:
    rows = []

    rows.extend(
        compare_keywords(
            ref_ds,
            pred_ds,
            TECHNICAL_TOP_LEVEL_KEYWORDS + sorted(EXPECTED_ALLOWED_DIFF_KEYWORDS),
            scope="top_level_named",
            index=None,
            atol=atol,
            allowed_keywords=EXPECTED_ALLOWED_DIFF_KEYWORDS,
        )
    )

    rows.extend(
        compare_all_non_sequence_elements(
            ref_ds,
            pred_ds,
            scope="top_level_all_non_sequence",
            index=None,
            atol=atol,
            allowed_keywords=EXPECTED_ALLOWED_DIFF_KEYWORDS,
        )
    )

    return deduplicate_rows(rows)


def compare_fraction_groups(ref_ds: pydicom.Dataset, pred_ds: pydicom.Dataset, atol: float) -> list[dict[str, Any]]:
    ref_groups = get_sequence(ref_ds, "FractionGroupSequence")
    pred_groups = get_sequence(pred_ds, "FractionGroupSequence")

    rows = []

    if len(ref_groups) != len(pred_groups):
        rows.append({
            "scope": "FractionGroupSequence",
            "index": None,
            "keyword": "length",
            "reference_value": len(ref_groups),
            "predicted_value": len(pred_groups),
            "allowed_difference": False,
        })

    for fg_idx in range(min(len(ref_groups), len(pred_groups))):
        ref_fg = ref_groups[fg_idx]
        pred_fg = pred_groups[fg_idx]

        rows.extend(
            compare_keywords(
                ref_fg,
                pred_fg,
                TECHNICAL_FRACTION_GROUP_KEYWORDS,
                scope="fraction_group",
                index=fg_idx,
                atol=atol,
            )
        )

        rows.extend(
            compare_all_non_sequence_elements(
                ref_fg,
                pred_fg,
                scope="fraction_group_all_non_sequence",
                index=fg_idx,
                atol=atol,
            )
        )

        ref_refs = get_sequence(ref_fg, "ReferencedBeamSequence")
        pred_refs = get_sequence(pred_fg, "ReferencedBeamSequence")

        if len(ref_refs) != len(pred_refs):
            rows.append({
                "scope": "ReferencedBeamSequence",
                "index": fg_idx,
                "keyword": "length",
                "reference_value": len(ref_refs),
                "predicted_value": len(pred_refs),
                "allowed_difference": False,
            })

        for rb_idx in range(min(len(ref_refs), len(pred_refs))):
            rows.extend(
                compare_keywords(
                    ref_refs[rb_idx],
                    pred_refs[rb_idx],
                    TECHNICAL_REFERENCED_BEAM_KEYWORDS,
                    scope=f"fraction_group_{fg_idx}_referenced_beam",
                    index=rb_idx,
                    atol=atol,
                )
            )
            rows.extend(
                compare_all_non_sequence_elements(
                    ref_refs[rb_idx],
                    pred_refs[rb_idx],
                    scope=f"fraction_group_{fg_idx}_referenced_beam_all_non_sequence",
                    index=rb_idx,
                    atol=atol,
                )
            )

    return deduplicate_rows(rows)


def compare_beams(ref_ds: pydicom.Dataset, pred_ds: pydicom.Dataset, atol: float) -> list[dict[str, Any]]:
    ref_beams = get_sequence(ref_ds, "BeamSequence")
    pred_beams = get_sequence(pred_ds, "BeamSequence")

    rows = []

    if len(ref_beams) != len(pred_beams):
        rows.append({
            "scope": "BeamSequence",
            "index": None,
            "keyword": "length",
            "reference_value": len(ref_beams),
            "predicted_value": len(pred_beams),
            "allowed_difference": False,
        })

    for beam_idx in range(min(len(ref_beams), len(pred_beams))):
        ref_beam = ref_beams[beam_idx]
        pred_beam = pred_beams[beam_idx]

        rows.extend(
            compare_keywords(
                ref_beam,
                pred_beam,
                TECHNICAL_BEAM_KEYWORDS,
                scope="beam",
                index=beam_idx,
                atol=atol,
            )
        )

        rows.extend(
            compare_all_non_sequence_elements(
                ref_beam,
                pred_beam,
                scope="beam_all_non_sequence",
                index=beam_idx,
                atol=atol,
            )
        )

    return deduplicate_rows(rows)


def compare_control_points(ref_ds: pydicom.Dataset, pred_ds: pydicom.Dataset, atol: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ref_beam = get_first_beam(ref_ds)
    pred_beam = get_first_beam(pred_ds)

    ref_cps = get_control_points(ref_beam)
    pred_cps = get_control_points(pred_beam)

    cp_rows = []
    bld_rows = []
    private_rows = []

    if len(ref_cps) != len(pred_cps):
        cp_rows.append({
            "scope": "ControlPointSequence",
            "cp_index": None,
            "keyword": "length",
            "reference_value": len(ref_cps),
            "predicted_value": len(pred_cps),
            "allowed_difference": False,
        })

    n = min(len(ref_cps), len(pred_cps))

    for cp_idx in range(n):
        ref_cp = ref_cps[cp_idx]
        pred_cp = pred_cps[cp_idx]

        cp_rows.extend(
            add_cp_index(
                compare_keywords(
                    ref_cp,
                    pred_cp,
                    TECHNICAL_CP_KEYWORDS,
                    scope="control_point_named",
                    index=cp_idx,
                    atol=atol,
                ),
                cp_idx,
            )
        )

        cp_rows.extend(
            add_cp_index(
                compare_all_non_sequence_elements(
                    ref_cp,
                    pred_cp,
                    scope="control_point_all_non_sequence",
                    index=cp_idx,
                    atol=atol,
                    excluded_tags={SINO_TAG},
                ),
                cp_idx,
            )
        )

        private_rows.extend(compare_private_cp_tags(ref_cp, pred_cp, cp_idx, atol))

        bld_rows.extend(compare_bld_sequence(ref_cp, pred_cp, cp_idx, atol))

    return deduplicate_rows(cp_rows), deduplicate_rows(bld_rows), deduplicate_rows(private_rows)


def compare_private_cp_tags(
    ref_cp: pydicom.Dataset,
    pred_cp: pydicom.Dataset,
    cp_idx: int,
    atol: float,
) -> list[dict[str, Any]]:
    rows = []

    ref_private = {
        elem.tag: elem
        for elem in ref_cp
        if elem.tag.is_private and elem.tag != SINO_TAG and elem.VR != "SQ"
    }
    pred_private = {
        elem.tag: elem
        for elem in pred_cp
        if elem.tag.is_private and elem.tag != SINO_TAG and elem.VR != "SQ"
    }

    for tag in sorted(set(ref_private) | set(pred_private)):
        ref_elem = ref_private.get(tag)
        pred_elem = pred_private.get(tag)

        ref_value = ref_elem.value if ref_elem is not None else None
        pred_value = pred_elem.value if pred_elem is not None else None

        if not values_equal(ref_value, pred_value, atol):
            rows.append({
                "cp_index": cp_idx,
                "tag": str(tag),
                "keyword": tag_name(tag),
                "vr_reference": ref_elem.VR if ref_elem is not None else None,
                "vr_predicted": pred_elem.VR if pred_elem is not None else None,
                "reference_value": value_to_string(ref_value),
                "predicted_value": value_to_string(pred_value),
                "allowed_difference": False,
            })

    return rows


def compare_bld_sequence(
    ref_cp: pydicom.Dataset,
    pred_cp: pydicom.Dataset,
    cp_idx: int,
    atol: float,
) -> list[dict[str, Any]]:
    rows = []

    ref_seq = get_sequence(ref_cp, "BeamLimitingDevicePositionSequence")
    pred_seq = get_sequence(pred_cp, "BeamLimitingDevicePositionSequence")

    if len(ref_seq) != len(pred_seq):
        rows.append({
            "cp_index": cp_idx,
            "bld_index": None,
            "keyword": "BeamLimitingDevicePositionSequenceLength",
            "reference_value": len(ref_seq),
            "predicted_value": len(pred_seq),
            "allowed_difference": False,
        })

    for bld_idx in range(min(len(ref_seq), len(pred_seq))):
        ref_bld = ref_seq[bld_idx]
        pred_bld = pred_seq[bld_idx]

        rows.extend(
            add_bld_index(
                add_cp_index(
                    compare_all_non_sequence_elements(
                        ref_bld,
                        pred_bld,
                        scope="beam_limiting_device_position",
                        index=bld_idx,
                        atol=atol,
                    ),
                    cp_idx,
                ),
                bld_idx,
            )
        )

    return rows


def decode_sino_value(value: Any) -> np.ndarray | None:
    if isinstance(value, bytes):
        text = value.decode("ascii", errors="ignore")
    elif isinstance(value, bytearray):
        text = bytes(value).decode("ascii", errors="ignore")
    else:
        text = str(value)

    text = text.replace("\x00", "").strip()
    parts = [p.strip() for p in text.split("\\") if p.strip()]

    try:
        arr = np.asarray([float(p) for p in parts], dtype=np.float32)
    except Exception:
        return None

    if arr.size != 64:
        return None

    return arr


def inspect_sinograms(ref_ds: pydicom.Dataset, pred_ds: pydicom.Dataset, zero_threshold: float) -> list[dict[str, Any]]:
    ref_cp = get_control_points(get_first_beam(ref_ds))
    pred_cp = get_control_points(get_first_beam(pred_ds))

    rows = []
    n = min(len(ref_cp), len(pred_cp))

    for cp_idx in range(n):
        ref_has = SINO_TAG in ref_cp[cp_idx]
        pred_has = SINO_TAG in pred_cp[cp_idx]

        if not ref_has and not pred_has:
            continue

        ref_row = decode_sino_value(ref_cp[cp_idx][SINO_TAG].value) if ref_has else None
        pred_row = decode_sino_value(pred_cp[cp_idx][SINO_TAG].value) if pred_has else None

        ref_stats = sino_stats(ref_row, zero_threshold)
        pred_stats = sino_stats(pred_row, zero_threshold)

        row = {
            "cp_index": cp_idx,
            "ref_has_sino": ref_has,
            "pred_has_sino": pred_has,
            "ref_valid": ref_stats["valid"],
            "pred_valid": pred_stats["valid"],
            "ref_all_zero": ref_stats["all_zero"],
            "pred_all_zero": pred_stats["all_zero"],
            "required_closed_row_violation": bool(ref_stats["all_zero"] and not pred_stats["all_zero"]),
            "ref_nonzero_count": ref_stats["nonzero_count"],
            "pred_nonzero_count": pred_stats["nonzero_count"],
            "ref_sum": ref_stats["sum"],
            "pred_sum": pred_stats["sum"],
            "ref_min": ref_stats["min"],
            "pred_min": pred_stats["min"],
            "ref_max": ref_stats["max"],
            "pred_max": pred_stats["max"],
            "ref_mean": ref_stats["mean"],
            "pred_mean": pred_stats["mean"],
        }

        if ref_row is not None and pred_row is not None:
            row["mae"] = float(np.mean(np.abs(ref_row - pred_row)))
            row["max_abs_diff"] = float(np.max(np.abs(ref_row - pred_row)))

        rows.append(row)

    return rows


def sino_stats(row: np.ndarray | None, zero_threshold: float) -> dict[str, Any]:
    if row is None:
        return {
            "valid": False,
            "all_zero": False,
            "nonzero_count": None,
            "sum": None,
            "min": None,
            "max": None,
            "mean": None,
        }

    abs_row = np.abs(row)

    return {
        "valid": True,
        "all_zero": bool(np.all(abs_row <= zero_threshold)),
        "nonzero_count": int(np.count_nonzero(abs_row > zero_threshold)),
        "sum": float(np.sum(row)),
        "min": float(np.min(row)),
        "max": float(np.max(row)),
        "mean": float(np.mean(row)),
    }


def add_cp_index(rows: list[dict[str, Any]], cp_idx: int) -> list[dict[str, Any]]:
    for row in rows:
        row["cp_index"] = cp_idx
    return rows


def add_bld_index(rows: list[dict[str, Any]], bld_idx: int) -> list[dict[str, Any]]:
    for row in rows:
        row["bld_index"] = bld_idx
    return rows


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []

    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    return out


def build_summary(
    top_rows: list[dict[str, Any]],
    fraction_rows: list[dict[str, Any]],
    beam_rows: list[dict[str, Any]],
    cp_rows: list[dict[str, Any]],
    bld_rows: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
    sino_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    unexpected_top = [r for r in top_rows if not r.get("allowed_difference", False)]
    unexpected_fraction = [r for r in fraction_rows if not r.get("allowed_difference", False)]
    unexpected_beam = [r for r in beam_rows if not r.get("allowed_difference", False)]
    unexpected_cp = [r for r in cp_rows if not r.get("allowed_difference", False)]
    unexpected_bld = [r for r in bld_rows if not r.get("allowed_difference", False)]
    unexpected_private = [r for r in private_rows if not r.get("allowed_difference", False)]
    closed_violations = [r for r in sino_rows if r.get("required_closed_row_violation", False)]

    return {
        "top_level_diff_count": len(top_rows),
        "unexpected_top_level_diff_count": len(unexpected_top),
        "fraction_group_diff_count": len(fraction_rows),
        "unexpected_fraction_group_diff_count": len(unexpected_fraction),
        "beam_diff_count": len(beam_rows),
        "unexpected_beam_diff_count": len(unexpected_beam),
        "control_point_diff_count": len(cp_rows),
        "unexpected_control_point_diff_count": len(unexpected_cp),
        "beam_limiting_device_position_diff_count": len(bld_rows),
        "unexpected_beam_limiting_device_position_diff_count": len(unexpected_bld),
        "private_control_point_diff_count": len(private_rows),
        "unexpected_private_control_point_diff_count": len(unexpected_private),
        "sinogram_rows_count": len(sino_rows),
        "required_closed_row_violation_count": len(closed_violations),
        "first_closed_row_violations": [r["cp_index"] for r in closed_violations[:30]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare technical RTPLAN parameters between GT and AI RTPLAN.")

    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--zero-threshold", type=float, default=1e-8)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference_path = args.reference.expanduser().resolve()
    predicted_path = args.predicted.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference RTPLAN not found: {reference_path}")
    if not predicted_path.exists():
        raise FileNotFoundError(f"Predicted RTPLAN not found: {predicted_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    ref_ds = pydicom.dcmread(str(reference_path), stop_before_pixels=True)
    pred_ds = pydicom.dcmread(str(predicted_path), stop_before_pixels=True)

    top_rows = compare_top_level(ref_ds, pred_ds, args.atol)
    fraction_rows = compare_fraction_groups(ref_ds, pred_ds, args.atol)
    beam_rows = compare_beams(ref_ds, pred_ds, args.atol)
    cp_rows, bld_rows, private_rows = compare_control_points(ref_ds, pred_ds, args.atol)
    sino_rows = inspect_sinograms(ref_ds, pred_ds, args.zero_threshold)

    write_csv(output_dir / "top_level_diff.csv", top_rows)
    write_csv(output_dir / "fraction_group_diff.csv", fraction_rows)
    write_csv(output_dir / "beam_diff.csv", beam_rows)
    write_csv(output_dir / "control_point_diff.csv", cp_rows)
    write_csv(output_dir / "beam_limiting_device_position_diff.csv", bld_rows)
    write_csv(output_dir / "private_control_point_diff.csv", private_rows)
    write_csv(output_dir / "sinogram_stats.csv", sino_rows)

    summary = build_summary(
        top_rows=top_rows,
        fraction_rows=fraction_rows,
        beam_rows=beam_rows,
        cp_rows=cp_rows,
        bld_rows=bld_rows,
        private_rows=private_rows,
        sino_rows=sino_rows,
    )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logging.info("Saved technical diff report in: %s", output_dir)
    logging.info("Summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
