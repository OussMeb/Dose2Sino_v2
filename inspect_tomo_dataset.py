#!/usr/bin/env python3
"""
inspect_tomo_dataset.py

Audit a Tomo/Radixact breast RT dataset for CT + RTDose -> LOT sinogram work.

Expected layout:
    DATA_ROOT/
        PATIENT_ID/
            .../Tomo_FB_copy/
                CT*.dcm
                RS*.dcm
                pareto_*/
                    RP*.dcm
                    RD*.dcm
                    co.json

Outputs:
    audit_outputs/dataset_summary.json
    audit_outputs/plans.csv
    audit_outputs/patients.csv
    audit_outputs/problems.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


SINO_TAG = (0x300D, 0x10A7)
GANTRY_PERIOD_TAG = (0x300D, 0x1040)
TREATMENT_PITCH_TAG = (0x300D, 0x1060)
COUCH_SPEED_TAG = (0x300D, 0x1080)


@dataclass
class Problem:
    level: str
    patient_id: str
    pareto: str
    message: str


@dataclass
class PatientRow:
    patient_id: str
    tomo_folder: str
    ct_count: int
    rs_count: int
    pareto_count: int
    valid_plan_count: int
    problems_count: int


@dataclass
class PlanRow:
    patient_id: str
    pareto: str
    tomo_folder: str
    ct_count: int
    ct_rows: int | None = None
    ct_cols: int | None = None
    ct_spacing_row_mm: float | None = None
    ct_spacing_col_mm: float | None = None
    ct_dz_mm: float | None = None
    ct_z_min_mm: float | None = None
    ct_z_max_mm: float | None = None
    ct_origin_x_mm: float | None = None
    ct_origin_y_mm: float | None = None
    ct_origin_z_mm: float | None = None
    rs_file: str | None = None
    rs_roi_count: int | None = None
    rs_roi_names: str | None = None
    rp_file: str | None = None
    rd_file: str | None = None
    co_json_exists: bool = False
    dose_units: str | None = None
    dose_type: str | None = None
    dose_summation_type: str | None = None
    dose_rows: int | None = None
    dose_cols: int | None = None
    dose_frames: int | None = None
    dose_spacing_row_mm: float | None = None
    dose_spacing_col_mm: float | None = None
    dose_dz_mm: float | None = None
    dose_grid_scaling: float | None = None
    dose_min_gy: float | None = None
    dose_max_gy: float | None = None
    fractions_planned: int | None = None
    beam_meterset_minutes: float | None = None
    n_control_points_declared: int | None = None
    n_control_points_actual: int | None = None
    sino_found: bool = False
    sino_rows: int | None = None
    sino_cols: int | None = None
    sino_min: float | None = None
    sino_max: float | None = None
    sino_mean: float | None = None
    sino_open_fraction_gt_0: float | None = None
    gantry_min_deg: float | None = None
    gantry_max_deg: float | None = None
    gantry_unique_count: int | None = None
    table_min_mm: float | None = None
    table_max_mm: float | None = None
    table_span_mm: float | None = None
    cmw_min: float | None = None
    cmw_max: float | None = None
    cmw_monotonic: bool | None = None
    field_size_est_mm: float | None = None
    jaw_span_median_mm: float | None = None
    gantry_period_sec: float | None = None
    treatment_pitch: float | None = None
    couch_speed_mm_per_s: float | None = None


def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def read_dicom(path: Path, pixels: bool = False) -> pydicom.Dataset:
    return pydicom.dcmread(str(path), stop_before_pixels=not pixels, force=True)


def get_tag_float(ds: pydicom.Dataset, tag: tuple[int, int]) -> float | None:
    try:
        if tag in ds:
            return safe_float(ds[tag].value)
    except Exception:
        return None
    return None


def find_tomo_folders(patient_folder: Path) -> list[Path]:
    return sorted(p for p in patient_folder.glob("**/Tomo_FB_copy") if p.is_dir())


def extract_ct_info(ct_files: list[Path], problems: list[Problem], patient_id: str) -> dict[str, Any]:
    if not ct_files:
        problems.append(Problem("ERROR", patient_id, "-", "No CT*.dcm files found"))
        return {"ct_count": 0}

    metas: list[pydicom.Dataset] = []
    for path in ct_files:
        try:
            ds = read_dicom(path, pixels=False)
            if str(getattr(ds, "Modality", "")).upper() == "CT":
                metas.append(ds)
        except Exception as exc:
            problems.append(Problem("WARN", patient_id, "-", f"Cannot read CT header {path}: {exc}"))

    if not metas:
        problems.append(Problem("ERROR", patient_id, "-", "CT files exist but no readable CT modality found"))
        return {"ct_count": len(ct_files)}

    metas.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))

    first = metas[0]
    last = metas[-1]
    z_values = [float(ds.ImagePositionPatient[2]) for ds in metas]
    dz_values = np.diff(z_values) if len(z_values) > 1 else np.array([], dtype=np.float32)

    dz = float(np.median(dz_values)) if dz_values.size else safe_float(getattr(first, "SliceThickness", None))
    if dz_values.size and float(np.std(dz_values)) > 0.05:
        problems.append(Problem("WARN", patient_id, "-", f"Non-uniform CT dz, std={float(np.std(dz_values)):.4f} mm"))

    orientation = tuple(float(x) for x in getattr(first, "ImageOrientationPatient", []))
    if orientation and orientation != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0):
        problems.append(Problem("WARN", patient_id, "-", f"Non-standard CT orientation: {orientation}"))

    return {
        "ct_count": len(metas),
        "ct_rows": int(first.Rows),
        "ct_cols": int(first.Columns),
        "ct_spacing_row_mm": safe_float(first.PixelSpacing[0]),
        "ct_spacing_col_mm": safe_float(first.PixelSpacing[1]),
        "ct_dz_mm": dz,
        "ct_z_min_mm": float(min(z_values)),
        "ct_z_max_mm": float(max(z_values)),
        "ct_origin_x_mm": safe_float(first.ImagePositionPatient[0]),
        "ct_origin_y_mm": safe_float(first.ImagePositionPatient[1]),
        "ct_origin_z_mm": safe_float(first.ImagePositionPatient[2]),
    }


def extract_rs_info(rs_file: Path | None, problems: list[Problem], patient_id: str) -> dict[str, Any]:
    if rs_file is None:
        problems.append(Problem("WARN", patient_id, "-", "No RS*.dcm file found"))
        return {"rs_file": None, "rs_roi_count": 0, "rs_roi_names": ""}

    try:
        ds = read_dicom(rs_file, pixels=False)
        names = []
        if hasattr(ds, "StructureSetROISequence"):
            for roi in ds.StructureSetROISequence:
                names.append(str(getattr(roi, "ROIName", "")))
        return {
            "rs_file": str(rs_file),
            "rs_roi_count": len(names),
            "rs_roi_names": "|".join(names[:200]),
        }
    except Exception as exc:
        problems.append(Problem("WARN", patient_id, "-", f"Cannot read RS {rs_file}: {exc}"))
        return {"rs_file": str(rs_file), "rs_roi_count": None, "rs_roi_names": None}


def extract_dose_info(rd_file: Path, problems: list[Problem], patient_id: str, pareto: str, inspect_pixels: bool) -> dict[str, Any]:
    try:
        ds = read_dicom(rd_file, pixels=inspect_pixels)
    except Exception as exc:
        problems.append(Problem("ERROR", patient_id, pareto, f"Cannot read RD {rd_file}: {exc}"))
        return {"rd_file": str(rd_file)}

    dose_frames = int(getattr(ds, "NumberOfFrames", 1))
    dose_dz = None
    if hasattr(ds, "GridFrameOffsetVector"):
        offsets = np.asarray([float(x) for x in ds.GridFrameOffsetVector], dtype=np.float32)
        if offsets.size > 1:
            dose_dz = float(np.median(np.diff(offsets)))
            if float(np.std(np.diff(offsets))) > 0.05:
                problems.append(Problem("WARN", patient_id, pareto, "Non-uniform RD GridFrameOffsetVector spacing"))

    out = {
        "rd_file": str(rd_file),
        "dose_units": str(getattr(ds, "DoseUnits", "")) or None,
        "dose_type": str(getattr(ds, "DoseType", "")) or None,
        "dose_summation_type": str(getattr(ds, "DoseSummationType", "")) or None,
        "dose_rows": int(getattr(ds, "Rows", 0)) or None,
        "dose_cols": int(getattr(ds, "Columns", 0)) or None,
        "dose_frames": dose_frames,
        "dose_spacing_row_mm": safe_float(getattr(ds, "PixelSpacing", [None, None])[0]),
        "dose_spacing_col_mm": safe_float(getattr(ds, "PixelSpacing", [None, None])[1]),
        "dose_dz_mm": dose_dz,
        "dose_grid_scaling": safe_float(getattr(ds, "DoseGridScaling", None)),
    }

    if inspect_pixels:
        try:
            arr = ds.pixel_array.astype(np.float32)
            scaling = float(getattr(ds, "DoseGridScaling", 1.0))
            arr *= scaling
            out["dose_min_gy"] = float(np.min(arr))
            out["dose_max_gy"] = float(np.max(arr))
        except Exception as exc:
            problems.append(Problem("WARN", patient_id, pareto, f"Cannot inspect RD pixel data: {exc}"))

    return out


def extract_sino(cps: Any, problems: list[Problem], patient_id: str, pareto: str) -> np.ndarray | None:
    rows: list[list[float]] = []

    for index, cp in enumerate(cps):
        if SINO_TAG not in cp:
            continue

        value = cp[SINO_TAG].value
        if isinstance(value, (bytes, bytearray)):
            try:
                row = [float(x) for x in value.decode(errors="ignore").split("\\") if x.strip()]
            except Exception:
                problems.append(Problem("WARN", patient_id, pareto, f"Cannot decode sinogram row at CP {index}"))
                continue
        elif isinstance(value, str):
            row = [float(x) for x in value.split("\\") if x.strip()]
        else:
            try:
                row = [float(x) for x in value]
            except Exception:
                problems.append(Problem("WARN", patient_id, pareto, f"Unknown sinogram value type at CP {index}: {type(value)}"))
                continue

        if len(row) != 64:
            problems.append(Problem("WARN", patient_id, pareto, f"Sinogram row at CP {index} has {len(row)} values, expected 64"))
            continue

        rows.append(row)

    if not rows:
        problems.append(Problem("ERROR", patient_id, pareto, "No LOT sinogram found in private tag 300D,10A7"))
        return None

    return np.asarray(rows, dtype=np.float32)


def extract_jaw_spans(cps: Any) -> list[float]:
    spans: list[float] = []
    bl = (0x300A, 0x011A)
    lj = (0x300A, 0x011C)

    for cp in cps:
        if bl not in cp:
            continue

        best_span = None
        for dev in cp[bl]:
            if lj not in dev:
                continue
            values = [float(x) for x in dev[lj].value]
            if len(values) != 2:
                continue
            span = abs(values[1] - values[0])
            if span < 120.0 and (best_span is None or span > best_span):
                best_span = span

        if best_span is not None:
            spans.append(float(best_span))

    return spans


def estimate_field_size_mm(jaw_span_median_mm: float | None) -> float | None:
    if jaw_span_median_mm is None:
        return None
    if 4.0 <= jaw_span_median_mm <= 10.0:
        return 10.0
    if 14.0 <= jaw_span_median_mm <= 26.0:
        return 25.0
    if 32.0 <= jaw_span_median_mm <= 60.0:
        return 50.0
    return float(jaw_span_median_mm)


def extract_beam_meterset_minutes(ds: pydicom.Dataset) -> float | None:
    try:
        if "FractionGroupSequence" in ds:
            values = []
            for fg in ds.FractionGroupSequence:
                if "ReferencedBeamSequence" not in fg:
                    continue
                for ref_beam in fg.ReferencedBeamSequence:
                    if hasattr(ref_beam, "BeamMeterset"):
                        values.append(float(ref_beam.BeamMeterset))
            if values:
                return float(sum(values))
    except Exception:
        pass

    try:
        if "BeamSequence" in ds and len(ds.BeamSequence) > 0 and hasattr(ds.BeamSequence[0], "BeamMeterset"):
            return float(ds.BeamSequence[0].BeamMeterset)
    except Exception:
        pass

    return None


def extract_fraction_count(ds: pydicom.Dataset) -> int | None:
    try:
        if "FractionGroupSequence" not in ds:
            return None
        values = []
        for fg in ds.FractionGroupSequence:
            if hasattr(fg, "NumberOfFractionsPlanned"):
                values.append(int(fg.NumberOfFractionsPlanned))
        return int(sum(values)) if values else None
    except Exception:
        return None


def extract_plan_info(rp_file: Path, problems: list[Problem], patient_id: str, pareto: str) -> dict[str, Any]:
    try:
        ds = read_dicom(rp_file, pixels=False)
    except Exception as exc:
        problems.append(Problem("ERROR", patient_id, pareto, f"Cannot read RP {rp_file}: {exc}"))
        return {"rp_file": str(rp_file)}

    out: dict[str, Any] = {"rp_file": str(rp_file)}
    out["fractions_planned"] = extract_fraction_count(ds)
    out["beam_meterset_minutes"] = extract_beam_meterset_minutes(ds)

    try:
        beam = ds.BeamSequence[0]
        cps = beam.ControlPointSequence
    except Exception as exc:
        problems.append(Problem("ERROR", patient_id, pareto, f"Cannot access BeamSequence/ControlPointSequence: {exc}"))
        return out

    out["n_control_points_declared"] = safe_float(getattr(beam, "NumberOfControlPoints", None))
    if out["n_control_points_declared"] is not None:
        out["n_control_points_declared"] = int(out["n_control_points_declared"])
    out["n_control_points_actual"] = len(cps)

    sino = extract_sino(cps, problems, patient_id, pareto)
    if sino is not None:
        out.update({
            "sino_found": True,
            "sino_rows": int(sino.shape[0]),
            "sino_cols": int(sino.shape[1]),
            "sino_min": float(np.min(sino)),
            "sino_max": float(np.max(sino)),
            "sino_mean": float(np.mean(sino)),
            "sino_open_fraction_gt_0": float(np.mean(sino > 0)),
        })
        if out["n_control_points_actual"] != sino.shape[0]:
            problems.append(Problem("WARN", patient_id, pareto, f"CP count {len(cps)} != sinogram rows {sino.shape[0]}"))

    gantry = np.asarray([float(getattr(cp, "GantryAngle", 0.0)) for cp in cps], dtype=np.float32)
    table = np.asarray([float(getattr(cp, "TableTopLateralPosition", 0.0)) for cp in cps], dtype=np.float32)
    cmw = np.asarray([float(getattr(cp, "CumulativeMetersetWeight", 0.0)) for cp in cps], dtype=np.float32)

    out.update({
        "gantry_min_deg": float(np.min(gantry)) if gantry.size else None,
        "gantry_max_deg": float(np.max(gantry)) if gantry.size else None,
        "gantry_unique_count": int(len(np.unique(np.round(gantry, 3)))) if gantry.size else None,
        "table_min_mm": float(np.min(table)) if table.size else None,
        "table_max_mm": float(np.max(table)) if table.size else None,
        "table_span_mm": float(np.max(table) - np.min(table)) if table.size else None,
        "cmw_min": float(np.min(cmw)) if cmw.size else None,
        "cmw_max": float(np.max(cmw)) if cmw.size else None,
        "cmw_monotonic": bool(np.all(np.diff(cmw) >= -1e-6)) if cmw.size > 1 else None,
    })

    jaw_spans = extract_jaw_spans(cps)
    if jaw_spans:
        jaw_median = float(np.median(jaw_spans))
        out["jaw_span_median_mm"] = jaw_median
        out["field_size_est_mm"] = estimate_field_size_mm(jaw_median)

    try:
        beam0 = ds.BeamSequence[0]
    except Exception:
        beam0 = ds

    out["gantry_period_sec"] = get_tag_float(beam0, GANTRY_PERIOD_TAG) or get_tag_float(ds, GANTRY_PERIOD_TAG)
    out["treatment_pitch"] = get_tag_float(beam0, TREATMENT_PITCH_TAG) or get_tag_float(ds, TREATMENT_PITCH_TAG)
    out["couch_speed_mm_per_s"] = get_tag_float(beam0, COUCH_SPEED_TAG) or get_tag_float(ds, COUCH_SPEED_TAG)

    if out.get("sino_min") is not None and out["sino_min"] < -1e-6:
        problems.append(Problem("WARN", patient_id, pareto, f"Sinogram min < 0: {out['sino_min']}"))
    if out.get("sino_max") is not None and out["sino_max"] > 1.0 + 1e-6:
        problems.append(Problem("WARN", patient_id, pareto, f"Sinogram max > 1: {out['sino_max']}"))
    if out.get("cmw_monotonic") is False:
        problems.append(Problem("WARN", patient_id, pareto, "CumulativeMetersetWeight is not monotonic"))

    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_numeric(values: list[Any]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and not isinstance(v, bool)]
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
    }


def audit_dataset(root: Path, out_dir: Path, inspect_dose_pixels: bool) -> None:
    problems: list[Problem] = []
    patient_rows: list[PatientRow] = []
    plan_rows: list[PlanRow] = []

    patient_folders = sorted(p for p in root.iterdir() if p.is_dir())

    for patient_folder in patient_folders:
        patient_id = patient_folder.name
        patient_problem_start = len(problems)
        tomo_folders = find_tomo_folders(patient_folder)

        if not tomo_folders:
            problems.append(Problem("ERROR", patient_id, "-", "No Tomo_FB_copy folder found"))
            patient_rows.append(PatientRow(patient_id, "", 0, 0, 0, 0, len(problems) - patient_problem_start))
            continue

        for tomo_folder in tomo_folders:
            ct_files = sorted(tomo_folder.glob("CT*.dcm"))
            rs_files = sorted(tomo_folder.glob("RS*.dcm"))
            rs_file = rs_files[0] if rs_files else None
            pareto_folders = sorted(p for p in tomo_folder.glob("pareto_*") if p.is_dir())

            ct_info = extract_ct_info(ct_files, problems, patient_id)
            rs_info = extract_rs_info(rs_file, problems, patient_id)

            valid_plan_count = 0

            if not pareto_folders:
                problems.append(Problem("ERROR", patient_id, "-", f"No pareto_* folders in {tomo_folder}"))

            for pareto_folder in pareto_folders:
                pareto = pareto_folder.name
                rp_files = sorted(pareto_folder.glob("RP*.dcm"))
                rd_files = sorted(pareto_folder.glob("RD*.dcm"))

                if not rp_files:
                    problems.append(Problem("ERROR", patient_id, pareto, "No RP*.dcm found"))
                    continue
                if not rd_files:
                    problems.append(Problem("ERROR", patient_id, pareto, "No RD*.dcm found"))
                    continue

                row = PlanRow(
                    patient_id=patient_id,
                    pareto=pareto,
                    tomo_folder=str(tomo_folder),
                    ct_count=ct_info.get("ct_count", 0),
                    **{k: v for k, v in ct_info.items() if k != "ct_count"},
                    **rs_info,
                    co_json_exists=(pareto_folder / "co.json").exists(),
                )

                plan_info = extract_plan_info(rp_files[0], problems, patient_id, pareto)
                dose_info = extract_dose_info(rd_files[0], problems, patient_id, pareto, inspect_dose_pixels)

                for key, value in {**plan_info, **dose_info}.items():
                    if hasattr(row, key):
                        setattr(row, key, value)

                valid_plan_count += int(bool(row.sino_found and row.rd_file and row.rp_file))
                plan_rows.append(row)

            patient_rows.append(PatientRow(
                patient_id=patient_id,
                tomo_folder=str(tomo_folder),
                ct_count=len(ct_files),
                rs_count=len(rs_files),
                pareto_count=len(pareto_folders),
                valid_plan_count=valid_plan_count,
                problems_count=len(problems) - patient_problem_start,
            ))

    out_dir.mkdir(parents=True, exist_ok=True)

    plan_dicts = [asdict(row) for row in plan_rows]
    patient_dicts = [asdict(row) for row in patient_rows]
    problem_dicts = [asdict(problem) for problem in problems]

    write_csv(out_dir / "plans.csv", plan_dicts)
    write_csv(out_dir / "patients.csv", patient_dicts)

    (out_dir / "problems.txt").write_text(
        "\n".join(f"[{p.level}] patient={p.patient_id} pareto={p.pareto}: {p.message}" for p in problems),
        encoding="utf-8",
    )

    summary = {
        "root": str(root),
        "n_patient_folders": len(patient_folders),
        "n_patient_rows": len(patient_rows),
        "n_plan_rows": len(plan_rows),
        "n_valid_sino_plans": sum(1 for row in plan_rows if row.sino_found),
        "n_problems": len(problems),
        "problem_counts_by_level": {
            level: sum(1 for p in problems if p.level == level)
            for level in sorted({p.level for p in problems})
        },
        "sino_rows": summarize_numeric([row.sino_rows for row in plan_rows]),
        "sino_open_fraction_gt_0": summarize_numeric([row.sino_open_fraction_gt_0 for row in plan_rows]),
        "beam_meterset_minutes": summarize_numeric([row.beam_meterset_minutes for row in plan_rows]),
        "table_span_mm": summarize_numeric([row.table_span_mm for row in plan_rows]),
        "field_size_est_mm_values": sorted({row.field_size_est_mm for row in plan_rows if row.field_size_est_mm is not None}),
        "dose_max_gy": summarize_numeric([row.dose_max_gy for row in plan_rows]),
        "outputs": {
            "plans_csv": str(out_dir / "plans.csv"),
            "patients_csv": str(out_dir / "patients.csv"),
            "problems_txt": str(out_dir / "problems.txt"),
        },
    }

    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Tomo dataset for CT + RTDose -> LOT sinogram work.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root, e.g. /mnt/LeGrosDisque/oussama/tomo_data",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("audit_outputs"),
        help="Output folder for CSV/JSON reports.",
    )
    parser.add_argument(
        "--inspect-dose-pixels",
        action="store_true",
        help="Read RD pixel arrays to report dose min/max. Slower but useful.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    audit_dataset(root=root, out_dir=out_dir, inspect_dose_pixels=args.inspect_dose_pixels)


if __name__ == "__main__":
    main()
