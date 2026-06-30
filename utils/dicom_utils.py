#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DICOM and RT-PLAN utilities for TomoTherapy/Radixact.
"""
import os
import logging
import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
from typing import Dict, Any


def find_rtplan_file(dicom_dir: Path) -> Path:
    """
    Find and return the first DICOM file with Modality=='RTPLAN'.

    If associations.json exists, use it to locate RP directly.
    Otherwise, search recursively in dicom_dir and parent/sibling directories
    (to handle _0, _1, etc. variants).
    """
    dicom_dir = Path(dicom_dir)

    # Try to use associations.json if available
    # Extract base patient ID if dicom_dir has _0, _1 suffix
    base_id = dicom_dir.name.rsplit("_", 1)[0] if "_" in dicom_dir.name else dicom_dir.name
    parent_dicom = dicom_dir.parent

    # Check for associations.json in data root
    data_root = Path("/mnt/LeGrosDisque/Julien/sianogramme/data")
    associations_path = data_root / base_id / "associations.json"

    if associations_path.exists():
        try:
            import json
            with open(associations_path, 'r', encoding='utf-8') as f:
                assoc = json.load(f)

            # Extract first RP file from associations
            for group in assoc.get('groups', []):
                for rp_info in group.get('rp', []):
                    rp_rel_path = rp_info.get('rp_path', '')
                    rp_full_path = parent_dicom / rp_rel_path
                    if rp_full_path.exists():
                        logging.debug(f"RT-PLAN found via associations: {rp_full_path}")
                        return rp_full_path
        except Exception as e:
            logging.debug(f"Could not use associations.json: {e}, falling back to search")

    # Fallback: Search in dicom_dir and sibling _0, _1 directories
    search_dirs = [dicom_dir]

    # Add sibling directories (340980_0, 340980_1, etc.)
    if "_" in dicom_dir.name:
        base = dicom_dir.name.rsplit("_", 1)[0]
        for sibling in parent_dicom.iterdir():
            if sibling.is_dir() and sibling.name.startswith(base + "_"):
                search_dirs.append(sibling)

    # Search in all directories
    for search_dir in search_dirs:
        for root, _, files in os.walk(search_dir):
            for f in files:
                try:
                    ds = pydicom.dcmread(str(Path(root) / f), stop_before_pixels=True, force=True)
                    if ds.Modality.upper() == "RTPLAN":
                        logging.debug(f"RT-PLAN found: {root}/{f}")
                        return Path(root) / f
                except Exception:
                    continue

    raise FileNotFoundError(f"No RT-PLAN found in {dicom_dir} or sibling directories")


def get_number_of_fractions_planned(rtplan_path: str) -> int | None:
    """
    Return NumberOfFractionsPlanned (300A,0078) if available.
    If multiple groups, return the sum.
    """
    ds = pydicom.dcmread(rtplan_path, stop_before_pixels=True)

    if not hasattr(ds, "FractionGroupSequence"):
        return None

    n_list = []
    for fg in ds.FractionGroupSequence:
        if hasattr(fg, "NumberOfFractionsPlanned"):
            try:
                n_list.append(int(fg.NumberOfFractionsPlanned))
            except Exception:
                pass

    if len(n_list) == 0:
        return None

    return int(sum(n_list))


def extract_beam_meterset_minutes(ds: pydicom.Dataset) -> float:
    """
    Tomo/Radixact: Beam Meterset (planned time in minutes) is found in
    FractionGroupSequence / ReferencedBeamSequence / BeamMeterset (300A,0086).
    Fallback to BeamSequence[0].BeamMeterset if present.
    """
    # 1) Canonical DICOM path for meterset per beam
    try:
        if "FractionGroupSequence" in ds:
            fgs = ds.FractionGroupSequence[0]
            if "ReferencedBeamSequence" in fgs and len(fgs.ReferencedBeamSequence) > 0:
                refb = fgs.ReferencedBeamSequence[0]
                if hasattr(refb, "BeamMeterset"):
                    return float(refb.BeamMeterset)
    except Exception:
        pass

    # 2) Direct fallback (rarely filled for Tomo/Radixact)
    try:
        if "BeamSequence" in ds and len(ds.BeamSequence) > 0:
            beam = ds.BeamSequence[0]
            if hasattr(beam, "BeamMeterset"):
                return float(beam.BeamMeterset)
    except Exception:
        pass

    return 0.0


def extract_sinogram(cps) -> np.ndarray:
    """
    Extract sinogram from ControlPointSequence for TomoTherapy.
    Returns normalized sinogram [N_CP, 64].
    """
    lines = []
    for cp in cps:
        tag = (0x300D, 0x10A7)  # Tomo: Private tag "Sinogram Data"
        if tag in cp:
            val = cp[tag].value
            if isinstance(val, (bytes, bytearray)):
                parts = val.decode(errors="ignore").split("\\")
                row = [float(x) for x in parts if x.strip() != ""]
                if len(row) == 64:
                    lines.append(row)

    if not lines:
        raise RuntimeError("Sinogram not found in RT-PLAN (private tag 300D,10A7 missing).")

    return np.array(lines, dtype=np.float32)  # ∈[0,1]


def extract_jaw_positions(cps) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract (jaw1_mm, jaw2_mm, device_types) for Y-pair (longitudinal) if possible.
    Heuristic:
      1) prioritize devices whose name/type suggests Y-axis
      2) fallback: pair with largest opening (span), under reasonable threshold
    """
    jaw1, jaw2, dtypes = [], [], []
    bl = (0x300A, 0x011A)  # BeamLimitingDevicePositionSequence
    lj = (0x300A, 0x011C)  # LeafJawPositions

    pref_long_names = {"asymy", "jawy", "yjaw", "mlcy", "jaws-y", "jaws_y", "tomo_y"}

    for cp in cps:
        if bl not in cp:
            jaw1.append(np.nan)
            jaw2.append(np.nan)
            dtypes.append("none")
            continue

        chosen = None
        chosen_name = None
        max_span = -1.0

        # Try Y-axis specific devices first
        for dev in cp[bl]:
            name = str(getattr(dev, "RTBeamLimitingDeviceType", "") or
                      getattr(dev, "ManufacturerDeviceIdentifier", "")).lower()
            if lj in dev:
                vals = list(map(float, dev[lj].value))
                if len(vals) == 2 and any(k in name for k in pref_long_names):
                    chosen, chosen_name = vals, name
                    break

        # Fallback: largest span
        if chosen is None:
            for dev in cp[bl]:
                name = str(getattr(dev, "RTBeamLimitingDeviceType", "") or
                          getattr(dev, "ManufacturerDeviceIdentifier", "")).lower()
                if lj in dev:
                    vals = list(map(float, dev[lj].value))
                    if len(vals) == 2:
                        span = abs(vals[1] - vals[0])
                        if span > max_span and span < 120.0:
                            max_span, chosen, chosen_name = span, vals, name or "unknown2"

        if chosen is None:
            jaw1.append(np.nan)
            jaw2.append(np.nan)
            dtypes.append("none")
        else:
            jaw1.append(chosen[0])
            jaw2.append(chosen[1])
            dtypes.append(chosen_name)

    return np.array(jaw1, dtype=np.float32), np.array(jaw2, dtype=np.float32), dtypes


def compute_field_size_mm(cps) -> float:
    """
    Estimate Tomo field size (mm) from Y-jaws.
    Maps to {10, 25, 50} when close to {~7, ~20, ~42} mm (plan coordinate system).
    """
    j1, j2, _ = extract_jaw_positions(cps)
    spans = np.abs(j2 - j1)
    spans = spans[np.isfinite(spans)]

    if spans.size == 0:
        return float("nan")

    span = float(np.median(spans))

    if 4.0 <= span <= 10.0:
        return 10.0
    if 14.0 <= span <= 26.0:
        return 25.0
    if 32.0 <= span <= 60.0:
        return 50.0

    return span


def extract_tomo_private_tags(ds: pydicom.Dataset) -> Dict[str, Any]:
    """
    Extract some private Tomo tags if present.
    Returns dict with 'gantry_period_sec', 'treatment_pitch', 'couch_speed_mm_per_s'.
    """
    vals = {
        "gantry_period_sec": float("nan"),
        "treatment_pitch": float("nan"),
        "couch_speed_mm_per_s": float("nan")
    }

    try:
        beam = ds[(0x300A, 0x00B0)][0]
    except Exception:
        beam = None

    def _get_tag(container, tag):
        try:
            if container is not None and tag in container:
                return container[tag].value
        except Exception:
            pass
        try:
            if tag in ds:
                return ds[tag].value
        except Exception:
            pass
        return None

    # Accuray Tomo private creator often 'TOMO_HA_01'
    # 300D,1040: Gantry Period (s)
    v = _get_tag(beam, (0x300D, 0x1040))
    if v is None:
        v = _get_tag(ds, (0x300D, 0x1040))
    if v is not None:
        try:
            vals["gantry_period_sec"] = float(v)
        except Exception:
            pass

    # 300D,1060: Treatment Pitch (unitless)
    v = _get_tag(beam, (0x300D, 0x1060))
    if v is None:
        v = _get_tag(ds, (0x300D, 0x1060))
    if v is not None:
        try:
            vals["treatment_pitch"] = float(v)
        except Exception:
            pass

    # 300D,1080: Couch Speed (mm/s)
    v = _get_tag(beam, (0x300D, 0x1080))
    if v is None:
        v = _get_tag(ds, (0x300D, 0x1080))
    if v is not None:
        try:
            vals["couch_speed_mm_per_s"] = float(v)
        except Exception:
            pass

    return vals


def save_sinogram_classic_view(sino: np.ndarray, out: Path) -> None:
    """Save sinogram as classic view (aspect='auto')."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(sino, aspect='auto', cmap='gray', origin='upper')
    fig.savefig(str(out), dpi=200)
    plt.close(fig)


def save_sinogram_berlingo_style(
    sino: np.ndarray,
    out_png: Path,
    out_npy: Path | None = None,
    nx: int = 64,
    ny: int = 12,
    save_uint8: bool = True
) -> np.ndarray:
    """
    Build 'berlingo style' sinogram montage:
      - resize laterally to NX
      - stack each CP in NY-line blocks
      - inversion (1-st) to match current PNG rendering
    Saves PNG and optionally NPY.
    Returns the montage (uint8 if save_uint8, else float32 0..1).
    """
    sino = np.asarray(sino, dtype=np.float32)
    sino = sino - float(np.min(sino))
    mx = float(np.max(sino))
    if mx > 0:
        sino = sino / mx

    Ncp, Ndet = sino.shape
    if Ndet != 64:
        logging.warning(f"Non-standard sino: {Ndet}")

    rz = zoom(sino, (1, nx / Ndet), order=1)  # [Ncp, NX]
    st = np.vstack([np.tile(rz[i], (ny, 1)) for i in range(Ncp)])  # [NY*Ncp, NX]

    montage01 = np.clip(1.0 - st, 0.0, 1.0).astype(np.float32)

    if save_uint8:
        img = (montage01 * 255.0 + 0.5).astype(np.uint8)
        plt.imsave(str(out_png), img, cmap="gray")
        montage_to_save = img
    else:
        plt.imsave(str(out_png), montage01, cmap="gray", vmin=0.0, vmax=1.0)
        montage_to_save = montage01

    if out_npy is not None:
        np.save(str(out_npy), montage_to_save)

    return montage_to_save

