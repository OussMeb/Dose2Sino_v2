#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask utilities for reading, resampling, and projecting 3D masks.
"""
import logging
import hashlib
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import List
from collections import Counter

from .json_utils import normalize_name, _norm


def resample_to_ref(
    img: sitk.Image | None,
    ref_img: sitk.Image,
    is_label: bool = True
) -> sitk.Image | None:
    """
    Resample img to ref_img grid.
    - is_label=True: nearest neighbor (masks)
    - is_label=False: linear (images)
    Returns None if img is None.
    """
    if img is None:
        return None

    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    out = sitk.Resample(
        img,
        ref_img,
        sitk.Transform(),
        interp,
        0.0,  # default value outside
        img.GetPixelIDValue()
    )
    return out


def read_mask_array_on_ref(path: Path, ref_img: sitk.Image | None) -> np.ndarray:
    """
    Read a NIfTI mask and resample to CT grid (ref_img) if provided.
    Returns: array Z,Y,X uint8 (0/1)
    """
    img = sitk.ReadImage(str(path))
    if ref_img is not None:
        img = resample_to_ref(img, ref_img, is_label=True)
    arr = sitk.GetArrayFromImage(img)
    return (arr > 0).astype(np.uint8)


def union_masks_from_paths(
    paths: List[Path],
    ref_img: sitk.Image | None = None
) -> np.ndarray | None:
    """
    Load and union masks (values >0) in Z,Y,X.
    If ref_img is provided, each mask is resampled to CT grid.
    Returns None if empty.
    """
    if not paths:
        return None

    out = None
    for p in paths:
        try:
            arr = read_mask_array_on_ref(p, ref_img)  # uint8 0/1 on CT grid
            if out is None:
                out = arr.copy()
            else:
                if arr.shape != out.shape:
                    logging.warning(f"Mask union: shape mismatch {arr.shape} vs {out.shape} (skip {p.name})")
                    continue
                out |= (arr > 0).astype(np.uint8)
        except Exception as e:
            logging.warning(f"Mask: read/resample failed {p.name}: {e}")

    if out is None:
        return None
    return out.astype(np.uint8)


def project_mask_to_filled_stack(
    mask_zyx: np.ndarray | None,
    angles: np.ndarray,
    tables: np.ndarray,
    x_iso: float,
    y_iso: float,
    z_iso: float,
    spacing_zyx: tuple[float, float, float],
    origin_zyx: tuple[float, float, float],
    nx: int,
    ny: int,
    geometry_func,
    caster_class,
    sad_mm: float,
    sdd_mm: float,
    width_mm: float,
    height_mm: float,
    fill_value: float = 0.5,
    use_proximity: bool = False,
    prox_lo: float = 0,
    prox_hi: float = 1,
    prox_mode: str = "absolute",
    prox_dmin: float = 750.0,
    prox_dmax: float = 950.0,
) -> np.ndarray:
    """
    Project a 3D mask via GPUCastMask (arc + entry).
    Output stack [NY, NX, N_CP] float32

    - use_proximity=False: hit * fill_value (classic)
    - use_proximity=True: proximity (0 outside ROI, [prox_lo..prox_hi] inside ROI)

    Args:
        mask_zyx: 3D binary mask (Z,Y,X)
        angles: Gantry angles (degrees) [N_CP]
        tables: Table positions (mm) [N_CP]
        x_iso, y_iso, z_iso: Isocenter coordinates (mm)
        spacing_zyx: Voxel spacing (mm) (Z,Y,X)
        origin_zyx: Volume origin (mm) (Z,Y,X)
        nx, ny: Detector resolution
        geometry_func: Function to compute source-detector geometry
        caster_class: GPU caster class (e.g., GPUCastMask)
        fill_value: Fill value when use_proximity=False
        use_proximity: Enable proximity encoding
        prox_lo, prox_hi: Proximity range
        prox_mode: Proximity mode ("absolute" or "relative")
        prox_dmin, prox_dmax: Distance range for proximity
    """
    n_cp = int(len(angles))
    stack = np.zeros((ny, nx, n_cp), dtype=np.float32)

    if mask_zyx is None or mask_zyx.max() <= 0:
        return stack

    # Binary volume (0/1)
    vol_mu = (mask_zyx > 0).astype(np.float32)

    caster = caster_class(
        vol_mu,
        spacing_zyx,
        origin_zyx
    )

    for i, (theta, dz) in enumerate(zip(angles, tables)):
        src, det, u, v, su, sv = geometry_func(
            x_iso, y_iso, z_iso, float(theta), float(dz),
            sad_mm, sdd_mm, width_mm, height_mm, nx, ny,
            flip_lr=True
        )
        v_vec = v * sv

        if use_proximity:
            I, entry_mm, prox = caster.run_with_entry(
                src, det, v_vec, su, sv, nx, ny,
                thr_hit=0.5,
                hit_mode="nearest",
                return_proximity=True,
                proximity_lo=float(prox_lo),
                proximity_hi=float(prox_hi),
                proximity_mode=str(prox_mode),
                proximity_dmin=float(prox_dmin),
                proximity_dmax=float(prox_dmax),
            )
            hit = (entry_mm >= 0.0)
            stack[:, :, i] = np.where(hit, prox, 0.0).astype(np.float32)
        else:
            I, entry_mm = caster.run_with_entry(
                src, det, v_vec, su, sv, nx, ny,
                thr_hit=0.5,
                hit_mode="nearest",
                return_proximity=False
            )
            hit = (entry_mm >= 0.0)
            stack[:, :, i] = hit.astype(np.float32) * float(fill_value)

    return stack.astype(np.float32)


def project_external_entry_exit_stacks(
    external_mask_zyx: np.ndarray | None,
    angles: np.ndarray,
    tables: np.ndarray,
    x_iso: float,
    y_iso: float,
    z_iso: float,
    spacing_zyx: tuple[float, float, float],
    origin_zyx: tuple[float, float, float],
    nx: int,
    ny: int,
    geometry_func,
    caster_class,
    entry_to_proximity_func,
    sad_mm: float,
    sdd_mm: float,
    width_mm: float,
    height_mm: float,
    prox_lo: float = 0,
    prox_hi: float = 1,
    prox_mode: str = "absolute",
    prox_dmin: float = 750.0,
    prox_dmax: float = 950.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (entry_stack, exit_stack) shape [NY, NX, N_CP]
    Encoding:
      - 0 if no hit
      - else proximity mapping in [prox_lo..prox_hi]
    """
    n_cp = int(len(angles))
    entry_stack = np.zeros((ny, nx, n_cp), dtype=np.float32)
    exit_stack = np.zeros((ny, nx, n_cp), dtype=np.float32)

    if external_mask_zyx is None or np.max(external_mask_zyx) <= 0:
        return entry_stack, exit_stack

    vol_mu = (external_mask_zyx > 0).astype(np.float32)

    caster = caster_class(
        vol_mu,
        spacing_zyx,
        origin_zyx
    )

    for i, (theta, dz) in enumerate(zip(angles, tables)):
        src, det, u, v, su, sv = geometry_func(
            x_iso, y_iso, z_iso, float(theta), float(dz),
            sad_mm, sdd_mm, width_mm, height_mm, nx, ny,
            flip_lr=True
        )
        v_vec = v * sv

        I, entry_mm, exit_mm, thick_mm = caster.run_with_entry_exit(
            src, det, v_vec, su, sv, nx, ny,
            thr_hit=0.5,
            hit_mode="nearest",
            return_proximity=False,
            return_thickness=True,
        )

        entry_prox = entry_to_proximity_func(
            entry_mm,
            lo=float(prox_lo),
            hi=float(prox_hi),
            mode=str(prox_mode),
            dmin=float(prox_dmin),
            dmax=float(prox_dmax),
        )

        exit_prox = entry_to_proximity_func(
            exit_mm,
            lo=float(prox_lo),
            hi=float(prox_hi),
            mode=str(prox_mode),
            dmin=float(prox_dmin),
            dmax=float(prox_dmax),
        )

        hit_e = (entry_mm >= 0.0)
        hit_x = (exit_mm >= 0.0)

        entry_stack[:, :, i] = np.where(hit_e, entry_prox, 0.0).astype(np.float32)
        exit_stack[:, :, i] = np.where(hit_x, exit_prox, 0.0).astype(np.float32)

    return entry_stack.astype(np.float32), exit_stack.astype(np.float32)


def apply_tomo_transform_to_stack(
    mask_zyx: np.ndarray | None,
    angles: np.ndarray,
    tables: np.ndarray,
    x_iso: float,
    y_iso: float,
    z_iso: float,
    spacing_zyx: tuple[float, float, float],
    origin_zyx: tuple[float, float, float],
    is_label: bool = True,
) -> np.ndarray:
    """
    Build a 3D stack [N_CP, NY_ct, NX_ct] where slice i is the axial mask slice
    at table position tables[i], rotated in-plane by gantry angle angles[i].

    angles[i]: gantry angle (deg) — patient rotated by -angles[i] in the beam frame
    tables[i]: table Z position (mm, DICOM) — selects the axial slice
    is_label:  nearest-neighbour (True) or bilinear (False) interpolation
    """
    from scipy.ndimage import affine_transform

    n_cp = int(len(angles))
    sz, sy, sx = mask_zyx.shape if mask_zyx is not None else (1, 1, 1)
    stack = np.zeros((n_cp, sy, sx), dtype=np.float32)

    if mask_zyx is None or mask_zyx.max() <= 0:
        return stack

    import torch
    if isinstance(mask_zyx, torch.Tensor):
        mask_zyx = mask_zyx.numpy()

    # Isocenter in pixel space: row ~ Y axis, col ~ X axis
    cy = (y_iso - origin_zyx[1]) / spacing_zyx[1]
    cx = (x_iso - origin_zyx[2]) / spacing_zyx[2]
    center = np.array([cy, cx])

    order = 0 if is_label else 1

    for i, (theta, dz) in enumerate(zip(angles, tables)):
        # tables[i] is a COUCH-relative position (DICOM TableTopLateralPosition on
        # this tomo data), NOT an absolute patient-z. The patient slice irradiated
        # at this control point is z_iso - dz, the SAME convention the canonical
        # geometry uses (compute_source_detector_geometry: z_corr = z_iso - delta_z).
        # Using dz directly as an absolute z mapped >half the control points onto a
        # single clamped CT edge slice (z range [122,180], 660/1201 clamped on
        # patient 183040) -> the berlingo showed the wrong head/foot slice, fully
        # decorrelated from the sinogram. z_iso - dz spans z_idx [35,162] (no clamp)
        # and centres the couch sweep on the PTV.
        z_patient = z_iso - float(dz)
        z_idx = int(np.clip(round((z_patient - origin_zyx[0]) / spacing_zyx[0]), 0, sz - 1))
        slice_2d = mask_zyx[z_idx].astype(np.float32)

        # Beam-frame rotation. The gantry-angle convention that aligns the rotated
        # anatomy with the LOT sinogram is (90 - theta), NOT theta: verified by
        # projecting the PTV mask and correlating with the target sinogram across
        # patients (theta -> centroid corr ~ -0.6 ANTI-correlated; 90-theta -> +0.87).
        # Using plain theta mis-registered every berlingo vs its sinogram (the
        # ~0.31 val plateau). See geometry-alignment memory.
        alpha = np.deg2rad(90.0 - float(theta))
        c, s = np.cos(alpha), np.sin(alpha)
        mat = np.array([[c, -s], [s, c]])
        offset = center - mat @ center

        stack[i] = affine_transform(
            slice_2d, mat, offset=offset,
            order=order, mode='constant', cval=0.0,
        )

    return stack


# ------------------------------------------------------------------
# Structure set conflict detection (multi-set validation)
# ------------------------------------------------------------------

def structure_set_id(struct_root: Path, p: Path) -> str:
    """
    Extract structure set ID from path.
    Example: structures/643eb7/Aorte.nii.gz => "643eb7"
    """
    try:
        rel = p.relative_to(struct_root)
        return rel.parts[0] if len(rel.parts) > 1 else "root"
    except Exception:
        return "root"


def build_struct_index(struct_root: Path) -> dict:
    """
    Build quick index of structures by canonical name and structure set.
    Returns: {normalized_name: [(set_id, path), ...], ...}

    Properly handles .nii.gz files by extracting the filename without extensions.
    """
    idx = {}
    if not struct_root.exists():
        return idx

    for p in struct_root.rglob("*.nii*"):
        sid = structure_set_id(struct_root, p)
        # For .nii.gz files, p.stem is "filename.nii", not "filename"
        # We need to get the first component of the filename
        file_name = p.name
        for ext in [".nii.gz", ".nii"]:
            if file_name.endswith(ext):
                file_name = file_name[:-len(ext)]
                break
        stem = normalize_name(file_name)
        if stem not in idx:
            idx[stem] = []
        idx[stem].append((sid, p))

    return idx


def find_candidates(struct_root: Path, query: str) -> list:
    """
    Find all candidates for a given query (insensitive to plurals/accents).
    Returns sorted list of Paths.

    Properly handles .nii.gz files by extracting the filename without extensions.
    """
    q = normalize_name(query)
    q2 = q[:-1] if q.endswith("s") else None
    out = []

    if not struct_root.exists():
        return []

    for p in struct_root.rglob("*.nii*"):
        # For .nii.gz files, p.stem is "filename.nii", not "filename"
        # We need to get the first component of the filename
        # Remove .nii, .nii.gz, .gz extensions
        file_name = p.name
        for ext in [".nii.gz", ".nii"]:
            if file_name.endswith(ext):
                file_name = file_name[:-len(ext)]
                break

        s = normalize_name(file_name)
        if q in s or (q2 and q2 in s):
            out.append(p)

    return sorted(set(out))


def md5_hash(path: Path, chunk: int = 1024 * 1024) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
    except Exception as e:
        logging.warning(f"Cannot hash {path}: {e}")
        return ""
    return h.hexdigest()


def dice_coeff(a: np.ndarray, b: np.ndarray) -> float:
    """Dice coefficient between two binary masks."""
    a = (a > 0).astype(np.bool_)
    b = (b > 0).astype(np.bool_)
    inter = np.logical_and(a, b).sum()
    den = a.sum() + b.sum()
    return (2.0 * inter / den) if den > 0 else 1.0


def choose_or_quarantine_roi(
    struct_root: Path,
    roi_name: str,
    ct_img: sitk.Image,
    patient_out_dir: Path,
    strict_mode: bool = True,
    preferred_structure_set_id: str | None = None
) -> tuple:
    """
    Smart ROI selection with multi-set conflict detection.

    Returns: (mask_array_or_none, conflict_dict_or_None)

    Args:
        preferred_structure_set_id: If provided, only use structures from this set ID
                                     (useful when processing specific CT/RS/RP groups from associations)

    If strict_mode=True:
      - If ROI exists in multiple structure sets with conflicting content -> raise RuntimeError
    If strict_mode=False:
      - Returns first match, logs warning if conflicts detected
    """
    cands = find_candidates(struct_root, roi_name)
    if not cands:
        return None, None

    # Group by structure set ID
    by_set = {}
    for p in cands:
        sid = structure_set_id(struct_root, p)
        if sid not in by_set:
            by_set[sid] = []
        by_set[sid].append(p)

    # If preferred structure set is specified, filter to only that set
    if preferred_structure_set_id is not None:
        if preferred_structure_set_id in by_set:
            logging.info(f"[ASSOCIATIONS] Using structure set {preferred_structure_set_id} for {roi_name}")
            by_set = {preferred_structure_set_id: by_set[preferred_structure_set_id]}
        else:
            available_sets = list(by_set.keys())
            logging.warning(
                f"[ASSOCIATIONS] Preferred structure set {preferred_structure_set_id} not found for {roi_name}, "
                f"available: {available_sets} -> no fallback across sets"
            )
            # Robust behavior: do NOT fall back to another structure set.
            # Let caller mark ROI as missing for this specific CT/RS/RP group.
            return None, None

    # Case 1: Single set -> union all matches
    if len(by_set) == 1:
        union = None
        for p in list(by_set.values())[0]:
            try:
                arr = read_mask_array_on_ref(p, ct_img)
                union = arr if union is None else (union | arr)
            except Exception as e:
                logging.warning(f"Cannot read {p.name} for {roi_name}: {e}")
        if union is not None:
            return union.astype(np.uint8), None
        return None, None

    # Case 2: Multiple sets -> compare
    # Sub-case 2a: Check byte-level identity
    all_cands = []
    for paths in by_set.values():
        all_cands.extend(paths)

    hashes = {str(p): md5_hash(p) for p in all_cands}
    if len(set(hashes.values())) == 1 and "" not in hashes.values():
        # All files byte-identical -> safe to use first
        try:
            return read_mask_array_on_ref(all_cands[0], ct_img).astype(np.uint8), None
        except Exception as e:
            logging.warning(f"Cannot read {all_cands[0]} despite identical hash: {e}")
            return None, None

    # Sub-case 2b: Resample + compare set-by-set
    set_masks = {}
    for sid, paths in by_set.items():
        u = None
        for p in paths:
            try:
                arr = read_mask_array_on_ref(p, ct_img)
                u = arr if u is None else (u | arr)
            except Exception as e:
                logging.warning(f"Cannot read {p.name} from set {sid}: {e}")
        if u is not None:
            set_masks[sid] = u.astype(np.uint8)

    if not set_masks:
        return None, None

    sids = list(set_masks.keys())
    base = set_masks[sids[0]]
    dice_results = {}
    conflicts = False

    for sid in sids[1:]:
        d = dice_coeff(base, set_masks[sid])
        dice_results[sid] = float(d)
        if d < 0.999:  # strict threshold
            conflicts = True

    if conflicts:
        conflict_info = {
            "roi": roi_name,
            "structure_sets": sids,
            "hashes": hashes,
            "dice_vs_first_set": dice_results,
            "files_by_set": {
                sid: [str(p) for p in paths]
                for sid, paths in by_set.items()
            }
        }

        if strict_mode:
            raise RuntimeError(f"ROI conflict multi-set: {roi_name} (sets={sids}, dice_vs_first={dice_results})")
        else:
            logging.warning(f"ROI multi-set conflict detected for {roi_name}: dice={dice_results}")
            return base, conflict_info

    return base, None


def debug_struct_files_for_patient(struct_dir: Path, max_print: int = 300) -> dict:
    """
    List all .nii* files present under struct_dir (recursive).
    Log a summary + sample.
    Returns a dict (useful for JSON if needed).
    """
    struct_dir = Path(struct_dir)
    files = sorted(struct_dir.rglob("*.nii*")) if struct_dir.exists() else []

    logging.info("========== [DEBUG-STRUCT-FILES] ==========")
    logging.info(f"[DEBUG-STRUCT-FILES] struct_dir={struct_dir}")
    logging.info(f"[DEBUG-STRUCT-FILES] total_files={len(files)}")

    if not files:
        logging.warning("[DEBUG-STRUCT-FILES] No .nii* files found.")
        logging.info("=========================================")
        return {"struct_dir": str(struct_dir), "total_files": 0, "files": []}

    # Useful stats
    ext = Counter([p.suffix.lower() for p in files])
    stems = Counter([p.stem for p in files])
    logging.info(f"[DEBUG-STRUCT-FILES] ext_counts={dict(ext)}")
    logging.info(f"[DEBUG-STRUCT-FILES] top_stems(30)={stems.most_common(30)}")

    # Print (watch out for volume)
    n_show = min(max_print, len(files))
    logging.info(f"[DEBUG-STRUCT-FILES] listing_first_{n_show}:")
    for p in files[:n_show]:
        try:
            rel = p.relative_to(struct_dir)
            logging.info(f"  - {rel}")
        except Exception:
            logging.info(f"  - {p}")

    if len(files) > n_show:
        logging.info(f"[DEBUG-STRUCT-FILES] ... +{len(files) - n_show} others not shown")

    logging.info("=========================================")

    return {
        "struct_dir": str(struct_dir),
        "total_files": len(files),
        "ext_counts": dict(ext),
        "top_stems_30": stems.most_common(30),
        "files": [str(p) for p in files],
    }

