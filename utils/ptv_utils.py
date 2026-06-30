#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTV (Planning Target Volume) utilities for dose extraction and clustering.
"""
import re
import logging
import difflib
import numpy as np
from pathlib import Path
from typing import List, Tuple

from .json_utils import normalize_name


# ==================================================================
# CONFIG HELPERS - Load locally to avoid circular imports
# ==================================================================

def _get_ptv_config():
    """Get PTV config values, load locally to avoid circular imports."""
    try:
        from .. import config
        return {
            "dose_min_gy": config.PTV_DOSE_MIN_GY,
            "dose_max_gy": config.PTV_DOSE_MAX_GY,
            "exclude_keywords": config.PTV_EXCLUDE_KEYWORDS,
        }
    except ImportError:
        # Fallback if config not available
        return {
            "dose_min_gy": 49.0,
            "dose_max_gy": 80.0,
            "exclude_keywords": ["old", "union", "ring"],
        }


# ==================================================================
# MAIN FUNCTIONS
# ==================================================================


def extract_ptv_dose_gy(stem: str) -> float | None:
    """
    Extract dose (Gy) from filename stem with clinical safety override.

    CRITICAL: Parse ALL doses including < 49 Gy to enable patient rejection logic.
    Validation (bins/thresholds) must be done at selection time, NOT here.

    IMPORTANT: Accept numbers only with clear dose context:
      - Explicit "Gy" or "cGy" markers (highest confidence)
      - Standard dose format: PTV/CTV_<number> where number has decimal (52.8) or is >= 16
      - Subtraction patterns: PTV_45-59.4 → parse lower value (45.0)
      - 4-digit cGy patterns (5600, 7000)
      - br/ri/hr keywords (but NOT in anatomical names like "Aires")

    Handles:
      - CTV patterns with dose (CTV_BR52.8_Gy, CTV_P1_59.4_Gy)
      - Comma decimals (69,96 → 69.96)
      - Robust underscore normalization (avoids 1_66 → 1.66 bug)
      - Clinical safety heuristic: P-pattern override (P1_66 → 66 Gy, not 1.66)

    Returns:
        float: Dose in Gy (can be < 49), or None if no dose extractable
    """
    dose = _extract_ptv_dose_gy_internal(stem)

    # HEURISTIC: If dose < 10 Gy and name contains P\d+_(\d{2}),
    # override with the 2-digit number (clinical pattern: P1_66 should be 66 Gy, not 1.66)
    if dose is not None and dose < 10.0:
        p_match = re.search(r"p\d+_(\d{2})", (stem or "").lower())
        if p_match:
            override_dose = float(p_match.group(1))
            if 16.0 <= override_dose <= 100.0:
                logging.warning(
                    f"[DOSE_OVERRIDE] {stem}: detected {dose:.2f} Gy (suspicious), "
                    f"overridden to {override_dose:.2f} Gy via P-pattern heuristic"
                )
                return override_dose

    return dose


def _extract_ptv_dose_gy_internal(stem: str) -> float | None:
    """
    Internal function: Extract dose (Gy) from filename stem WITHOUT clinical override.

    This is the core extraction logic. Use extract_ptv_dose_gy() for the public API
    which applies clinical safety heuristics.
    """
    s = (stem or "").strip().lower()
    if not s:
        return None

    # Normalize decimal separators: French comma → dot
    s = s.replace(",", ".")  # CRITICAL: Handle French comma decimals

    # Normalize underscores CAREFULLY: only (\d{2})_(\d{1,2}) → decimal
    # This prevents 1_66 → 1.66 (P1_66 bug) while preserving 52_8 → 52.8
    s = re.sub(r"(\d{2})_(\d{1,2})", r"\1.\2", s)

    # SPECIAL CASE: Subtraction patterns like "PTV_45-59.4" or "PTV69-54"
    # These represent volume subtractions (e.g., PTV_high - PTV_mid)
    # Parse as: lower value (the base/fuzzy dose)
    subtraction_match = re.search(r"(?:ptv|ctv)[^\d]*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if subtraction_match:
        lower_str = subtraction_match.group(1)
        upper_str = subtraction_match.group(2)
        v_lower = float(lower_str)
        v_upper = float(upper_str)

        # Take the lower value as the base dose
        v = min(v_lower, v_upper)

        # Apply cGy conversion if needed
        if v >= 1000.0:
            v = v / 100.0
        elif v >= 100.0:
            v = v / 10.0

        # Keep plausible doses
        if 1.0 <= v <= 100.0:
            return float(v)

    # STRATEGY 1: Look for explicit Gy/cGy markers (most reliable)
    # Examples: "52.8_Gy", "52.8 Gy", "52.8Gy", "5280cGy"
    gy_pattern = re.findall(r"(\d+(?:\.\d+)?)\s*(?:_\s*)?(?:c)?gy\b", s)
    if gy_pattern:
        cand = []
        for x_str in gy_pattern:
            v = float(x_str)

            # Check if preceded by 'cgy' to determine if we need to divide
            idx = s.find(x_str)
            context_before = s[max(0, idx-10):idx].lower()

            if "cgy" in context_before or (v >= 100 and "cgy" in s[idx:idx+20]):
                v = v / 100.0

            # Keep plausible doses
            if 1.0 <= v <= 100.0:
                cand.append(v)

        if cand:
            return float(max(cand))

    # STRATEGY 2: Standard PTV/CTV dose format without explicit Gy
    # Pattern: PTV_<number> or CTV_<number> where number is likely a dose
    # Heuristic: number with decimal point (52.8) OR integer >= 16 (lower threshold now)
    # Match: PTV_54, PTV_60, PTV_52.8, CTV_54, etc. (with or without extensions like .nii.gz)
    ptv_format = re.findall(r"(?:ptv|ctv)_(\d+(?:\.\d+)?)(?:[_\.\-]|$)", s)
    if ptv_format:
        cand = []
        for x_str in ptv_format:
            v = float(x_str)

            # Accept if: has decimal point (52.8, 69.96) OR is >= 16 (includes PTV_16, PTV_21, etc.)
            if "." in x_str or v >= 16.0:
                if 1.0 <= v <= 100.0:
                    cand.append(v)

        if cand:
            return float(max(cand))

    # STRATEGY 3: FUZZY CTV patterns like "CTV_BR52.8_Gy" or "CTV_P1_59.4_Gy"
    # Pattern: CTV_<LABEL><dose>_Gy or CTV_<LABEL>_<dose>_Gy
    # Examples: CTV_BR52.8_Gy, CTV_P1_59.4_Gy
    # Extract: <dose> value (no underscore before dose)
    ctv_fuzzy = re.findall(r"ctv_[a-z0-9]*(\d+(?:\.\d+)?)(?:_)?gy", s)
    if ctv_fuzzy:
        cand = []
        for x_str in ctv_fuzzy:
            v = float(x_str)
            if 1.0 <= v <= 100.0:
                cand.append(v)

        if cand:
            return float(max(cand))

    # STRATEGY 4: PTV<number>Gy patterns (e.g., PTV45Gy, PTV45Gysv2)
    ptv_gy_pattern = re.findall(r"(?:ptv|ctv)[^\d]*(\d+(?:\.\d+)?)\s*gy", s)
    if ptv_gy_pattern:
        cand = []
        for x_str in ptv_gy_pattern:
            v = float(x_str)
            if 1.0 <= v <= 100.0:
                cand.append(v)

        if cand:
            return float(max(cand))

    # STRATEGY 5: 4-digit cGy numbers (e.g., 5600, 7000)
    large_nums = re.findall(r"(?:ptv|ctv)[^\d]*(\d{4})\b", s)
    if large_nums:
        cand = []
        for x_str in large_nums:
            v = float(x_str)
            if v >= 1000.0:
                v = v / 100.0

            if 1.0 <= v <= 100.0:
                cand.append(v)

        if cand:
            return float(max(cand))

    # STRATEGY 6: Fallback to br/ri/hr keywords ONLY if clearly isolated
    # Don't match keywords in anatomical names (e.g., "Aires" contains "re")
    if re.search(r"(^|[_\W])br($|[_\W])", s) and "aire" not in s:
        return 54.0
    if re.search(r"(^|[_\W])ri($|[_\W])", s) and "aire" not in s:
        return 59.4
    if re.search(r"(^|[_\W])hr($|[_\W])", s) and "aire" not in s:
        return 70.0

    # No dose found with proper context
    return None



def _is_subtraction_roi_name(name: str) -> bool:
    """
    Detect if ROI name represents a subtraction volume (e.g., PTV_52.8_-_PTV_59.4).

    These differential volumes must be excluded from PTV clustering as they represent
    the difference between two PTVs, not an atomic treatment volume.

    Args:
        name: ROI filename stem (e.g., "PTV_52.8_-_PTV_59.4")

    Returns:
        True if the name contains a subtraction pattern, False otherwise

    Examples:
        >>> _is_subtraction_roi_name("PTV_52.8_-_PTV_59.4")
        True
        >>> _is_subtraction_roi_name("PTV_45-PTV_59.4")
        True
        >>> _is_subtraction_roi_name("CTV_45 - CTV_54")
        True
        >>> _is_subtraction_roi_name("PTV_52.8")
        False
        >>> _is_subtraction_roi_name("PTV-BR")  # Not a subtraction
        False
    """
    if not name:
        return False

    s_lower = name.lower()

    # Quick check: if no dash/minus, it's not a subtraction
    if "-" not in s_lower:
        return False

    # Strategy 1: Explicit pattern with two PTV/CTV mentions and doses
    # Matches: PTV_52.8_-_PTV_59.4, PTV_45-PTV_59.4, CTV_45 - CTV_54
    pattern_explicit = r"(?:ptv|ctv)[^\d]*(\d+(?:[.,]\d+)?)[^a-z]*[-_]+\s*(?:ptv|ctv)[^\d]*(\d+(?:[.,]\d+)?)"
    if re.search(pattern_explicit, s_lower):
        return True

    # Strategy 2: Single PTV/CTV with two doses separated by dash
    # Matches: PTV_52.8_-_59.4, PTV_45-59.4
    # But NOT: PTV-BR, PTV-54Gy (no second number)
    if re.search(r"(?:ptv|ctv)", s_lower):
        # Look for: number [dash/underscore/space] number
        # Make sure both sides have numbers
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*[-_]+\s*(\d+(?:[.,]\d+)?)", s_lower)
        if match:
            # Verify these are dose-like numbers (not just any numbers)
            num1 = float(match.group(1).replace(",", "."))
            num2 = float(match.group(2).replace(",", "."))
            # Doses are typically 16-100 Gy (or fractionated cGy like 1800-7000)
            if (16 <= num1 <= 100 and 16 <= num2 <= 100) or (1600 <= num1 <= 10000 and 1600 <= num2 <= 10000):
                return True

    return False


def find_ptv_candidates(struct_dir: Path, preferred_structure_set_id: str | None = None) -> List[Tuple[float, Path]]:
    """
    Détecte des candidats PTV de manière robuste.

    CRITICAL: Parse ALL doses including < 49 Gy to enable rejection logic.
    Filtering by bins happens at selection time, NOT here.

    Args:
        struct_dir: Root structures directory
        preferred_structure_set_id: If provided, only search in this structure set subdirectory
                                   (STRICT: no fallback to other structure sets)

    Objectifs:
      - Ne pas perdre des PTV valides à cause des tirets
      - Exclure proprement les volumes dérivés / non pertinents (old/union/ring/opt/minus/subtract)
      - EXCLURE les volumes de soustraction (PTV_52.8_-_PTV_59.4)
      - Dose obligatoire via extract_ptv_dose_gy()
      - STRICT structure set filtering: NO fallback if preferred set doesn't have PTVs

    Stratégie:
      1) Passe "strict" (préférence): startswith ptv, pas de forbidden keywords, dose ok
         (on tolère les tirets, ils ne sont plus bloquants)
      2) Si très peu de résultats, passe "fuzzy" en plus (pas à la place):
         récupère des stems proches de "ptv" via difflib, puis même filtrage
      3) Déduplique par filename

    Remarque:
      - On neutralise explicitement un éventuel mot-clé "ptv" dans exclude_keywords (config),
        sinon ça exclurait tout.
      - On n'applique AUCUN filtre de dose minimale ici (dose_min_gy) pour permettre
        la détection des patients avec doses < 49 Gy (rejected).
      - IMPORTANT: If preferred_structure_set_id is specified and that set has no PTVs,
        we return EMPTY list (no silent fallback to other structure sets).
    """
    # Filter by structure set if specified
    if preferred_structure_set_id:
        search_dir = Path(struct_dir) / preferred_structure_set_id
        if not search_dir.exists():
            logging.warning(f"[PTV] Preferred structure set directory not found: {search_dir}")
            return []
        files = list(search_dir.rglob("*.nii*"))
        logging.info(f"[PTV] Searching in structure set {preferred_structure_set_id}: {len(files)} files")
    else:
        files = list(Path(struct_dir).rglob("*.nii*"))

    out: List[Tuple[float, Path]] = []

    cfg = _get_ptv_config()
    exclude_keywords = [k.lower() for k in cfg.get("exclude_keywords", []) if str(k).strip()]

    # CRITICAL: Never exclude "ptv" keyword itself
    exclude_keywords = [k for k in exclude_keywords if k != "ptv"]

    def _is_forbidden(s: str) -> bool:
        s_lower = (s or "").lower()

        for keyword in exclude_keywords:
            if keyword and keyword in s_lower:
                return True

        # Legacy filters (backward compatibility)
        if re.search(r"(^|[_\W])opt($|[_\W])", s_lower):
            return True
        if re.search(r"\b(minus|moins|subtract|soustra|substract)\b", s_lower):
            return True

        # CRITICAL: Exclude subtraction volumes (PTV_52.8_-_PTV_59.4, etc.)
        # These are differential volumes, not atomic treatment volumes
        if _is_subtraction_roi_name(s):
            logging.debug(f"[PTV] Excluded subtraction ROI: {s}")
            return True

        return False

    def _is_ptv_like(stem: str) -> bool:
        # On se base sur normalize_name pour gérer "PTV", "PTV.", "PTV_" etc.
        return normalize_name((stem or "").lower()).startswith("ptv")

    def _try_add(f: Path):
        raw = f.stem.lower().strip()
        if not _is_ptv_like(raw):
            return
        if _is_forbidden(raw):
            return
        dose = extract_ptv_dose_gy(raw)
        if dose is None:
            return
        # IMPORTANT: No minimum dose filter here - we need ALL doses including < 49 Gy
        out.append((float(dose), f))

    # ---------------- pass 1: strict (mais tirets autorisés) ----------------
    for f in files:
        _try_add(f)

    # ---------------- pass 2: fuzzy en complément si peu de résultats ----------------
    # Heuristique: si on a 0 ou 1 PTV, on essaie de récupérer des "quasi-PTV"
    if len(out) <= 1 and files:
        names = [p.stem.lower().strip() for p in files]
        close = difflib.get_close_matches("ptv", names, n=24, cutoff=0.72)

        for f in files:
            raw = f.stem.lower().strip()
            if raw not in close:
                continue
            _try_add(f)

        if len(out) > 1:
            logging.warning(f"Fuzzy PTV enabled: {[p.name for _, p in out]}")

    # ---------------- dédoublonnage + tri ----------------
    best_by_file = {}
    for d, p in out:
        key = p.name
        if key not in best_by_file:
            best_by_file[key] = (d, p)
        else:
            # si doublon (rare), on garde la dose la plus grande (cohérent avec extract_ptv_dose_gy max(cand))
            prev_d, _ = best_by_file[key]
            if d > prev_d:
                best_by_file[key] = (d, p)

    out = list(best_by_file.values())
    out.sort(key=lambda x: (x[0], x[1].name))

    for dose, p in out:
        logging.info(f"PTV found: {p.name} -> {dose:.2f} Gy")

    return out

def cluster_ptvs_by_dose(
    ptv_list: List[Tuple[float, Path]],
    tol_gy: float = 0.5
) -> List[Tuple[float, List[Path]]]:
    """
    Group PTVs by dose levels (Gy) with tolerance.
    Returns: sorted list [(dose_rep, [paths...]), ...] where dose_rep = cluster mean.
    """
    if not ptv_list:
        return []

    pts = sorted([(float(d), p) for d, p in ptv_list], key=lambda x: x[0])
    clusters: List[Tuple[List[float], List[Path]]] = []

    for d, p in pts:
        if not clusters:
            clusters.append(([d], [p]))
            continue

        prev_doses, prev_paths = clusters[-1]
        center = float(np.mean(prev_doses))
        if abs(d - center) <= float(tol_gy):
            prev_doses.append(d)
            prev_paths.append(p)
        else:
            clusters.append(([d], [p]))

    out: List[Tuple[float, List[Path]]] = []
    for doses, paths in clusters:
        out.append((float(np.mean(doses)), paths))

    out.sort(key=lambda x: x[0])
    return out


def normalize_gy_to_unit(
    dose_gy: float,
    vmin_gy: float = 49.0,
    vmax_gy: float = 75.0
) -> float:
    """Return a 0..1 value (clamped) from a dose in Gy."""
    vmin = float(vmin_gy)
    vmax = float(vmax_gy)
    if vmax <= vmin:
        return 0.0
    x = (float(dose_gy) - vmin) / (vmax - vmin)
    return float(np.clip(x, 0.0, 1.0))


def dose_to_bin(d: float) -> str:
    """
    Map dose to PTV bin using strict bins:
      ptv_br  : 49 <= d < 55  (baseline risk)
      ptv_ri  : 55 <= d < 63  (intermediate risk)
      ptv_hr  : 63 <= d       (high risk)

    Doses < 49 Gy are mapped to "ptv_unknown" (used for rejection logic).
    """
    if d is None:
        return "ptv_unknown"

    d = float(d)

    if d < 49.0:
        return "ptv_unknown"
    if d < 55.0:
        return "ptv_br"
    if d < 63.0:
        return "ptv_ri"
    return "ptv_hr"


def build_ptv_channel_metadata(
    ptv_clusters: list[tuple[float, list[Path]]],
    vmin_gy: float,
    vmax_gy: float
) -> dict:
    """
    Build metadata dict:
      channels: {ptv_br/ptv_ri/ptv_hr: {...}}
      clusters: [...]
    """
    # Clusters as listable
    clusters_payload = []
    for dose_rep, paths in ptv_clusters:
        clusters_payload.append({
            "dose_rep_gy": float(dose_rep),
            "n_files": int(len(paths)),
            "files": [Path(p).name for p in paths],
        })

    # Aggregate by channel - CORRECTED: use ptv_br/ptv_ri/ptv_hr instead of ptv_low/mid/high
    by_ch = {"ptv_br": [], "ptv_ri": [], "ptv_hr": []}
    for dose_rep, paths in ptv_clusters:
        ch = dose_to_bin(float(dose_rep))
        # Map ptv_low/mid/high to ptv_br/ri/hr
        ch_mapping = {"ptv_low": "ptv_br", "ptv_mid": "ptv_ri", "ptv_high": "ptv_hr"}
        ch = ch_mapping.get(ch, ch)
        by_ch[ch].append((float(dose_rep), [Path(p) for p in paths]))

    channels = {}
    for ch in ("ptv_br", "ptv_ri", "ptv_hr"):
        items = by_ch[ch]
        if not items:
            channels[ch] = None
            continue

        doses = sorted({float(d) for d, _ in items})
        # United files
        files = []
        for _, ps in items:
            files.extend([Path(p).name for p in ps])
        files = sorted(set(files))

        dose_rep = float(max(doses))
        dose_norm = normalize_gy_to_unit(dose_rep, vmin_gy=vmin_gy, vmax_gy=vmax_gy)

        channels[ch] = {
            "doses_gy": doses,
            "dose_rep_gy": dose_rep,
            "dose_norm_rep": float(dose_norm),
            "n_files": int(len(files)),
            "files": files,
        }

    return {
        "clusters": clusters_payload,
        "channels": channels
    }


# ==================================================================
# UNIT TESTS
# ==================================================================

def test_extract_ptv_dose_gy():
    """
    Unit tests for extract_ptv_dose_gy() with comprehensive coverage.

    CRITICAL CASES:
      1. P-pattern bug: P1_66 should be 66.0, NOT 1.66 (underscore override)
      2. Decimal underscore: 52_8 should be 52.8 (normaliz underscores)
      3. French decimals: 69,96 should be 69.96 (comma decimals)
      4. Explicit Gy markers: 44_Gy should be 44.0
      5. Standard PTV format: PTV_54 should be 54.0
      6. Fuzzy CTV: CTV_BR52.8_Gy should be 52.8
    """
    test_cases = [
        # (stem, expected_dose, description)
        ("CTV_P1_66_gY", 66.0, "P1_66 pattern (critical: NOT 1.66)"),
        ("CTV_P1_59.4_Gy", 59.4, "P1_59.4 with decimal"),
        ("PTV_52_8", 52.8, "Underscore decimal"),
        ("PTV_T_69,96Gy", 69.96, "French comma decimal"),
        ("CTV_44_Gy", 44.0, "Explicit Gy marker"),
        ("PTV_54", 54.0, "Standard PTV format"),
        ("CTV_BR52.8_Gy", 52.8, "Fuzzy CTV label"),
        ("PTV_69_96_Gy", 69.96, "Double underscore pattern"),
        ("CTV_45", 45.0, "CTV without Gy"),
        ("ptv_hr_70_gy", 70.0, "hr keyword with dose"),
        ("PTV_45-59.4", 45.0, "Subtraction pattern (lower)"),
        ("CTV_5600cGy", 56.0, "cGy conversion"),
        (None, None, "None input"),
        ("", None, "Empty string"),
        ("random_file.nii.gz", None, "No dose context"),
    ]

    all_pass = True
    for stem, expected, desc in test_cases:
        result = extract_ptv_dose_gy(stem)
        passed = result == expected
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
            print(f"{status}: {stem!r:25s} → {result!r:6s} (expected {expected!r:6s}) | {desc}")
        else:
            print(f"{status}: {stem!r:25s} → {result!r:6s} | {desc}")

    if all_pass:
        print("\n✅ All tests PASSED")
    else:
        print("\n❌ Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PTV_UTILS UNIT TESTS")
    print("=" * 80 + "\n")
    test_extract_ptv_dose_gy()
