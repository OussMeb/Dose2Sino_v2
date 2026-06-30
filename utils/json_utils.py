#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON and serialization utilities.
"""
import json
import logging
import unicodedata
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any


def _now_iso() -> str:
    """Return current timestamp in ISO format."""
    return datetime.now().isoformat(timespec="seconds")


def _safe_float(x) -> float | None:
    """Convert to float safely, returning None if invalid."""
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _arr_stats(a: np.ndarray | None) -> dict | None:
    """Compute basic statistics for an array."""
    if a is None:
        return None
    try:
        a = np.asarray(a)
        if a.size == 0:
            return {"shape": list(a.shape), "dtype": str(a.dtype), "size": 0}

        af = a.astype(np.float32, copy=False) if a.dtype != np.float32 else a
        mn = _safe_float(np.min(af))
        mx = _safe_float(np.max(af))
        mean = _safe_float(np.mean(af))
        std = _safe_float(np.std(af))
        nnz = int(np.count_nonzero(af))

        return {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "size": int(a.size),
            "min": mn,
            "max": mx,
            "mean": mean,
            "std": std,
            "nonzero": nnz,
            "nonzero_frac": _safe_float(nnz / float(a.size)) if a.size else None
        }
    except Exception as e:
        return {"error": str(e)}


def _sanitize_for_json(obj: Any) -> Any:
    """Make object JSON-safe: Path->str, numpy scalars->py scalars, NaN/Inf->None."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    return str(obj)


def _norm(s: str) -> str:
    """Lowercase + remove accents + trim/collapse spaces."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_name(s: str) -> str:
    """
    Normalize a name for file/key:
    - lowercase
    - remove accents
    - replace non-alphanumeric with underscore
    - remove multiple underscores
    """
    if s is None:
        return "none"
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s == "":
        s = "unnamed"
    return s

