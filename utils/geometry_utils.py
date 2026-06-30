#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry utilities for source-detector configuration.
"""
import numpy as np


def compute_source_detector_geometry(
    x_iso: float,
    y_iso: float,
    z_iso: float,
    theta_deg: float,
    delta_z: float,
    sad_mm: float,
    sdd_mm: float,
    width_mm: float,
    height_mm: float,
    nx: int,
    ny: int,
    angle_zero_is_anterior: bool = True,
    flip_lr: bool = True,
    flip_ud: bool = False
) -> tuple:
    """
    Compute source and detector geometry for a given angle.

    Args:
        x_iso, y_iso, z_iso: Isocenter coordinates (mm)
        theta_deg: Gantry angle (degrees)
        delta_z: Table position (mm)
        sad_mm: Source-axis distance (mm)
        sdd_mm: Source-detector distance (mm)
        width_mm, height_mm: Detector dimensions (mm)
        nx, ny: Detector resolution (pixels)
        angle_zero_is_anterior: If True, 0° = anterior (AP)
        flip_lr: Flip left-right
        flip_ud: Flip up-down

    Returns:
        (src, det_ctr, u, v, su, sv): Source position, detector center,
                                       u/v vectors, and pixel spacings
    """
    z_corr = z_iso - delta_z
    th = np.deg2rad(theta_deg)

    sx = x_iso + sad_mm * np.sin(th)

    if angle_zero_is_anterior:
        sy = y_iso - sad_mm * np.cos(th)  # 0° = source anterior (AP)
    else:
        sy = y_iso + sad_mm * np.cos(th)  # 0° = source posterior (PA)

    src = np.array([sx, sy, z_corr], dtype=np.float32)

    dir_iso = np.array([x_iso - src[0], y_iso - src[1], z_corr - src[2]], dtype=np.float32)
    dir_iso = dir_iso / (np.linalg.norm(dir_iso) + 1e-8)

    det_ctr = src + dir_iso * sdd_mm

    arb = np.array([0, 0, 1], dtype=np.float32)
    if abs(np.dot(arb, dir_iso)) > 0.999:
        arb = np.array([0, 1, 0], dtype=np.float32)

    u = np.cross(dir_iso, arb)
    u = u / (np.linalg.norm(u) + 1e-8)

    v = np.cross(dir_iso, u)
    v = v / (np.linalg.norm(v) + 1e-8)

    su = (sdd_mm / sad_mm * width_mm) / nx
    sv = (sdd_mm / sad_mm * height_mm) / ny

    if flip_lr:
        su = -su
    if flip_ud:
        sv = -sv

    return src, det_ctr, u, v, su, sv

