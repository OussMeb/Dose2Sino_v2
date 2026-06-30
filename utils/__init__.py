#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities package for preprocessing pipeline.
"""

# Patient processing and filtering utilities
from .preprocessing_utils import (
    is_patient_complete,
    filter_incomplete_patients,
    filter_valid_patient_ids,
)

# JSON and serialization utilities
from .json_utils import (
    _now_iso,
    _safe_float,
    _arr_stats,
    _sanitize_for_json,
    _norm,
    normalize_name,
)

# I/O utilities
from .io_utils import (
    _ensure_parent_dir,
    save_array_npy,
    save_array_npz,
    save_png,
    save_array_auto,
    build_x_from_saved_montages,
)

# Geometry utilities
from .geometry_utils import (
    compute_source_detector_geometry,
)

# DICOM and RT-PLAN utilities
from .dicom_utils import (
    find_rtplan_file,
    get_number_of_fractions_planned,
    extract_beam_meterset_minutes,
    extract_sinogram,
    extract_jaw_positions,
    compute_field_size_mm,
    extract_tomo_private_tags,
    save_sinogram_classic_view,
    save_sinogram_berlingo_style,
)

# PTV utilities
from .ptv_utils import (
    extract_ptv_dose_gy,
    find_ptv_candidates,
    cluster_ptvs_by_dose,
    normalize_gy_to_unit,
    dose_to_bin,
    build_ptv_channel_metadata,
)

# Quarantine utilities
from .quarantine_utils import (
    read_quarantine_index,
    analyze_quarantine,
    print_quarantine_summary,
    export_quarantine_csv,
    get_quarantine_summary_table,
)

# Mask utilities
from .mask_utils import (
    resample_to_ref,
    read_mask_array_on_ref,
    union_masks_from_paths,
    project_mask_to_filled_stack,
    project_external_entry_exit_stacks,
    apply_tomo_transform_to_stack,
    structure_set_id,
    build_struct_index,
    find_candidates,
    md5_hash,
    dice_coeff,
    choose_or_quarantine_roi,
    debug_struct_files_for_patient,
)

# Channel order utilities
from .channel_order_utils import (
    build_channel_order,
    assert_channel_order,
    save_channel_order,
    load_channel_order,
    get_ptv_channel_slice,
    validate_channel_order_consistency,
)

# Similarity validation
from .similarity_validator import (
    compute_ptv_sino_similarity,
    validate_ptv_sino_similarity,
    generate_similarity_figure,
)

# Multiprocess safety utilities
from .file_lock import (
    FileLock,
    acquire_patient_lock,
)

from .atomic_write import (
    atomic_save_npy,
    atomic_save_json,
    atomic_save_png,
    atomic_move,
)

from .patient_sentinel import (
    write_patient_sentinel,
    verify_patient_sentinel,
    get_sentinel_info,
)

__all__ = [
    # Preprocessing utils
    "is_patient_complete",
    "filter_incomplete_patients",
    "filter_valid_patient_ids",
    # JSON utils
    "_now_iso",
    "_safe_float",
    "_arr_stats",
    "_sanitize_for_json",
    "_norm",
    "normalize_name",
    # I/O utils
    "_ensure_parent_dir",
    "save_array_npy",
    "save_array_npz",
    "save_png",
    "save_array_auto",
    "build_x_from_saved_montages",
    # Geometry utils
    "compute_source_detector_geometry",
    # DICOM utils
    "find_rtplan_file",
    "get_number_of_fractions_planned",
    "extract_beam_meterset_minutes",
    "extract_sinogram",
    "extract_jaw_positions",
    "compute_field_size_mm",
    "extract_tomo_private_tags",
    "save_sinogram_classic_view",
    "save_sinogram_berlingo_style",
    # PTV utils
    "extract_ptv_dose_gy",
    "find_ptv_candidates",
    "cluster_ptvs_by_dose",
    "normalize_gy_to_unit",
    "dose_to_bin",
    "build_ptv_channel_metadata",
    # Quarantine utils
    "read_quarantine_index",
    "analyze_quarantine",
    "print_quarantine_summary",
    "export_quarantine_csv",
    "get_quarantine_summary_table",
    # Mask utils
    "resample_to_ref",
    "read_mask_array_on_ref",
    "union_masks_from_paths",
    "project_mask_to_filled_stack",
    "project_external_entry_exit_stacks",
    "apply_tomo_transform_to_stack",
    "structure_set_id",
    "build_struct_index",
    "find_candidates",
    "md5_hash",
    "dice_coeff",
    "choose_or_quarantine_roi",
    "debug_struct_files_for_patient",
    # Channel order utilities
    "build_channel_order",
    "assert_channel_order",
    "save_channel_order",
    "load_channel_order",
    "get_ptv_channel_slice",
    "validate_channel_order_consistency",
    # Similarity validation
    "compute_ptv_sino_similarity",
    "validate_ptv_sino_similarity",
    "generate_similarity_figure",
    # Multiprocess safety
    "FileLock",
    "acquire_patient_lock",
    "atomic_save_npy",
    "atomic_save_json",
    "atomic_save_png",
    "atomic_move",
    "write_patient_sentinel",
    "verify_patient_sentinel",
    "get_sentinel_info",
]

