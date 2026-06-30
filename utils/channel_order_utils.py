#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Channel ordering utilities for strict, unique, and stable channel ordering.

Ensures that all patient files are generated with:
1. PTV channels (ptv_br, ptv_ri, ptv_hr) at positions 0, 1, 2 (in that order)
2. Other structures in a fixed order
3. Each patient has a channel_order.json documenting the exact channel order

This module replaces the fragile "expected_names" construction scattered in pipeline.py
with centralized, testable functions.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# PTV channel names in strict order
PTV_CHANNEL_NAMES = ["ptv_br", "ptv_ri", "ptv_hr"]
PTV_CHANNEL_COUNT = len(PTV_CHANNEL_NAMES)


def build_channel_order(
    ptv_mode: str,
    base_order: List[str]
) -> List[str]:
    """
    Build the final channel order for a patient.

    The base_order should be a canonical order that includes all structures
    with PTV channels at the beginning, followed by external encoding,
    then OARs and groups.

    Args:
        ptv_mode: Processing mode ("separate", "both", "none", etc.)
        base_order: Canonical base order (typically OAR_AND_GROUPS_ORDER from config)

    Returns:
        Final list of channel names in the required order

    Raises:
        ValueError: If the base_order doesn't have PTV channels at the beginning
                   when ptv_mode requires them
    """
    # Validate base order structure
    if not base_order:
        raise ValueError("base_order cannot be empty")

    # Check that PTV channels are at the beginning of base_order
    if len(base_order) >= PTV_CHANNEL_COUNT:
        actual_ptv = base_order[:PTV_CHANNEL_COUNT]
        if actual_ptv != PTV_CHANNEL_NAMES:
            raise ValueError(
                f"base_order must start with {PTV_CHANNEL_NAMES}, "
                f"but got {actual_ptv}"
            )
    else:
        raise ValueError(
            f"base_order must have at least {PTV_CHANNEL_COUNT} elements "
            f"(PTV channels), got {len(base_order)}"
        )

    # Determine which channels to include based on ptv_mode
    if ptv_mode in {"separate", "both"}:
        # Include PTV channels: base_order is already correct
        order = list(base_order)
    elif ptv_mode in {"none", "exclude"}:
        # Exclude PTV channels: remove from base_order
        order = [ch for ch in base_order if ch not in PTV_CHANNEL_NAMES]
    else:
        # Unknown mode: default to including base_order as-is
        logger.warning(f"Unknown ptv_mode '{ptv_mode}', using full base_order")
        order = list(base_order)

    return order


def assert_channel_order(
    order: List[str],
    require_ptv: bool = True
) -> None:
    """
    Validate channel order coherence and safety.

    Checks:
    - No duplicates
    - If require_ptv=True: order[0:3] == ["ptv_br", "ptv_ri", "ptv_hr"]
    - (Optional) external_entry and external_exit are present

    Args:
        order: List of channel names
        require_ptv: If True, enforce PTV-first constraint

    Raises:
        AssertionError: If order is invalid
    """
    if not order:
        raise AssertionError("Channel order cannot be empty")

    # Check for duplicates
    if len(order) != len(set(order)):
        duplicates = [ch for ch in set(order) if order.count(ch) > 1]
        raise AssertionError(f"Duplicate channels in order: {duplicates}")

    # Check PTV constraint if required
    if require_ptv:
        if len(order) < PTV_CHANNEL_COUNT:
            raise AssertionError(
                f"Order must have at least {PTV_CHANNEL_COUNT} channels "
                f"when require_ptv=True, got {len(order)}"
            )

        actual_ptv = order[:PTV_CHANNEL_COUNT]
        if actual_ptv != PTV_CHANNEL_NAMES:
            raise AssertionError(
                f"First {PTV_CHANNEL_COUNT} channels must be {PTV_CHANNEL_NAMES}, "
                f"got {actual_ptv}"
            )

    # (Optional) Check for external encoding channels
    external_channels = {"external_entry", "external_exit"}
    missing_external = external_channels - set(order)
    if missing_external:
        logger.warning(
            f"Channel order missing external encoding channels: {missing_external}. "
            f"This may be expected if they are generated separately."
        )


def save_channel_order(
    patient_out_dir: Path,
    order: List[str]
) -> None:
    """
    Save channel order to patient_out_dir/channel_order.json.

    This allows downstream code (especially similarity validation)
    to know the exact channel ordering without guessing.

    Args:
        patient_out_dir: Patient output directory
        order: List of channel names

    Raises:
        ValueError: If order is invalid
    """
    patient_out_dir = Path(patient_out_dir)

    # Validate order before saving
    try:
        assert_channel_order(order, require_ptv=True)
    except AssertionError as e:
        raise ValueError(f"Cannot save invalid channel order: {e}")

    # Create directory if needed
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    output_path = patient_out_dir / "channel_order.json"
    data = {"order": order, "ptv_count": PTV_CHANNEL_COUNT}

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved channel order to {output_path}")
    except IOError as e:
        raise ValueError(f"Failed to save channel order: {e}")


def load_channel_order(patient_out_dir: Path) -> Optional[List[str]]:
    """
    Load channel order from patient_out_dir/channel_order.json.

    Returns None if the file doesn't exist or if it's invalid.

    Args:
        patient_out_dir: Patient output directory

    Returns:
        List of channel names, or None if not found/invalid
    """
    patient_out_dir = Path(patient_out_dir)
    input_path = patient_out_dir / "channel_order.json"

    if not input_path.exists():
        logger.debug(f"Channel order file not found: {input_path}")
        return None

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract order (handle both old and new format)
        if isinstance(data, dict) and "order" in data:
            order = data["order"]
        elif isinstance(data, list):
            order = data
        else:
            logger.error(f"Invalid channel_order.json format: {data}")
            return None

        # Validate loaded order
        try:
            assert_channel_order(order, require_ptv=True)
        except AssertionError as e:
            logger.error(f"Loaded channel order is invalid: {e}")
            return None

        return order

    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load channel order: {e}")
        return None


def get_ptv_channel_slice(order: List[str]) -> Tuple[int, int]:
    """
    Get the (start, end) slice indices for PTV channels.

    Finds the indices of ptv_br, ptv_ri, ptv_hr in the order
    and verifies they are consecutive and in the correct order.

    Args:
        order: List of channel names

    Returns:
        Tuple (start, end) with end exclusive. Typically (0, 3).

    Raises:
        ValueError: If PTV channels are not found or not consecutive
    """
    indices = []
    for ptv_name in PTV_CHANNEL_NAMES:
        try:
            idx = order.index(ptv_name)
            indices.append(idx)
        except ValueError:
            raise ValueError(
                f"PTV channel '{ptv_name}' not found in order: {order}"
            )

    # Verify consecutive and in correct order
    if indices != sorted(indices):
        raise ValueError(
            f"PTV channels not in order: {list(zip(PTV_CHANNEL_NAMES, indices))}"
        )

    if indices != list(range(indices[0], indices[0] + PTV_CHANNEL_COUNT)):
        raise ValueError(
            f"PTV channels not consecutive: {list(zip(PTV_CHANNEL_NAMES, indices))}"
        )

    start = indices[0]
    end = start + PTV_CHANNEL_COUNT

    return (start, end)


def validate_channel_order_consistency(
    order_list: List[List[str]],
    patient_ids: Optional[List[str]] = None
) -> bool:
    """
    Validate that all orders in a list are identical.

    Useful for checking consistency across multiple patients.

    Args:
        order_list: List of channel order lists
        patient_ids: Optional list of patient IDs for logging

    Returns:
        True if all orders are identical, False otherwise
    """
    if not order_list:
        logger.warning("order_list is empty")
        return True

    canonical = order_list[0]

    for i, order in enumerate(order_list[1:], start=1):
        if order != canonical:
            patient_info = f" (patient {patient_ids[i]})" if patient_ids else ""
            logger.error(
                f"Channel order mismatch{patient_info}: "
                f"expected {canonical}, got {order}"
            )
            return False

    logger.info(f"✅ All {len(order_list)} channel orders are consistent")
    return True

