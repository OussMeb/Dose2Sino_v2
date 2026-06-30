#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Similarity validation for X_montage vs sinogram.

Provides automatic quality control by comparing PTV projections
with the sinogram to detect geometric/projection errors.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Any, List
import warnings
import threading

# Import channel order utilities for robust channel detection
try:
    from .channel_order_utils import (
        load_channel_order,
        get_ptv_channel_slice,
    )
    HAS_CHANNEL_ORDER_UTILS = True
except ImportError:
    HAS_CHANNEL_ORDER_UTILS = False
    logging.warning("channel_order_utils not available - will use hardcoded ptv_channels")
    # Stub definitions when import fails
    def load_channel_order(patient_out_dir):
        return None
    def get_ptv_channel_slice(order):
        raise ValueError("channel_order_utils not available")

# Matplotlib for figure generation (optional, catches import errors)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving figures
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None  # type: ignore
    plt = None  # type: ignore
    HAS_MATPLOTLIB = False
    logging.warning("matplotlib not available - similarity figures will not be generated")

# Global lock for matplotlib operations to prevent race conditions in parallel jobs
_matplotlib_lock = threading.Lock()


def resize_detectors_linear(sino_cp_det: np.ndarray, out_det: int) -> np.ndarray:
    """
    Linear interpolation on detector axis.
    Input shape: (rows, det_in)
    Output shape: (rows, out_det)
    """
    rows, det_in = sino_cp_det.shape
    if det_in == out_det:
        return sino_cp_det.astype(np.float32, copy=False)

    x_in = np.linspace(0.0, 1.0, det_in, endpoint=True)
    x_out = np.linspace(0.0, 1.0, out_det, endpoint=True)

    out = np.empty((rows, out_det), dtype=np.float32)
    sino_cp_det = sino_cp_det.astype(np.float32, copy=False)

    for i in range(rows):
        out[i] = np.interp(x_out, x_in, sino_cp_det[i])
    return out


def infer_ny_per_cp(x_len: int, sino_cp: int, prefer: int = 12) -> int:
    """
    Infer NY (rows per CP) from X_montage and sino dimensions.
    """
    x_len = int(x_len)
    sino_cp = int(sino_cp)

    if sino_cp > 0 and x_len % sino_cp == 0:
        ny = x_len // sino_cp
        return max(1, int(ny))

    if prefer > 0 and x_len % prefer == 0:
        cp_try = x_len // prefer
        if cp_try == sino_cp:
            return int(prefer)

    best = None
    max_ny_try = min(512, x_len) if x_len > 0 else 1
    for ny_try in range(1, max_ny_try + 1):
        if x_len % ny_try != 0:
            continue
        cp_try = x_len // ny_try
        absdiff = abs(cp_try - sino_cp)
        if best is None or absdiff < best[0]:
            best = (absdiff, ny_try, cp_try)
            if absdiff == 0:
                break

    if best is not None:
        return max(1, int(best[1]))

    if prefer > 0 and x_len % prefer == 0:
        return int(prefer)
    return 1


def compute_ptv_sino_similarity(
    X_montage: np.ndarray,
    sino: np.ndarray,
    ptv_channels: Optional[Tuple[int, int]] = None,
    channel_order: Optional[List[str]] = None,
    method: str = "correlation"
) -> Dict[str, Any]:
    """
    Compute similarity metrics between PTV channels of X_montage and sinogram.

    Args:
        X_montage: Shape (C, H, W) where H = CP * NY
        sino: Shape (CP, det_in)
        ptv_channels: Tuple (start, end) for PTV channel range.
                     If None and channel_order is provided, computed from channel_order.
                     If both None, defaults to (0, 3) with a warning.
        channel_order: List of channel names. If provided, used to compute ptv_channels dynamically.
        method: Similarity method ("correlation", "mse", "ssim", "all")

    Returns:
        Dict with similarity metrics:
        - correlation: Pearson correlation coefficient
        - mse: Mean squared error (lower is better)
        - overlap_ratio: Ratio of overlapping non-zero regions
        - pattern_match: Custom metric for sinusoidal pattern matching
    """
    # Determine PTV channels from channel_order if available
    if ptv_channels is None:
        if channel_order is not None and HAS_CHANNEL_ORDER_UTILS:
            try:
                ptv_channels = get_ptv_channel_slice(channel_order)
                logging.debug(f"Computed PTV channels from channel_order: {ptv_channels}")
            except ValueError as e:
                logging.warning(f"Failed to compute PTV channels from channel_order: {e}. Using default (0, 3).")
                ptv_channels = (0, 3)
        else:
            if channel_order is not None:
                logging.warning("channel_order provided but channel_order_utils not available. Using default ptv_channels=(0, 3).")
            else:
                logging.warning("No channel_order provided and ptv_channels not specified. Using default (0, 3). "
                               "This assumes PTV channels are at indices 0:3, which may not be accurate.")
            ptv_channels = (0, 3)

    C, Xlen, det_out = X_montage.shape
    sino_cp, det_in = sino.shape

    # Infer NY
    ny = infer_ny_per_cp(Xlen, sino_cp)

    # Extract PTV channels and compute mean
    ch_start, ch_end = ptv_channels
    ch_end = min(ch_end, C)
    if ch_start >= ch_end:
        return {"error": "invalid_channels", "correlation": 0.0}

    ptv_subset = X_montage[ch_start:ch_end, :, :]
    ptv_mean = np.mean(ptv_subset, axis=0).astype(np.float32)  # (H, W)

    # Resize sinogram to match X_montage dimensions
    sino_rep = np.repeat(sino, ny, axis=0).astype(np.float32)  # (CP*NY, det_in)
    sino_resized = resize_detectors_linear(sino_rep, out_det=det_out)  # (CP*NY, det_out)

    # Crop to common size
    min_h = min(ptv_mean.shape[0], sino_resized.shape[0])
    min_w = min(ptv_mean.shape[1], sino_resized.shape[1])
    ptv_crop = ptv_mean[:min_h, :min_w]
    sino_crop = sino_resized[:min_h, :min_w]

    # Flatten for correlation
    ptv_flat = ptv_crop.flatten()
    sino_flat = sino_crop.flatten()

    results = {}

    # 1. Pearson correlation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ptv_flat.std() > 1e-8 and sino_flat.std() > 1e-8:
            corr = np.corrcoef(ptv_flat, sino_flat)[0, 1]
            results["correlation"] = float(corr) if np.isfinite(corr) else 0.0
        else:
            results["correlation"] = 0.0

    # 2. MSE (normalized)
    mse = np.mean((ptv_crop - sino_crop) ** 2)
    results["mse"] = float(mse)

    # 3. Overlap ratio (how much of sino non-zero is covered by PTV)
    sino_mask = sino_crop > 0.1
    ptv_mask = ptv_crop > 0.1

    if sino_mask.sum() > 0:
        overlap = np.logical_and(sino_mask, ptv_mask).sum()
        results["overlap_ratio"] = float(overlap / sino_mask.sum())
    else:
        results["overlap_ratio"] = 0.0

    # 4. Pattern match: correlation on binarized images
    sino_bin = (sino_crop > 0.1).astype(np.float32)
    ptv_bin = (ptv_crop > 0.1).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sino_bin.std() > 1e-8 and ptv_bin.std() > 1e-8:
            pattern_corr = np.corrcoef(sino_bin.flatten(), ptv_bin.flatten())[0, 1]
            results["pattern_match"] = float(pattern_corr) if np.isfinite(pattern_corr) else 0.0
        else:
            results["pattern_match"] = 0.0

    # 5. Centroid alignment (check if wave patterns are aligned)
    try:
        # For each row (detector), find centroid of non-zero region
        sino_centroids = []
        ptv_centroids = []

        for i in range(min_h):
            s_row = sino_crop[i, :]
            p_row = ptv_crop[i, :]

            s_nz = np.where(s_row > 0.1)[0]
            p_nz = np.where(p_row > 0.1)[0]

            if len(s_nz) > 0 and len(p_nz) > 0:
                sino_centroids.append(s_nz.mean())
                ptv_centroids.append(p_nz.mean())

        if len(sino_centroids) > 10:
            centroid_corr = np.corrcoef(sino_centroids, ptv_centroids)[0, 1]
            results["centroid_alignment"] = float(centroid_corr) if np.isfinite(centroid_corr) else 0.0
        else:
            results["centroid_alignment"] = 0.0
    except Exception:
        results["centroid_alignment"] = 0.0

    # 6. Composite score (weighted average)
    w_corr = 0.3
    w_overlap = 0.3
    w_pattern = 0.2
    w_centroid = 0.2

    composite = (
        w_corr * max(0, results["correlation"]) +
        w_overlap * results["overlap_ratio"] +
        w_pattern * max(0, results["pattern_match"]) +
        w_centroid * max(0, results["centroid_alignment"])
    )
    results["composite_score"] = float(composite)

    return results


def generate_similarity_figure(
    X_montage: np.ndarray,
    sino: np.ndarray,
    output_path: Path,
    ptv_channels: Optional[Tuple[int, int]] = None,
    channel_order: Optional[List[str]] = None,
    similarity_metrics: Optional[Dict] = None,
    patient_id: str = "unknown"
) -> bool:
    """
    Generate a comparison figure showing PTV vs sinogram with similarity metrics.

    Args:
        X_montage: Channel montage array
        sino: Sinogram array
        output_path: Where to save the figure
        ptv_channels: PTV channel slice. If None and channel_order provided, computed from channel_order.
        channel_order: List of channel names for dynamic PTV detection
        similarity_metrics: Pre-computed similarity metrics dict
        patient_id: Patient ID for title

    Returns True if figure was successfully saved.
    Thread-safe: uses a global lock to prevent matplotlib race conditions in parallel jobs.
    """
    if not HAS_MATPLOTLIB:
        logging.warning("Cannot generate similarity figure: matplotlib not available")
        return False

    # Use global lock to prevent matplotlib race conditions in parallel jobs
    with _matplotlib_lock:
        return _generate_similarity_figure_unsafe(
            X_montage, sino, output_path, ptv_channels=ptv_channels,
            channel_order=channel_order, similarity_metrics=similarity_metrics, patient_id=patient_id
        )


def _generate_similarity_figure_unsafe(
    X_montage: np.ndarray,
    sino: np.ndarray,
    output_path: Path,
    ptv_channels: Optional[Tuple[int, int]] = None,
    channel_order: Optional[List[str]] = None,
    similarity_metrics: Optional[Dict] = None,
    patient_id: str = "unknown"
) -> bool:
    """
    Internal function: generate figure WITHOUT locking (call generate_similarity_figure instead).

    Returns True if figure was successfully saved.
    """
    if not HAS_MATPLOTLIB:
        logging.warning("Cannot generate similarity figure: matplotlib not available")
        return False

    try:
        # Determine PTV channels from channel_order if needed
        if ptv_channels is None:
            if channel_order is not None and HAS_CHANNEL_ORDER_UTILS:
                try:
                    ptv_channels = get_ptv_channel_slice(channel_order)
                    logging.debug(f"Computed PTV channels from channel_order: {ptv_channels}")
                except ValueError as e:
                    logging.warning(f"Failed to compute PTV channels: {e}. Using default (0, 3).")
                    ptv_channels = (0, 3)
            else:
                logging.debug("Using default ptv_channels=(0, 3)")
                ptv_channels = (0, 3)

        # Extract dimensions
        C, Xlen, det_out = X_montage.shape
        sino_cp, det_in = sino.shape

        # Infer NY (rows per control point)
        ny = infer_ny_per_cp(Xlen, sino_cp)

        # Extract PTV mean
        ch_start, ch_end = ptv_channels
        ch_end = min(ch_end, C)
        ptv_subset = X_montage[ch_start:ch_end, :, :]
        ptv_mean = np.mean(ptv_subset, axis=0).astype(np.float32)

        # Resize sino
        sino_rep = np.repeat(sino, ny, axis=0).astype(np.float32)
        sino_resized = resize_detectors_linear(sino_rep, out_det=det_out)

        # Crop
        min_h = min(ptv_mean.shape[0], sino_resized.shape[0])
        min_w = min(ptv_mean.shape[1], sino_resized.shape[1])
        ptv_crop = ptv_mean[:min_h, :min_w]
        sino_crop = sino_resized[:min_h, :min_w]

        # Create figure with 4 subplots (vertical stack)
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))

        # Metrics text
        if similarity_metrics:
            metrics_text = (
                f"Correlation: {similarity_metrics.get('correlation', 0):.3f} | "
                f"Overlap: {similarity_metrics.get('overlap_ratio', 0):.3f} | "
                f"Pattern: {similarity_metrics.get('pattern_match', 0):.3f} | "
                f"Centroid: {similarity_metrics.get('centroid_alignment', 0):.3f} | "
                f"COMPOSITE: {similarity_metrics.get('composite_score', 0):.3f}"
            )
        else:
            metrics_text = "No metrics available"

        fig.suptitle(f"Similarity Analysis: {patient_id}\n{metrics_text}", fontsize=14, fontweight='bold')

        # Calculate dynamic contrast limits based on percentiles
        # Use 2nd and 98th percentile to improve contrast while avoiding outliers
        ptv_p2, ptv_p98 = np.percentile(ptv_crop, [2, 98])
        sino_p2, sino_p98 = np.percentile(sino_crop, [2, 98])

        # Ensure min != max to avoid matplotlib warnings
        ptv_vmin = ptv_p2 if ptv_p98 > ptv_p2 else 0
        ptv_vmax = ptv_p98 if ptv_p98 > ptv_p2 else 1
        sino_vmin = sino_p2 if sino_p98 > sino_p2 else 0
        sino_vmax = sino_p98 if sino_p98 > sino_p2 else 1

        # Plot 1: PTV mean with magnification (zoom 2x using cubic interpolation)
        from scipy.ndimage import zoom
        zoom_factor = 2.0
        ptv_zoomed = zoom(ptv_crop, zoom_factor, order=3)  # cubic interpolation
        im1 = axes[0].imshow(ptv_zoomed.T, aspect='auto', origin='lower', cmap='viridis', vmin=ptv_vmin, vmax=ptv_vmax)
        axes[0].set_title(f"PTV Mean ×{zoom_factor:.0f} (channels {ch_start}:{ch_end})", fontweight='bold')
        axes[0].set_xlabel("CP × NY (magnified)")
        axes[0].set_ylabel("Detectors (magnified)")
        plt.colorbar(im1, ax=axes[0], fraction=0.02)

        # Plot 2: Sinogram (original size)
        im2 = axes[1].imshow(sino_crop.T, aspect='auto', origin='lower', cmap='viridis', vmin=sino_vmin, vmax=sino_vmax)
        axes[1].set_title(f"Sinogram (resized {det_in}→{det_out} det)", fontweight='bold')
        axes[1].set_xlabel("CP × NY")
        axes[1].set_ylabel("Detectors")
        plt.colorbar(im2, ax=axes[1], fraction=0.02)

        # Plot 3: Overlay (RGB) - alignment visualization
        ptv_norm = np.clip(ptv_crop, 0, 1)
        sino_norm = np.clip(sino_crop, 0, 1)

        overlay = np.zeros((ptv_norm.shape[0], ptv_norm.shape[1], 3), dtype=np.float32)
        overlay[..., 0] = ptv_norm  # Red = PTV
        overlay[..., 1] = sino_norm  # Green = Sino
        overlay[..., 2] = 0.7 * (ptv_norm * sino_norm)  # Blue = overlap

        axes[2].imshow(overlay.transpose(1, 0, 2), aspect='auto', origin='lower')
        axes[2].set_title("Overlay (R=PTV, G=Sino, B=Overlap)", fontweight='bold')
        axes[2].set_xlabel("CP × NY")
        axes[2].set_ylabel("Detectors")

        # Plot 4: Gamma Index (geometric error/difference map)
        # Simpler version: just show normalized absolute difference
        try:
            # Calculate normalized difference (0-1 range)
            diff = np.abs(ptv_norm - sino_norm)

            # Already in 0-1 range, just scale to 0-100 for display
            error_map = diff * 100  # Convert to percentage (0-100%)

            # Create custom colormap: green (perfect) -> yellow -> orange -> red (bad)
            from matplotlib.colors import LinearSegmentedColormap
            colors_gamma = ['green', 'limegreen', 'yellow', 'orange', 'red']
            cmap_gamma = LinearSegmentedColormap.from_list('error', colors_gamma, N=100)

            im4 = axes[3].imshow(error_map.T, aspect='auto', origin='lower', cmap=cmap_gamma, vmin=0, vmax=100)
            axes[3].set_title("Error Map (0-100%) - Green=Perfect, Red=Maximum Diff", fontweight='bold')
            axes[3].set_xlabel("CP × NY")
            axes[3].set_ylabel("Detectors")
            cbar = plt.colorbar(im4, ax=axes[3], fraction=0.02)
            cbar.set_label("Error (%)")

            # Calculate statistics
            mean_error = np.mean(error_map)
            max_error = np.max(error_map)
            pass_ratio = np.sum(error_map < 10) / error_map.size * 100  # Pass if < 10% diff

            # Add stats to image
            stats_text = f"Mean: {mean_error:.1f}% | Max: {max_error:.1f}% | Pass(<10%): {pass_ratio:.1f}%"
            axes[3].text(0.02, 0.98, stats_text,
                        transform=axes[3].transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        except Exception as e:
            logging.warning(f"[SIMILARITY] Failed to compute error map: {e}")
            axes[3].text(0.5, 0.5, f"Error Map\n(computation failed)",
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title("Error Map - Error", fontweight='bold')

        plt.tight_layout()

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # CRITICAL FIX #3: Use worker-specific temp file names to avoid conflicts
        # Each worker gets a unique identifier (PID + timestamp) so temp files don't collide
        import tempfile
        import shutil
        import os
        import time

        worker_id = f"{os.getpid()}_{int(time.time() * 1000000) % 1000000}"
        temp_filename = f".similarity_{worker_id}.png"
        tmp_path = output_path.parent / temp_filename

        try:
            plt.savefig(str(tmp_path), dpi=150, bbox_inches='tight')
            # Atomically rename temp file to final location
            # This is atomic on POSIX systems, preventing partial writes
            tmp_path.rename(output_path)
            logging.debug(f"[SIMILARITY] Figure saved: {output_path} (worker {worker_id})")
        except Exception as save_err:
            logging.error(f"[SIMILARITY] Failed to save figure: {save_err}")
            # Clean up temp file on error
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise
        finally:
            # Always close the figure to free matplotlib resources
            plt.close(fig)

            # Final cleanup of any leftover temp files from this worker
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

        return True

    except Exception as e:
        logging.error(f"[SIMILARITY] Failed to generate figure: {e}")
        # CRITICAL FIX #2b: Close ONLY our figure, not all figures globally
        # This prevents affecting other workers' pyplot state
        try:
            if 'fig' in locals():
                plt.close(fig)
        except:
            pass
        return False


def validate_ptv_sino_similarity(
    patient_out_dir: Path,
    threshold_composite: float = 0.4,
    threshold_correlation: float = 0.3,
    ptv_channels: Optional[Tuple[int, int]] = None,
    channel_order: Optional[List[str]] = None,
    save_figure_on_failure: bool = True,
    save_figure_always: bool = False,
    patient_id: str = "unknown"
) -> Tuple[bool, Dict]:
    """
    Validate similarity between PTV projections and sinogram.

    Args:
        patient_out_dir: Path to patient output directory containing X_montage.npy and sino.npy
        threshold_composite: Minimum composite score to pass (default 0.4)
        threshold_correlation: Minimum correlation to pass (default 0.3)
        ptv_channels: Tuple (start, end) for PTV channels.
                     If None, will attempt to load from channel_order.json or use default (0,3).
        channel_order: List of channel names. If None, will attempt to load from channel_order.json.
        save_figure_on_failure: Save comparison figure if validation fails
        save_figure_always: Always save comparison figure
        patient_id: Patient ID for logging and figure titles

    Returns:
        Tuple (passed: bool, metrics: dict)
    """
    patient_out_dir = Path(patient_out_dir)

    # Try to load channel_order if not provided
    if channel_order is None and HAS_CHANNEL_ORDER_UTILS:
        channel_order = load_channel_order(patient_out_dir)
        if channel_order is not None:
            logging.debug(f"Loaded channel_order from JSON: {channel_order}")

    x_path = patient_out_dir / "X_montage.npy"
    sino_path = patient_out_dir / "sino.npy"

    if not x_path.exists():
        logging.error(f"[SIMILARITY] X_montage.npy not found: {x_path}")
        return False, {"error": "x_montage_missing"}

    if not sino_path.exists():
        logging.error(f"[SIMILARITY] sino.npy not found: {sino_path}")
        return False, {"error": "sino_missing"}

    try:
        # CRITICAL FIX #1: Force deep copy to prevent multiprocessing data sharing
        # Without .copy(), different workers might alias the same array in memory,
        # causing identical data to be processed by multiple workers (multiprocess bug)
        X_montage = np.load(x_path).copy()
        sino = np.load(sino_path).copy()

        # Log checksums for debugging multiprocess issues
        import hashlib
        x_checksum = hashlib.sha256(X_montage.tobytes()).hexdigest()[:8]
        sino_checksum = hashlib.sha256(sino.tobytes()).hexdigest()[:8]
        logging.debug(f"[SIMILARITY] Loaded data: X_montage checksum={x_checksum}, sino checksum={sino_checksum}")
    except Exception as e:
        logging.error(f"[SIMILARITY] Failed to load arrays: {e}")
        return False, {"error": f"load_failed: {e}"}

    # Validate shapes
    if X_montage.ndim != 3:
        return False, {"error": f"X_montage wrong dims: {X_montage.shape}"}
    if sino.ndim != 2:
        return False, {"error": f"sino wrong dims: {sino.shape}"}

    # Compute metrics with dynamic PTV channel detection
    metrics = compute_ptv_sino_similarity(
        X_montage, sino,
        ptv_channels=ptv_channels,
        channel_order=channel_order,
    )

    # Check thresholds
    composite = metrics.get("composite_score", 0)
    correlation = metrics.get("correlation", 0)

    passed = (composite >= threshold_composite) and (correlation >= threshold_correlation)

    metrics["passed"] = passed
    metrics["threshold_composite"] = threshold_composite
    metrics["threshold_correlation"] = threshold_correlation

    # Generate figure if needed
    if save_figure_always or (not passed and save_figure_on_failure):
        figure_path = patient_out_dir / "similarity_comparison.png"
        generate_similarity_figure(
            X_montage, sino, figure_path,
            ptv_channels=ptv_channels,
            channel_order=channel_order,
            similarity_metrics=metrics,
            patient_id=patient_id
        )
        metrics["figure_path"] = str(figure_path)

    # Save metrics to JSON file for analysis
    try:
        metrics_json_path = patient_out_dir / "similarity_metrics.json"
        # Create a JSON-serializable version of metrics
        metrics_for_json = {}
        for key, val in metrics.items():
            if isinstance(val, (int, float, str, bool, type(None))):
                metrics_for_json[key] = val
            elif isinstance(val, np.floating):
                metrics_for_json[key] = float(val)
            elif isinstance(val, np.integer):
                metrics_for_json[key] = int(val)
            else:
                metrics_for_json[key] = str(val)

        import json
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_for_json, f, indent=2, ensure_ascii=False)
        logging.info(f"[SIMILARITY] Metrics saved: {metrics_json_path}")
    except Exception as e:
        logging.warning(f"[SIMILARITY] Failed to save metrics JSON: {e}")

    if passed:
        logging.info(f"[SIMILARITY] ✅ {patient_id}: PASSED (composite={composite:.3f}, corr={correlation:.3f})")
    else:
        logging.warning(f"[SIMILARITY] ❌ {patient_id}: FAILED (composite={composite:.3f}, corr={correlation:.3f})")

    return passed, metrics


# For command-line testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python similarity_validator.py <patient_dir>")
        sys.exit(1)

    patient_dir = Path(sys.argv[1])
    passed, metrics = validate_ptv_sino_similarity(
        patient_dir,
        save_figure_always=True,
        patient_id=patient_dir.name
    )

    print(f"\nResults for {patient_dir.name}:")
    print(f"  Passed: {passed}")
    for k, v in metrics.items():
        if k != "passed":
            print(f"  {k}: {v}")

