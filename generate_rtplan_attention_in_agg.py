#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_rtplan_attention_in_agg_all_paretos.py

Adapted RTPLAN injector for the accepted baseline:
    models.unet_attention_in_agg.DosePredictionAttentionInAgg

Supports:
    - nested patient path:
        /mnt/data/shared/tomo_data/<PATIENT_ID>/RC_Publi_Tomo_Halcyon_DIBH/Tomo_FB_copy/
    - single Pareto injection
    - all Pareto folders in one run
    - model inference + injection
    - optional direct .npy injection for one Pareto

Private Tomo/Radixact LOT sinogram tag:
    (300D,10A7)

Single Pareto:
    python generate_rtplan_attention_in_agg_all_paretos.py \
      --patient-dir /mnt/data/shared/tomo_data/297768/RC_Publi_Tomo_Halcyon_DIBH/Tomo_FB_copy \
      --pareto-index 0 \
      --checkpoint /path/to/best_model.pth \
      --output /home/oussama/Desktop/Project/raystation_exports/297768_p0_ai_rtplan.dcm \
      --base-filters 16 \
      --device cuda \
      --save-npy \
      --save-preview

All Paretos:
    python generate_rtplan_attention_in_agg_all_paretos.py \
      --patient-dir /mnt/data/shared/tomo_data/297768/RC_Publi_Tomo_Halcyon_DIBH/Tomo_FB_copy \
      --all-paretos \
      --checkpoint /path/to/best_model.pth \
      --output /home/oussama/Desktop/Project/raystation_exports/297768_all_paretos \
      --base-filters 16 \
      --device cuda \
      --save-npy \
      --save-preview
"""

from __future__ import annotations

import argparse
import copy
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch
from pydicom.uid import generate_uid

from models.unet_attention_in_agg import DosePredictionAttentionInAgg
from utils.patient import RTDataset


SINO_TAG = (0x300D, 0x10A7)

# Tomo3 machine constraints
TOMO3_MAX_LEAF_CYCLES_PER_SECOND = 160  # Radixact allows 270
TOMO3_MIN_LEAF_OPEN_MS = 20.0           # minimum duration a leaf must stay open
DEFAULT_CP_DURATION_MS = 300.0          # ~300 ms per control point on Tomo3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def count_leaf_cycles(sino: np.ndarray) -> np.ndarray:
    """Count leaf open/close transitions per control point row.

    A leaf cycle is counted each time a leaf goes from closed (0) to open (>0)
    looking at adjacent CPs. Returns array of shape [N_CP] with cycle counts.
    The first CP is compared against an implicit all-zero prior row.
    """
    _, n_leaves = sino.shape
    open_mask = sino > 0.0
    prior = np.zeros((1, n_leaves), dtype=bool)
    padded = np.vstack([prior, open_mask])  # [N_CP+1, 64]
    # A new cycle starts when a leaf transitions closed→open
    transitions = (~padded[:-1]) & padded[1:]  # [N_CP, 64]
    return transitions.sum(axis=1).astype(np.int32)


def apply_leaf_noise_threshold(sino: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    """Zero out leaf values below threshold to suppress spurious leaf cycles.

    Returns the cleaned sinogram and the number of values zeroed.
    """
    mask = (sino > 0.0) & (sino < threshold)
    n_zeroed = int(mask.sum())
    sino = sino.copy()
    sino[mask] = 0.0
    return sino, n_zeroed


def check_leaf_cycles_constraint(
    sino: np.ndarray,
    cp_duration_ms: float = DEFAULT_CP_DURATION_MS,
    max_per_second: int = TOMO3_MAX_LEAF_CYCLES_PER_SECOND,
) -> bool:
    """Log leaf-cycle stats and warn if the machine constraint is violated.

    Returns True if the sinogram is within limits, False otherwise.
    """
    cycles_per_cp = count_leaf_cycles(sino)
    cp_per_second = 1000.0 / cp_duration_ms
    cycles_per_second = cycles_per_cp * cp_per_second

    worst_cp = int(np.argmax(cycles_per_second))
    worst_val = float(cycles_per_second[worst_cp])
    mean_val = float(cycles_per_second.mean())

    logging.info(
        "Leaf cycles/s — mean: %.1f  max: %.1f (CP %d)  limit: %d",
        mean_val,
        worst_val,
        worst_cp,
        max_per_second,
    )

    if worst_val > max_per_second:
        logging.warning(
            "CONSTRAINT VIOLATED: CP %d has %.1f leaf cycles/s (limit %d). "
            "Consider increasing --leaf-noise-threshold.",
            worst_cp,
            worst_val,
            max_per_second,
        )
        return False

    logging.info("Leaf-cycle constraint OK (max %.1f <= %d /s).", worst_val, max_per_second)
    return True


def row_to_tomo_bytes(row: np.ndarray) -> bytes:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.size != 64:
        raise ValueError(f"Expected 64 leaf values, got {row.size}.")

    parts = []
    for value in row:
        value = float(value)
        parts.append("0" if abs(value) < 1e-8 else f"{value:.7g}")

    raw = "\\".join(parts).encode("ascii")
    if len(raw) % 2 != 0:
        raw += b" "
    return raw


def resolve_tomo_dir(patient_dir: Path) -> Path:
    """
    Accept either:
        .../<PATIENT_ID>/RC_Publi_Tomo_Halcyon_DIBH/Tomo_FB_copy
        .../<PATIENT_ID>/RC_Publi_Tomo_Halcyon_DIBH
        .../<PATIENT_ID>
    """
    path = patient_dir.expanduser().resolve()

    if path.name == "Tomo_FB_copy" and path.exists():
        return path

    direct = path / "Tomo_FB_copy"
    if direct.exists():
        return direct.resolve()

    matches = sorted(path.glob("**/Tomo_FB_copy"))
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        "Could not locate Tomo_FB_copy. "
        f"Received: {path}. Expected nested path like "
        "/mnt/data/shared/tomo_data/<PATIENT_ID>/RC_Publi_Tomo_Halcyon_DIBH/Tomo_FB_copy"
    )


def infer_dataset_root_and_patient_id(tomo_dir: Path, dataset_root: Path | None, patient_id: str | None) -> tuple[Path, str]:
    """
    For:
        /mnt/data/shared/tomo_data/297768/RC_Publi_Tomo_Halcyon_DIBH/Tomo_FB_copy

    returns:
        root = /mnt/data/shared/tomo_data
        patient_id = 297768
    """
    if dataset_root is not None and patient_id is not None:
        return dataset_root.expanduser().resolve(), str(patient_id)

    numeric_ancestor = None
    for ancestor in tomo_dir.parents:
        if re.fullmatch(r"\d+", ancestor.name):
            numeric_ancestor = ancestor
            break

    if patient_id is None:
        if numeric_ancestor is None:
            raise ValueError("Could not infer patient_id. Use --patient-id.")
        patient_id = numeric_ancestor.name

    if dataset_root is None:
        if numeric_ancestor is None:
            raise ValueError("Could not infer dataset root. Use --dataset-root.")
        dataset_root = numeric_ancestor.parent

    return dataset_root.expanduser().resolve(), str(patient_id)


def available_pareto_indices(tomo_dir: Path) -> list[int]:
    indices = []
    for folder in sorted(tomo_dir.glob("pareto_*")):
        if not folder.is_dir():
            continue
        suffix = folder.name.replace("pareto_", "")
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def find_reference_plan(tomo_dir: Path, pareto_index: int) -> Path:
    pareto_dir = tomo_dir / f"pareto_{pareto_index}"
    if not pareto_dir.exists():
        raise FileNotFoundError(f"Pareto folder not found: {pareto_dir}")

    plan_files = sorted(pareto_dir.glob("RP*.dcm"))
    if not plan_files:
        raise FileNotFoundError(f"No RP*.dcm found in {pareto_dir}")

    return plan_files[0]


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("generator_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state = checkpoint

    state = {str(k).removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    logging.info("Loaded checkpoint: %s", checkpoint_path)


def align_model_output(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 5 and output.shape[1] == 1 and output.shape[-1] == 1:
        return output[:, 0, :, :, 0]
    if output.ndim == 4 and output.shape[1] == 1:
        return output[:, 0, :, :]
    if output.ndim == 3:
        return output
    raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}")


def apply_activation(output: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "sigmoid":
        return torch.sigmoid(output)
    if activation == "clamp":
        return torch.clamp(output, 0.0, 1.0)
    if activation == "none":
        return output
    raise ValueError(f"Unknown output activation: {activation}")


def build_dataset(dataset_root: Path, args: argparse.Namespace) -> RTDataset:
    return RTDataset(
        root_dir=str(dataset_root),
        augmentation=None,
        max_dose=args.max_dose,
        reduction_ratio=args.reduction_ratio,
        use_cache=not args.no_cache,
        cache_dir=str(args.cache_dir.expanduser().resolve()),
    )


def find_dataset_sample(dataset: RTDataset, patient_id: str, pareto_index: int) -> dict[str, Any]:
    for idx, info in enumerate(dataset.samples):
        if str(info.get("patient_id")) == str(patient_id) and int(info.get("pareto_index")) == int(pareto_index):
            sample = dataset[idx]
            logging.info(
                "Loaded sample patient=%s pareto=%s input=%s",
                patient_id,
                pareto_index,
                tuple(sample["input"].shape),
            )
            return sample

    raise ValueError(f"Could not find patient={patient_id}, pareto={pareto_index} in RTDataset.")


def create_model(args: argparse.Namespace, device: torch.device) -> DosePredictionAttentionInAgg:
    model = DosePredictionAttentionInAgg(
        base_filters=args.base_filters,
        in_channel=2,
        attention_kernel_size=args.attention_kernel_size,
        detector_width=args.detector_width,
    ).to(device)

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for inference mode.")

    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    return model


def run_model_inference(
    model: torch.nn.Module,
    dataset: RTDataset,
    patient_id: str,
    pareto_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    sample = find_dataset_sample(dataset, patient_id, pareto_index)
    inp = sample["input"].unsqueeze(0).to(device)

    with torch.no_grad():
        raw = align_model_output(model(inp))
        pred = apply_activation(raw, args.output_activation)

    sino = pred[0].detach().float().cpu().numpy()

    if sino.ndim != 2 or sino.shape[1] != 64:
        raise ValueError(f"Expected predicted sinogram [N_CP,64], got {sino.shape}")

    if args.flip_cp_axis:
        sino = sino[::-1].copy()
        logging.info("Applied CP-axis flip.")

    sino = np.clip(sino, 0.0, 1.0).astype(np.float32)
    logging.info(
        "Predicted pareto=%s shape=%s min=%.6f max=%.6f mean=%.6f",
        pareto_index,
        sino.shape,
        float(sino.min()),
        float(sino.max()),
        float(sino.mean()),
    )
    return sino


def load_sino_npy(path: Path, flip_cp_axis: bool) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Predicted sinogram .npy not found: {path}")

    sino = np.squeeze(np.load(str(path))).astype(np.float32)

    if sino.ndim != 2 or sino.shape[1] != 64:
        raise ValueError(f"Expected .npy sinogram [N_CP,64], got {sino.shape}")

    if flip_cp_axis:
        sino = sino[::-1].copy()
        logging.info("Applied CP-axis flip to .npy.")

    return np.clip(sino, 0.0, 1.0).astype(np.float32)


def inject_sinogram(reference_plan: Path, sino: np.ndarray, output: Path, plan_suffix: str) -> pydicom.Dataset:
    ds = copy.deepcopy(pydicom.dcmread(str(reference_plan)))
    beam = ds[(0x300A, 0x00B0)][0]
    cps = beam[(0x300A, 0x0111)].value
    sino_cps = [cp for cp in cps if SINO_TAG in cp]

    if sino.ndim != 2 or sino.shape[1] != 64:
        raise ValueError(f"Expected sino [N_CP,64], got {sino.shape}")

    if sino.shape[0] != len(sino_cps):
        logging.warning(
            "Prediction CP count %d != RTPLAN sinogram CP count %d. "
            "Using truncate/repeat-last-row alignment.",
            sino.shape[0],
            len(sino_cps),
        )
        if sino.shape[0] > len(sino_cps):
            sino = sino[: len(sino_cps)]
        else:
            pad = np.tile(sino[-1:], (len(sino_cps) - sino.shape[0], 1))
            sino = np.vstack([sino, pad])

    for idx, cp in enumerate(sino_cps):
        cp[SINO_TAG].value = row_to_tomo_bytes(sino[idx])

    now = datetime.now()
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S")
    ds.SeriesDate = now.strftime("%Y%m%d")
    ds.SeriesTime = now.strftime("%H%M%S")

    old_label = str(getattr(ds, "RTPlanLabel", "PLAN"))
    old_name = str(getattr(ds, "RTPlanName", "PLAN"))
    old_description = str(getattr(ds, "PlanDescription", ""))

    ds.RTPlanLabel = clean_dicom_text(f"{old_label}_{plan_suffix}", 16)
    ds.RTPlanName = clean_dicom_text(f"{old_name} {plan_suffix}", 64)
    ds.PlanDescription = clean_dicom_text(
        f"{old_description} [AI predicted LOT sinogram: {plan_suffix}]",
        1024,
    )

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output), write_like_original=False)

    logging.info("Saved RTPLAN: %s", output)
    logging.info("Injected rows: %d", len(sino_cps))
    return ds


def clean_dicom_text(value: str, max_len: int) -> str:
    return value.replace("\n", " ").replace("\r", " ").strip()[:max_len]


def save_preview(sino: np.ndarray, output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = output_path.with_suffix(".png")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(sino, cmap="hot", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("Injected AI-predicted LOT sinogram")
    ax.set_xlabel("Leaf")
    ax.set_ylabel("Control point")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(str(preview), dpi=140, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved preview: %s", preview)


def make_output_path(output_arg: Path, patient_id: str, pareto_index: int, all_paretos: bool) -> Path:
    output_arg = output_arg.expanduser()

    if all_paretos:
        return output_arg / f"{patient_id}_pareto_{pareto_index}_AI_SINO.dcm"

    if output_arg.suffix.lower() == ".dcm":
        return output_arg

    return output_arg / f"{patient_id}_pareto_{pareto_index}_AI_SINO.dcm"


def save_optional_artifacts(sino: np.ndarray, output_path: Path, args: argparse.Namespace) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_npy:
        npy_path = output_path.with_suffix(".npy")
        np.save(str(npy_path), sino)
        logging.info("Saved .npy: %s", npy_path)

    if args.save_preview:
        save_preview(sino, output_path)


def process_one_pareto(
    pareto_index: int,
    tomo_dir: Path,
    patient_id: str,
    dataset: RTDataset | None,
    model: torch.nn.Module | None,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    reference_plan = find_reference_plan(tomo_dir, pareto_index)
    output_path = make_output_path(args.output, patient_id, pareto_index, args.all_paretos)

    if args.predicted_sino_npy is not None:
        if args.all_paretos:
            raise ValueError("--predicted-sino-npy supports single Pareto only. Use model inference for --all-paretos.")
        sino = load_sino_npy(args.predicted_sino_npy, args.flip_cp_axis)
    else:
        if dataset is None or model is None:
            raise RuntimeError("Internal error: dataset/model missing for inference mode.")
        sino = run_model_inference(model, dataset, patient_id, pareto_index, args, device)

    if args.min_leaf_open_ms > 0.0:
        threshold = args.min_leaf_open_ms / args.cp_duration_ms
        sino, n_zeroed = apply_leaf_noise_threshold(sino, threshold)
        if n_zeroed:
            logging.info(
                "Min leaf open time %.0f ms → threshold %.4f: zeroed %d leaf values.",
                args.min_leaf_open_ms,
                threshold,
                n_zeroed,
            )

    check_leaf_cycles_constraint(sino, cp_duration_ms=args.cp_duration_ms)

    if np.any(sino[-1] > 0.0):
        logging.info("Forcing last CP row to all-zeros (Tomo beam end-segment rule).")
        sino = sino.copy()
        sino[-1] = 0.0

    save_optional_artifacts(sino, output_path, args)
    inject_sinogram(reference_plan, sino, output_path, plan_suffix=f"AI_P{pareto_index}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject accepted-baseline AI LOT sinograms into Tomo/Radixact RTPLAN files."
    )

    parser.add_argument(
        "--patient-dir",
        type=Path,
        required=True,
        help="Can be patient root, RC folder, or exact Tomo_FB_copy folder.",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--patient-id", type=str, default=None)

    parser.add_argument("--pareto-index", type=int, default=0)
    parser.add_argument("--all-paretos", action="store_true")
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--predicted-sino-npy", type=Path, default=None)

    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--attention-kernel-size", type=int, default=15)
    parser.add_argument("--detector-width", type=int, default=64)
    parser.add_argument("--reduction-ratio", type=int, default=8)
    parser.add_argument("--max-dose", type=float, default=70.0)

    parser.add_argument("--cache-dir", type=Path, default=Path("/mnt/data/shared/tomo_data/cache_sino"))
    parser.add_argument("--no-cache", action="store_true")

    parser.add_argument("--output-activation", choices=("sigmoid", "clamp", "none"), default="sigmoid")
    parser.add_argument("--flip-cp-axis", action="store_true")

    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument(
        "--min-leaf-open-ms",
        type=float,
        default=TOMO3_MIN_LEAF_OPEN_MS,
        help=(
            "Minimum leaf open time in ms (machine constraint). "
            "Leaf values below min_leaf_open_ms/cp_duration_ms are zeroed. "
            "Set to 0 to disable. Default: 20 ms (Tomo3)."
        ),
    )
    parser.add_argument(
        "--cp-duration-ms",
        type=float,
        default=DEFAULT_CP_DURATION_MS,
        help="Duration of one control point in ms, used to compute leaf cycles/s. Default: 300 ms (Tomo3).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint is None and args.predicted_sino_npy is None:
        raise ValueError("Provide either --checkpoint or --predicted-sino-npy.")

    tomo_dir = resolve_tomo_dir(args.patient_dir)
    dataset_root, patient_id = infer_dataset_root_and_patient_id(tomo_dir, args.dataset_root, args.patient_id)

    logging.info("Resolved Tomo dir: %s", tomo_dir)
    logging.info("Resolved dataset root: %s", dataset_root)
    logging.info("Resolved patient ID: %s", patient_id)

    if args.all_paretos:
        pareto_indices = available_pareto_indices(tomo_dir)
        if not pareto_indices:
            raise FileNotFoundError(f"No pareto_* folders found in {tomo_dir}")
    else:
        pareto_indices = [args.pareto_index]

    logging.info("Paretos to process: %s", pareto_indices)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = None
    model = None

    if args.predicted_sino_npy is None:
        dataset = build_dataset(dataset_root, args)
        model = create_model(args, device)

    outputs = []
    for pareto_index in pareto_indices:
        logging.info("========== Processing pareto %s ==========", pareto_index)
        out = process_one_pareto(
            pareto_index=pareto_index,
            tomo_dir=tomo_dir,
            patient_id=patient_id,
            dataset=dataset,
            model=model,
            args=args,
            device=device,
        )
        outputs.append(str(out))

    logging.info("========== Done ==========")
    for out in outputs:
        logging.info("Generated: %s", out)


if __name__ == "__main__":
    main()
