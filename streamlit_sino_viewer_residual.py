#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_sino_viewer_residual.py

Streamlit viewer for:
1. Standard supervised/sanity sinogram outputs.
2. Residual GAN refinement outputs.
3. Checkpoint inference for a chosen patient + pareto.

Run:
    streamlit run streamlit_sino_viewer_residual.py

Expected project files for residual checkpoint inference:
    models/unet_attention_in_agg.py
    models/sino_residual_refiner.py
    utils/patient.py
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import torch
except Exception:
    torch = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None


APP_TITLE = "Sinogram Viewer: Standard + Residual Refinement"


def add_project_root(project_root: str | Path) -> None:
    root = str(Path(project_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not read JSON `{path}`: {exc}")
        return None


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read CSV `{path}`: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_npy(path: str) -> np.ndarray:
    return np.load(path)


def squeeze2d(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze, got {arr.shape}")
    return arr.astype(np.float32)


def final_array_dir(output_dir: Path) -> Path:
    candidates = [
        output_dir / "visualizations" / "final_all_paretos",
        output_dir / "final_all_paretos",
        output_dir / "visualizations",
        output_dir,
    ]
    for candidate in candidates:
        if candidate.exists() and list(candidate.glob("pareto_*_target.npy")):
            return candidate
    return output_dir / "visualizations" / "final_all_paretos"


def pareto_id(path: Path) -> str:
    match = re.search(r"pareto_(.+?)_(?:pred|target|baseline|refined|delta)", path.name)
    return match.group(1) if match else path.stem


def discover_standard_arrays(arr_dir: Path) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}

    for p in arr_dir.glob("pareto_*_target.npy"):
        out.setdefault(pareto_id(p), {})["target"] = p

    patterns = {
        "pred": ["pareto_*_pred_prob.npy", "pareto_*_pred.npy", "pareto_*_prediction.npy"],
        "baseline": ["pareto_*_baseline.npy"],
        "refined": ["pareto_*_refined.npy"],
        "delta": ["pareto_*_delta.npy"],
    }

    for key, pats in patterns.items():
        for pat in pats:
            for p in arr_dir.glob(pat):
                out.setdefault(pareto_id(p), {})[key] = p

    return {k: v for k, v in out.items() if "target" in v}


def discover_visual_pngs(output_dir: Path) -> list[Path]:
    vis_dir = output_dir / "visualizations"
    if not vis_dir.exists():
        return []
    return sorted(vis_dir.glob("*.png"))


def discover_checkpoints(output_dir: Path) -> list[Path]:
    names = [
        "best_residual_gan_checkpoint.pt",
        "best_gan_checkpoint.pt",
        "best_checkpoint.pt",
        "final_checkpoint.pt",
    ]
    ckpts = [output_dir / name for name in names if (output_dir / name).exists()]
    ckpts.extend(sorted(output_dir.glob("checkpoint_epoch_*.pt")))
    return ckpts


def metrics_dict(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = squeeze2d(pred)
    gt = squeeze2d(gt)
    diff = np.abs(pred - gt)
    open_mask = gt > 1e-6
    closed_mask = ~open_mask

    return {
        "MAE": float(diff.mean()),
        "Open L1": float(diff[open_mask].mean()) if open_mask.any() else float("nan"),
        "Closed pred": float(np.abs(pred[closed_mask]).mean()) if closed_mask.any() else float("nan"),
        "Max abs": float(diff.max()),
        "Open frac": float(open_mask.mean()),
        "Pred mean": float(pred.mean()),
        "GT mean": float(gt.mean()),
    }


def show_metric_cards(metrics: dict[str, float], prefix: str = "") -> None:
    cols = st.columns(min(7, len(metrics)))
    for col, (key, value) in zip(cols, metrics.items()):
        col.metric(f"{prefix}{key}", f"{value:.5f}" if np.isfinite(value) else "NA")


def fig_standard_triplet(pred: np.ndarray, gt: np.ndarray, title: str):
    pred = squeeze2d(pred)
    gt = squeeze2d(gt)
    diff = np.abs(pred - gt)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for axis, img, name, cmap, vmax in [
        (ax[0], pred, "Prediction", "hot", 1.0),
        (ax[1], gt, "GT", "hot", 1.0),
        (ax[2], diff, "|Pred-GT|", "RdYlGn_r", 0.5),
    ]:
        im = axis.imshow(img, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        axis.set_title(name)
        axis.set_xlabel("Leaf")
        axis.set_ylabel("Control point")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    return fig


def fig_residual_compare(
    baseline: np.ndarray,
    refined: np.ndarray,
    gt: np.ndarray,
    delta: np.ndarray | None,
    title: str,
):
    baseline = squeeze2d(baseline)
    refined = squeeze2d(refined)
    gt = squeeze2d(gt)
    base_diff = np.abs(baseline - gt)
    refined_diff = np.abs(refined - gt)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    plots = [
        (ax[0, 0], baseline, "Frozen baseline", "hot", 0, 1),
        (ax[0, 1], refined, "Residual refined", "hot", 0, 1),
        (ax[0, 2], gt, "GT", "hot", 0, 1),
        (ax[1, 0], base_diff, "|Baseline-GT|", "RdYlGn_r", 0, 0.5),
        (ax[1, 1], refined_diff, "|Refined-GT|", "RdYlGn_r", 0, 0.5),
    ]

    for axis, img, name, cmap, vmin, vmax in plots:
        im = axis.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(name)
        axis.set_xlabel("Leaf")
        axis.set_ylabel("Control point")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    if delta is not None:
        delta = squeeze2d(delta)
        lim = max(float(np.max(np.abs(delta))), 1e-6)
        im = ax[1, 2].imshow(delta, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
        ax[1, 2].set_title("Bounded delta")
        ax[1, 2].set_xlabel("Leaf")
        ax[1, 2].set_ylabel("Control point")
        fig.colorbar(im, ax=ax[1, 2], fraction=0.046, pad=0.04)
    else:
        ax[1, 2].axis("off")

    fig.suptitle(title)
    return fig


def fig_hist(pred: np.ndarray, gt: np.ndarray, bins: int):
    pred = squeeze2d(pred)
    gt = squeeze2d(gt)
    diff = np.abs(pred - gt)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.hist(gt.ravel(), bins=bins, range=(0, 1), alpha=0.55, label="GT")
    ax.hist(pred.ravel(), bins=bins, range=(0, 1), alpha=0.55, label="Prediction")
    ax.hist(diff.ravel(), bins=bins, range=(0, 1), alpha=0.45, label="Abs diff")
    ax.set_title("Histogram distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend()
    return fig


def show_surface(arr: np.ndarray, title: str, cp_stride: int, leaf_stride: int):
    arr = squeeze2d(arr)[::cp_stride, ::leaf_stride]

    if go is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(arr, cmap="hot", vmin=0, vmax=1, aspect="auto")
        ax.set_title(title)
        st.pyplot(fig)
        return

    y = np.arange(arr.shape[0]) * cp_stride
    x = np.arange(arr.shape[1]) * leaf_stride

    fig = go.Figure(data=[go.Surface(z=arr, x=x, y=y, colorscale="Hot", cmin=0, cmax=1)])
    fig.update_layout(
        title=title,
        height=700,
        scene=dict(
            xaxis_title="Leaf",
            yaxis_title="Control point",
            zaxis_title="Value",
            zaxis=dict(range=[0, 1]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_loss_history(output_dir: Path) -> None:
    st.subheader("Train / validation curves")

    df = read_csv(output_dir / "loss_history.csv")
    if df is None or df.empty:
        st.info("No loss_history.csv found.")
        return

    st.dataframe(df.tail(30), use_container_width=True)

    xcol = "epoch" if "epoch" in df.columns else df.columns[0]
    numeric_cols = [c for c in df.columns if c != xcol and pd.api.types.is_numeric_dtype(df[c])]

    preferred = [
        "val_open_l1",
        "val_mae",
        "val_closed_abs_pred",
        "val_loss",
        "train_open_l1",
        "train_mae",
        "train_closed_abs_pred",
        "train_loss",
        "train_mean_abs_delta",
        "val_mean_abs_delta",
    ]

    defaults = [c for c in preferred if c in numeric_cols][:5]
    selected = st.multiselect("Metrics", numeric_cols, default=defaults if defaults else numeric_cols[:4])

    if not selected:
        return

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for col in selected:
        ax.plot(df[xcol], df[col], marker="o", markersize=2, linewidth=1.5, label=col)

    ax.set_xlabel(xcol)
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    st.pyplot(fig)


def show_tables(output_dir: Path) -> None:
    st.subheader("Tables")

    for name in [
        "baseline_val_per_sample.csv",
        "best_val_per_sample.csv",
        "val_per_sample_latest.csv",
        "test_per_sample.csv",
        "per_sample_metrics_final.csv",
        "per_sample_metrics_latest.csv",
    ]:
        df = read_csv(output_dir / name)
        if df is not None and not df.empty:
            with st.expander(name, expanded=name in {"best_val_per_sample.csv", "test_per_sample.csv"}):
                st.dataframe(df, use_container_width=True)

    for name in ["test_metrics.json", "split_manifest.json", "config.json"]:
        obj = read_json(output_dir / name)
        if obj is not None:
            with st.expander(name, expanded=False):
                st.json(obj)


def standard_array_view(output_dir: Path, patient_id: str) -> None:
    arr_dir = final_array_dir(output_dir)
    arrays = discover_standard_arrays(arr_dir)

    if not arrays:
        st.info("No standard final arrays found. Use Residual PNGs or Checkpoint inference mode.")
        return

    keys = sorted(arrays.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    selected = st.selectbox("Pareto", keys)

    paths = arrays[selected]
    gt = squeeze2d(load_npy(str(paths["target"])))

    pred_path = paths.get("refined") or paths.get("pred") or paths.get("baseline")
    if pred_path is None:
        st.warning("No prediction array found.")
        return

    pred = squeeze2d(load_npy(str(pred_path)))
    baseline = squeeze2d(load_npy(str(paths["baseline"]))) if "baseline" in paths else None
    delta = squeeze2d(load_npy(str(paths["delta"]))) if "delta" in paths else None

    st.caption(f"Array dir: `{arr_dir}`")
    st.caption(f"Prediction: `{pred_path}`")
    st.caption(f"GT: `{paths['target']}`")

    if baseline is not None:
        st.markdown("#### Residual arrays")
        c1, c2 = st.columns(2)
        with c1:
            show_metric_cards(metrics_dict(baseline, gt), "Baseline ")
        with c2:
            show_metric_cards(metrics_dict(pred, gt), "Refined ")

        st.pyplot(fig_residual_compare(baseline, pred, gt, delta, f"Patient {patient_id} | Pareto {selected}"))
    else:
        show_metric_cards(metrics_dict(pred, gt))
        st.pyplot(fig_standard_triplet(pred, gt, f"Patient {patient_id} | Pareto {selected}"))

    with st.expander("Histograms / 3D", expanded=False):
        bins = st.slider("Histogram bins", 20, 200, 80)
        st.pyplot(fig_hist(pred, gt, bins))
        cp_stride = st.slider("CP stride", 1, 20, 4)
        leaf_stride = st.slider("Leaf stride", 1, 8, 1)
        show_surface(pred, "Prediction surface", cp_stride, leaf_stride)


def residual_png_view(output_dir: Path) -> None:
    pngs = discover_visual_pngs(output_dir)
    if not pngs:
        st.info("No PNG visualizations found.")
        return

    st.subheader("Saved PNG visualizations")
    labels = [p.name for p in pngs]
    idx = st.selectbox("Visualization", range(len(pngs)), format_func=lambda i: labels[i])
    st.image(str(pngs[idx]), caption=str(pngs[idx]), use_container_width=True)


def load_checkpoint(path: Path, device: torch.device):
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    return torch.load(path, map_location=device, weights_only=False)


def get_ckpt_epoch(path: Path) -> str:
    match = re.search(r"epoch_(\d+)", path.name)
    if match:
        return match.group(1)
    if "best" in path.name:
        return "best"
    return path.stem


@st.cache_resource(show_spinner=False)
def cached_dataset(project_root: str, data_path: str, cache_dir: str, max_dose: float, reduction_ratio: int, use_cache: bool):
    add_project_root(project_root)
    mod = importlib.import_module("utils.patient")
    return mod.RTDataset(
        root_dir=data_path,
        augmentation=None,
        max_dose=max_dose,
        reduction_ratio=reduction_ratio,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )


def find_sample_idx(dataset, patient_id: str, pareto_index: str) -> int | None:
    for idx, sample in enumerate(dataset.samples):
        if str(sample.get("patient_id")) == str(patient_id) and str(sample.get("pareto_index")) == str(pareto_index):
            return idx
    return None


def align_target_torch(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 5 and target.shape[1] == 1 and target.shape[-1] == 1:
        return target[:, 0, :, :, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        return target[:, 0, :, :]
    if target.ndim == 3:
        return target
    raise ValueError(f"Unexpected target shape {tuple(target.shape)}")


def align_logits_torch(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 5 and output.shape[1] == 1 and output.shape[-1] == 1:
        return output[:, 0, :, :, 0]
    if output.ndim == 4 and output.shape[1] == 1:
        return output[:, 0, :, :]
    if output.ndim == 3:
        return output
    raise ValueError(f"Unexpected output shape {tuple(output.shape)}")


def make_condition_maps_torch(inputs: torch.Tensor, reduction: str) -> torch.Tensor:
    x = inputs[:, :2]
    if reduction == "mean":
        return x.mean(dim=-1)
    if reduction == "max":
        return x.amax(dim=-1)
    if reduction == "meanmax":
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
    raise ValueError(reduction)


def to_2d_channel_torch(sino: torch.Tensor) -> torch.Tensor:
    if sino.ndim == 3:
        return sino.unsqueeze(1)
    if sino.ndim == 4:
        return sino
    raise ValueError(f"Expected [B,N,64] or [B,1,N,64], got {tuple(sino.shape)}")


def load_state(model, state: dict[str, Any], candidate_keys: list[str], strict: bool = True) -> None:
    selected = None
    for key in candidate_keys:
        if key in state:
            selected = state[key]
            break
    if selected is None:
        selected = state

    cleaned = {str(k).removeprefix("module."): v for k, v in selected.items()}
    model.load_state_dict(cleaned, strict=strict)


@st.cache_resource(show_spinner=True)
def cached_models(
    project_root: str,
    ckpt_path: str,
    gen_ckpt_path: str,
    base_filters: int,
    attention_kernel_size: int,
    detector_width: int,
    condition_reduction: str,
    refiner_base_channels: int,
    delta_scale: float,
    device_text: str,
):
    if torch is None:
        raise RuntimeError("PyTorch is not available.")

    add_project_root(project_root)

    gen_mod = importlib.import_module("models.unet_attention_in_agg")
    ref_mod = importlib.import_module("models.sino_residual_refiner")

    device = torch.device(device_text)

    Generator = getattr(gen_mod, "DosePredictionAttentionInAgg")
    Refiner = getattr(ref_mod, "SinoResidualRefiner2D")

    generator = Generator(
        base_filters=int(base_filters),
        in_channel=2,
        attention_kernel_size=int(attention_kernel_size),
        detector_width=int(detector_width),
    ).to(device)

    ckpt = load_checkpoint(Path(ckpt_path), device)

    if "generator_state_dict" in ckpt:
        load_state(generator, ckpt, ["generator_state_dict"], strict=True)
    else:
        gen_ckpt = load_checkpoint(Path(gen_ckpt_path), device)
        load_state(generator, gen_ckpt, ["model_state_dict", "generator_state_dict", "state_dict"], strict=True)

    for p in generator.parameters():
        p.requires_grad_(False)
    generator.eval()

    condition_channels = 2 if condition_reduction in {"mean", "max"} else 4

    refiner = Refiner(
        condition_channels=condition_channels,
        base_channels=int(refiner_base_channels),
        delta_scale=float(delta_scale),
    ).to(device)

    if "refiner_state_dict" in ckpt:
        load_state(refiner, ckpt, ["refiner_state_dict"], strict=True)
        refiner.eval()
    else:
        refiner = None

    return generator, refiner


def checkpoint_inference_view(output_dir: Path, config: dict[str, Any]) -> None:
    st.subheader("Checkpoint inference: choose patient + pareto")

    if torch is None:
        st.error("PyTorch is not available.")
        return

    ckpts = discover_checkpoints(output_dir)
    if not ckpts:
        st.info("No checkpoints found in this output directory.")
        return

    project_root = st.text_input("Project root", value=str(Path.cwd()))
    data_path = st.text_input("DATA_PATH", value=str(config.get("data_path", "/mnt/data/shared/tomo_data")))
    cache_dir = st.text_input("CACHE_DIR", value=str(config.get("cache_dir", "/mnt/data/shared/tomo_data/cache_sino")))

    c1, c2, c3 = st.columns(3)
    with c1:
        max_dose = st.number_input("MAX_DOSE", value=float(config.get("max_dose", 70.0)))
        reduction_ratio = st.number_input("REDUCTION_RATIO", min_value=1, value=int(config.get("reduction_ratio", 8)))
        use_cache = st.checkbox("Use cache", value=bool(config.get("use_cache", True)))
    with c2:
        base_filters = st.number_input("base_filters", min_value=1, value=int(config.get("base_filters", 16)))
        attention_kernel_size = st.number_input("attention_kernel_size", min_value=1, value=int(config.get("attention_kernel_size", 15)))
        detector_width = st.number_input("detector_width", min_value=1, value=int(config.get("detector_width", 64)))
    with c3:
        condition_reduction = st.selectbox(
            "condition_reduction",
            ["mean", "max", "meanmax"],
            index=["mean", "max", "meanmax"].index(str(config.get("condition_reduction", "mean"))),
        )
        refiner_base_channels = st.number_input(
            "refiner_base_channels",
            min_value=1,
            value=int(config.get("refiner_base_channels", 32)),
        )
        delta_scale = st.number_input("delta_scale", value=float(config.get("delta_scale", 0.10)))

    ckpt_idx = st.selectbox(
        "Checkpoint",
        range(len(ckpts)),
        format_func=lambda i: f"{get_ckpt_epoch(ckpts[i])} | {ckpts[i].name}",
    )
    ckpt_path = ckpts[ckpt_idx]

    gen_ckpt_path = st.text_input(
        "Generator checkpoint fallback",
        value=str(config.get("generator_checkpoint", "")),
        help="Only used if the selected checkpoint does not contain generator_state_dict.",
    )

    device_text = "cuda" if torch.cuda.is_available() else "cpu"
    device_text = st.selectbox("Device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"], index=0)

    if "dataset_loaded" not in st.session_state:
        st.session_state["dataset_loaded"] = False

    if st.button("Load dataset", type="primary"):
        st.session_state["dataset_loaded"] = True

    if not st.session_state["dataset_loaded"]:
        st.info("Load dataset to choose patient and pareto.")
        return

    try:
        dataset = cached_dataset(project_root, data_path, cache_dir, max_dose, int(reduction_ratio), use_cache)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        return

    patients = sorted({str(s.get("patient_id")) for s in dataset.samples})
    patient_id = st.selectbox("Patient", patients)

    paretos = sorted(
        {str(s.get("pareto_index")) for s in dataset.samples if str(s.get("patient_id")) == str(patient_id)},
        key=lambda x: int(x) if x.isdigit() else x,
    )
    pareto_index = st.selectbox("Pareto", paretos)

    sample_idx = find_sample_idx(dataset, patient_id, pareto_index)
    if sample_idx is None:
        st.error("Sample not found.")
        return

    if st.button("Run inference", type="primary"):
        try:
            generator, refiner = cached_models(
                project_root=project_root,
                ckpt_path=str(ckpt_path),
                gen_ckpt_path=gen_ckpt_path,
                base_filters=int(base_filters),
                attention_kernel_size=int(attention_kernel_size),
                detector_width=int(detector_width),
                condition_reduction=condition_reduction,
                refiner_base_channels=int(refiner_base_channels),
                delta_scale=float(delta_scale),
                device_text=device_text,
            )

            device = torch.device(device_text)
            sample = dataset[sample_idx]

            x = sample["input"].unsqueeze(0).to(device).float()
            y = sample["target"].unsqueeze(0).to(device).float()

            with torch.no_grad():
                logits = align_logits_torch(generator(x))
                baseline = torch.sigmoid(logits)

                gt = align_target_torch(y)

                if refiner is not None:
                    condition = make_condition_maps_torch(x, condition_reduction)
                    ref_out = refiner(condition, to_2d_channel_torch(baseline))
                    refined = ref_out["refined"][:, 0]
                    delta = ref_out["delta"][:, 0]
                else:
                    refined = baseline
                    delta = None

            st.session_state["inference_result"] = {
                "baseline": baseline[0].detach().cpu().numpy(),
                "refined": refined[0].detach().cpu().numpy(),
                "gt": gt[0].detach().cpu().numpy(),
                "delta": delta[0].detach().cpu().numpy() if delta is not None else None,
                "patient_id": patient_id,
                "pareto_index": pareto_index,
                "checkpoint": str(ckpt_path),
            }

        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return

    result = st.session_state.get("inference_result")
    if not result:
        return

    baseline = result["baseline"]
    refined = result["refined"]
    gt = result["gt"]
    delta = result["delta"]

    st.caption(f"Checkpoint: `{result['checkpoint']}`")

    c1, c2 = st.columns(2)
    with c1:
        show_metric_cards(metrics_dict(baseline, gt), "Baseline ")
    with c2:
        show_metric_cards(metrics_dict(refined, gt), "Refined ")

    st.pyplot(
        fig_residual_compare(
            baseline,
            refined,
            gt,
            delta,
            f"Patient {result['patient_id']} | Pareto {result['pareto_index']}",
        )
    )

    with st.expander("Histograms / 3D", expanded=False):
        bins = st.slider("Inference histogram bins", 20, 200, 80)
        st.pyplot(fig_hist(refined, gt, bins))
        show_surface(refined, "Refined surface", 4, 1)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.sidebar.header("Output folder")
    output_dir = Path(
        st.sidebar.text_input(
            "Folder",
            value="runs/residual_gan_refinement_full/run_YYYYMMDD_HHMMSS_bf16_resgan_delta0.1_adv0.005",
        )
    ).expanduser()

    if not output_dir.exists():
        st.error(f"Output folder does not exist: {output_dir}")
        return

    config_obj = read_json(output_dir / "config.json") or {}
    config = config_obj if isinstance(config_obj, dict) else {}

    with st.sidebar.expander("config.json", expanded=False):
        if config:
            st.json(config)
        else:
            st.write("No config.json found.")

    patient_id = str(config.get("patient_id", "unknown"))

    mode = st.sidebar.radio(
        "Viewer mode",
        [
            "Auto standard arrays",
            "Residual saved PNGs",
            "Checkpoint inference",
            "Train/Val curves",
            "Tables",
        ],
    )

    if mode == "Auto standard arrays":
        standard_array_view(output_dir, patient_id)
    elif mode == "Residual saved PNGs":
        residual_png_view(output_dir)
    elif mode == "Checkpoint inference":
        checkpoint_inference_view(output_dir, config)
    elif mode == "Train/Val curves":
        plot_loss_history(output_dir)
    elif mode == "Tables":
        show_tables(output_dir)


if __name__ == "__main__":
    main()
