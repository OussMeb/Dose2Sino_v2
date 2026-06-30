#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_sanity_viewer.py

Standalone Streamlit viewer for sanity/full training outputs.

Run:
    streamlit run streamlit_sanity_viewer.py

Main expected folder:
    sanity_outputs/.../patient_XXXX/
        config.json
        loss_history.csv
        per_sample_metrics_final.csv
        selected_samples.json
        visualizations/final_all_paretos/
            pareto_0_pred_prob.npy
            pareto_0_target.npy

Clinical DVH note:
    The model predicts sinograms, not dose volumes. True DVH requires pred/GT
    dose volumes + structure mask. This app includes true DVH only when those
    arrays are uploaded/provided. Otherwise it shows sinogram cumulative curves.
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
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    import torch
except Exception:
    torch = None


def add_project_root(project_root: str) -> None:
    root = str(Path(project_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None


def read_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
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
    for c in candidates:
        if c.exists() and list(c.glob("pareto_*_target.npy")):
            return c
    return output_dir / "visualizations" / "final_all_paretos"


def pareto_id(path: Path) -> str:
    m = re.search(r"pareto_(.+?)_(?:pred|target)", path.name)
    return m.group(1) if m else path.stem


def discover_paretos(arr_dir: Path) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for p in arr_dir.glob("pareto_*_target.npy"):
        out.setdefault(pareto_id(p), {})["target"] = p
    for pat in ["pareto_*_pred_prob.npy", "pareto_*_pred.npy", "pareto_*_prediction.npy"]:
        for p in arr_dir.glob(pat):
            out.setdefault(pareto_id(p), {})["pred"] = p
    return {k: v for k, v in out.items() if "target" in v}


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = squeeze2d(pred)
    gt = squeeze2d(gt)
    d = np.abs(pred - gt)
    open_mask = gt > 1e-6
    closed_mask = ~open_mask
    return {
        "MAE": float(d.mean()),
        "Open L1": float(d[open_mask].mean()) if open_mask.any() else float("nan"),
        "Closed pred": float(np.abs(pred[closed_mask]).mean()) if closed_mask.any() else float("nan"),
        "Max abs": float(d.max()),
        "Open frac": float(open_mask.mean()),
        "Pred mean": float(pred.mean()),
        "GT mean": float(gt.mean()),
    }


def show_metric_cards(m: dict[str, float]) -> None:
    cols = st.columns(7)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.5f}")


def fig_sino(pred: np.ndarray, gt: np.ndarray, title: str):
    pred = squeeze2d(pred)
    gt = squeeze2d(gt)
    diff = np.abs(pred - gt)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for a, img, name, cmap, vmax in [
        (ax[0], pred, "Prediction", "hot", 1),
        (ax[1], gt, "GT", "hot", 1),
        (ax[2], diff, "|Pred-GT|", "RdYlGn_r", 0.5),
    ]:
        im = a.imshow(img, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        a.set_title(name)
        a.set_xlabel("Leaf")
        a.set_ylabel("Control point")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    return fig


def fig_hist(pred: np.ndarray, gt: np.ndarray, bins: int):
    pred = squeeze2d(pred)
    gt = squeeze2d(gt)
    diff = np.abs(pred - gt)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.hist(gt.ravel(), bins=bins, range=(0, 1), alpha=0.55, label="GT")
    ax.hist(pred.ravel(), bins=bins, range=(0, 1), alpha=0.55, label="Pred")
    ax.hist(diff.ravel(), bins=bins, range=(0, 1), alpha=0.45, label="Abs diff")
    ax.set_title("Histogram distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend()
    return fig


def show_surface(arr: np.ndarray, title: str, cp_stride: int, leaf_stride: int):
    arr = squeeze2d(arr)[::cp_stride, ::leaf_stride]
    if go is None:
        st.warning("plotly is not installed. Showing 2D fallback.")
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


def cumulative_curve(pred: np.ndarray, gt: np.ndarray):
    pred = squeeze2d(pred).ravel()
    gt = squeeze2d(gt).ravel()
    x = np.linspace(0, 1, 201)
    yp = np.array([(pred >= t).mean() * 100 for t in x])
    yg = np.array([(gt >= t).mean() * 100 for t in x])
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(x, yg, label="GT")
    ax.plot(x, yp, label="Pred")
    ax.set_title("Sinogram cumulative distribution — not clinical DVH")
    ax.set_xlabel("Leaf opening threshold")
    ax.set_ylabel("Pixels ≥ threshold (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def load_uploaded_array(uploaded) -> np.ndarray | None:
    if uploaded is None:
        return None
    try:
        if uploaded.name.endswith(".npy"):
            return np.load(uploaded)
        if uploaded.name.endswith(".npz"):
            data = np.load(uploaded)
            return np.asarray(data[list(data.keys())[0]])
    except Exception as exc:
        st.error(f"Upload load error: {exc}")
    return None


def load_mask_path(path_text: str) -> np.ndarray | None:
    if not path_text:
        return None
    p = Path(path_text).expanduser()
    if not p.exists():
        st.warning(f"Mask path not found: {p}")
        return None
    try:
        if p.suffix == ".npy":
            return np.load(p)
        if p.suffix == ".npz":
            data = np.load(p)
            return np.asarray(data[list(data.keys())[0]])
        if p.name.endswith(".nii") or p.name.endswith(".nii.gz"):
            sitk = importlib.import_module("SimpleITK")
            return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
    except Exception as exc:
        st.warning(f"Mask load failed: {exc}")
    return None


def normalize(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def overlay_mask(img: np.ndarray, mask: np.ndarray | None):
    base = normalize(img)
    rgb = np.stack([base, base, base], axis=-1)
    if mask is None or mask.shape != img.shape:
        return rgb
    m = mask > 0
    rgb[m, 0] = 1.0
    rgb[m, 1] *= 0.45
    rgb[m, 2] *= 0.45
    return rgb


@st.cache_resource(show_spinner=False)
def get_dataset(project_root: str, data_path: str, cache_dir: str, max_dose: float, reduction_ratio: int, use_cache: bool):
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


def find_sample_index(ds, patient_id: str, pareto: str):
    for i, s in enumerate(ds.samples):
        if str(s.get("patient_id")) == str(patient_id) and str(s.get("pareto_index")) == str(pareto):
            return i
    return None


@st.cache_data(show_spinner=True)
def load_rt_sample(project_root: str, data_path: str, cache_dir: str, max_dose: float, reduction_ratio: int, use_cache: bool, patient_id: str, pareto: str):
    if torch is None:
        raise RuntimeError("PyTorch is needed to load RTDataset.")
    ds = get_dataset(project_root, data_path, cache_dir, max_dose, reduction_ratio, use_cache)
    idx = find_sample_index(ds, patient_id, pareto)
    if idx is None:
        raise ValueError(f"Could not find patient={patient_id}, pareto={pareto}")
    sample = ds[idx]
    return {
        "input": sample["input"].detach().cpu().numpy(),
        "target": sample["target"].detach().cpu().numpy(),
    }


def ct_dose_struct_tab(config: dict[str, Any], patient_id: str, pareto: str):
    st.subheader("CT + RTDose + optional structure")
    st.info("Loads CT/dose berlingo from RTDataset/cache. First load can be slow.")

    c1, c2 = st.columns(2)
    with c1:
        project_root = st.text_input("Project root", value=str(Path.cwd()))
        data_path = st.text_input("DATA_PATH", value=str(config.get("data_path", "/mnt/LeGrosDisque/oussama/tomo_data/")))
        cache_dir = st.text_input("CACHE_DIR", value=str(config.get("cache_dir", "/mnt/LeGrosDisque/oussama/tomo_data/cache_sino")))
    with c2:
        max_dose = st.number_input("MAX_DOSE", value=float(config.get("max_dose", 70.0)))
        reduction_ratio = st.number_input("REDUCTION_RATIO", min_value=1, value=int(config.get("reduction_ratio", 8)))
        use_cache = st.checkbox("Use cache", value=bool(config.get("use_cache", True)))

    if st.button("Load CT/dose sample", type="primary"):
        try:
            st.session_state["rt_sample"] = load_rt_sample(
                project_root, data_path, cache_dir, max_dose, int(reduction_ratio), use_cache, patient_id, str(pareto)
            )
        except Exception as exc:
            st.error(str(exc))

    sample = st.session_state.get("rt_sample")
    if not sample:
        return

    inp = np.asarray(sample["input"])
    if inp.shape[0] < 2:
        st.error(f"Expected [2,N,H,W], got {inp.shape}")
        return

    ct = inp[0]
    dose = inp[1] * float(max_dose)
    st.write(f"Input shape: `{inp.shape}`")

    idx = st.slider("Control point/slice", 0, int(ct.shape[0] - 1), int(ct.shape[0] // 2))
    alpha = st.slider("Dose alpha", 0.0, 1.0, 0.35)

    mask_upload = st.file_uploader("Optional structure mask upload (.npy/.npz)", type=["npy", "npz"])
    mask_path = st.text_input("Optional structure mask path (.npy/.npz/.nii/.nii.gz)")
    mask = load_uploaded_array(mask_upload) or load_mask_path(mask_path)

    mask_slice = None
    if mask is not None:
        mask = np.squeeze(mask)
        if mask.shape == ct.shape:
            mask_slice = mask[idx]
        elif mask.shape == ct[idx].shape:
            mask_slice = mask
        else:
            st.warning(f"Mask shape {mask.shape} does not match volume {ct.shape} or slice {ct[idx].shape}")

    fig, ax = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    ax[0].imshow(ct[idx], cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("CT berlingo")
    ax[1].imshow(dose[idx], cmap="hot", vmin=0, vmax=max_dose)
    ax[1].set_title("RTDose berlingo Gy")
    ax[2].imshow(ct[idx], cmap="gray", vmin=0, vmax=1)
    ax[2].imshow(dose[idx], cmap="hot", alpha=alpha, vmin=0, vmax=max_dose)
    ax[2].set_title("CT + dose")
    ax[3].imshow(overlay_mask(ct[idx], mask_slice))
    ax[3].set_title("CT + struct")
    for a in ax:
        a.axis("off")
    st.pyplot(fig)

    mode = st.radio("3D/projection mode", ["MIP", "Mean projection", "Plotly volume"], horizontal=True)
    if mode in ["MIP", "Mean projection"]:
        reducer = np.max if mode == "MIP" else np.mean
        fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        ax2[0].imshow(reducer(ct, axis=0), cmap="gray")
        ax2[0].set_title(f"CT {mode}")
        ax2[1].imshow(reducer(dose, axis=0), cmap="hot")
        ax2[1].set_title(f"Dose {mode}")
        ax2[2].imshow(reducer(ct, axis=0), cmap="gray")
        ax2[2].imshow(reducer(dose, axis=0), cmap="hot", alpha=0.35)
        ax2[2].set_title("Overlay")
        for a in ax2:
            a.axis("off")
        st.pyplot(fig2)
    else:
        if go is None:
            st.warning("plotly not installed.")
        else:
            vol_name = st.selectbox("Volume", ["dose", "ct"])
            stride = st.slider("3D stride", 1, 8, 4)
            vol = dose if vol_name == "dose" else normalize(ct)
            vol = vol[::stride, ::stride, ::stride]
            zz, yy, xx = np.indices(vol.shape)
            fig3 = go.Figure(data=go.Volume(
                x=xx.flatten(), y=yy.flatten(), z=zz.flatten(), value=vol.flatten(),
                opacity=0.12, surface_count=12
            ))
            fig3.update_layout(title=f"3D {vol_name}, stride={stride}", height=700)
            st.plotly_chart(fig3, use_container_width=True)


def compute_dvh(dose: np.ndarray, mask: np.ndarray):
    dose = np.asarray(dose, dtype=np.float32)
    mask = np.asarray(mask) > 0
    if dose.shape != mask.shape:
        raise ValueError(f"shape mismatch dose={dose.shape}, mask={mask.shape}")
    vals = dose[mask]
    if vals.size == 0:
        raise ValueError("empty mask")
    x = np.linspace(0, float(vals.max()), 250)
    y = np.array([(vals >= t).mean() * 100 for t in x])
    return x, y


def dvh_tab(pred: np.ndarray, gt: np.ndarray):
    st.subheader("DVH / distributions")
    st.warning("True DVH requires 3D dose volumes + structure mask. Sinogram-only outputs cannot produce clinical DVH.")
    st.pyplot(cumulative_curve(pred, gt))

    st.markdown("#### Optional true DVH")
    c1, c2, c3 = st.columns(3)
    with c1:
        pred_dose = load_uploaded_array(st.file_uploader("Pred dose .npy/.npz", type=["npy", "npz"], key="pred_dose"))
    with c2:
        gt_dose = load_uploaded_array(st.file_uploader("GT dose .npy/.npz", type=["npy", "npz"], key="gt_dose"))
    with c3:
        mask = load_uploaded_array(st.file_uploader("Structure mask .npy/.npz", type=["npy", "npz"], key="dvh_mask"))

    if pred_dose is None or gt_dose is None or mask is None:
        return

    try:
        xp, yp = compute_dvh(pred_dose, mask)
        xg, yg = compute_dvh(gt_dose, mask)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(xg, yg, label="GT dose")
        ax.plot(xp, yp, label="Pred dose")
        ax.set_title("Clinical DVH")
        ax.set_xlabel("Dose")
        ax.set_ylabel("Volume ≥ dose (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    except Exception as exc:
        st.error(f"DVH failed: {exc}")



def parse_epoch_log(path: Path) -> pd.DataFrame | None:
    """Parse training.log when loss_history.csv is unavailable."""
    if not path.exists():
        return None

    rows = []
    epoch_summary = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+-\s+Train Loss:\s+([0-9.eE+-]+),\s+Val Loss:\s+([0-9.eE+-]+),\s+LR:\s+([0-9.eE+-]+)"
    )
    sanity_summary = re.compile(
        r"Epoch\s+0*(\d+)/0*(\d+)\s+\|\s+train_loss=([0-9.eE+-]+)\s+\|\s+eval_loss=([0-9.eE+-]+)"
    )

    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = epoch_summary.search(line)
            if m:
                rows.append({
                    "epoch": int(m.group(1)),
                    "train_loss": float(m.group(3)),
                    "val_loss": float(m.group(4)),
                    "lr": float(m.group(5)),
                })
                continue

            m = sanity_summary.search(line)
            if m:
                rows.append({
                    "epoch": int(m.group(1)),
                    "train_loss": float(m.group(3)),
                    "eval_loss": float(m.group(4)),
                })
    except Exception as exc:
        st.warning(f"Could not parse log `{path}`: {exc}")
        return None

    if not rows:
        return None

    return pd.DataFrame(rows).drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")


def find_metric_runs(raw_paths: str, recursive: bool) -> dict[str, pd.DataFrame]:
    """
    Find run folders from pasted folders.

    Priority:
        1. loss_history.csv
        2. training.log / same_patient_overfit.log parsed fallback
    """
    runs: dict[str, pd.DataFrame] = {}

    search_roots = [
        Path(line.strip()).expanduser()
        for line in raw_paths.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    for root in search_roots:
        if not root.exists():
            st.warning(f"Path does not exist: `{root}`")
            continue

        candidate_dirs: set[Path] = set()

        if root.is_file():
            candidate_dirs.add(root.parent)
        else:
            candidate_dirs.add(root)
            if recursive:
                for file_name in ["loss_history.csv", "training.log", "same_patient_overfit.log"]:
                    for file_path in root.rglob(file_name):
                        candidate_dirs.add(file_path.parent)

        for folder in sorted(candidate_dirs):
            df = None
            source = None

            loss_csv = folder / "loss_history.csv"
            if loss_csv.exists():
                df = read_csv(loss_csv)
                source = "loss_history.csv"

            if df is None or df.empty:
                for log_name in ["training.log", "same_patient_overfit.log"]:
                    parsed = parse_epoch_log(folder / log_name)
                    if parsed is not None and not parsed.empty:
                        df = parsed
                        source = log_name
                        break

            if df is None or df.empty:
                continue

            if "epoch" not in df.columns:
                df = df.copy()
                df.insert(0, "epoch", np.arange(1, len(df) + 1))

            label = folder.name
            parent = folder.parent.name
            if parent and parent not in {"visualizations", "final_all_paretos"}:
                label = f"{parent}/{folder.name}"

            # Keep labels unique.
            base_label = label
            suffix = 2
            while label in runs:
                label = f"{base_label} #{suffix}"
                suffix += 1

            df = df.copy()
            df.attrs["folder"] = str(folder)
            df.attrs["source"] = source
            runs[label] = df

    return runs


def train_val_multi_folder_plot(output_dir: Path) -> None:
    st.subheader("Train/Val metric comparison across sanity runs")

    st.caption(
        "Paste sanity output folders or a parent folder. "
        "The app searches for `loss_history.csv`, or parses `training.log` / `same_patient_overfit.log`."
    )

    raw_paths = st.text_area(
        "Folders, one per line",
        value=str(output_dir),
        height=140,
        key="metric_compare_folders_clean",
    )

    recursive = st.checkbox(
        "Search recursively inside these folders",
        value=True,
        key="metric_compare_recursive_clean",
    )

    runs = find_metric_runs(raw_paths, recursive=recursive)

    if not runs:
        st.info("No runs found. Expected `loss_history.csv`, `training.log`, or `same_patient_overfit.log`.")
        return

    with st.expander("Detected runs", expanded=True):
        detected_rows = [
            {
                "run": label,
                "folder": df.attrs.get("folder", ""),
                "source": df.attrs.get("source", ""),
                "epochs": int(df["epoch"].max()) if "epoch" in df.columns else len(df),
            }
            for label, df in runs.items()
        ]
        st.dataframe(pd.DataFrame(detected_rows), use_container_width=True)

    selected_runs = st.multiselect(
        "Runs to plot",
        options=list(runs.keys()),
        default=list(runs.keys()),
        key="metric_compare_selected_runs_clean",
    )

    if not selected_runs:
        st.info("Choose at least one run.")
        return

    metric_options = sorted({
        col
        for label in selected_runs
        for col in runs[label].columns
        if col != "epoch" and pd.api.types.is_numeric_dtype(runs[label][col])
    })

    if not metric_options:
        st.info("No numeric metrics found.")
        return

    preferred = [
        "eval_loss",
        "val_loss",
        "train_loss",
        "eval_mae",
        "train_mae",
        "eval_open_l1",
        "train_open_l1",
        "eval_closed_abs_pred",
        "train_closed_abs_pred",
        "mae",
        "loss",
    ]

    default_metric = next((m for m in preferred if m in metric_options), metric_options[0])

    metric = st.selectbox(
        "Metric",
        options=metric_options,
        index=metric_options.index(default_metric),
        key="metric_compare_single_metric_clean",
    )

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    for label in selected_runs:
        df = runs[label]
        if metric not in df.columns:
            continue
        ax.plot(df["epoch"], df[metric], marker="o", markersize=2, linewidth=1.6, label=label)

    ax.set_title(f"{metric} comparison")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.markdown("#### Best epoch per run for selected metric")
    summary_rows = []
    for label in selected_runs:
        df = runs[label]
        if metric not in df.columns:
            continue
        idx = int(df[metric].idxmin())
        summary_rows.append({
            "run": label,
            "best_epoch": int(df.loc[idx, "epoch"]),
            f"best_{metric}": float(df.loc[idx, metric]),
            "folder": df.attrs.get("folder", ""),
            "source": df.attrs.get("source", ""),
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    with st.expander("Raw selected metric tables", expanded=False):
        for label in selected_runs:
            df = runs[label]
            if metric in df.columns:
                st.write(f"### {label}")
                st.dataframe(df[["epoch", metric]].tail(50), use_container_width=True)


def loss_tab(output_dir: Path):
    st.subheader("Training / validation curves")
    df = read_csv(output_dir / "loss_history.csv")
    if df is None or df.empty:
        st.info("No loss_history.csv found.")
        return
    st.dataframe(df.tail(20), use_container_width=True)

    xcol = "epoch" if "epoch" in df.columns else df.columns[0]
    num_cols = [c for c in df.columns if c != xcol and pd.api.types.is_numeric_dtype(df[c])]
    defaults = [c for c in ["train_loss", "eval_loss", "val_loss", "train_mae", "eval_mae", "eval_open_l1"] if c in num_cols]
    selected = st.multiselect("Columns", num_cols, default=defaults if defaults else num_cols[:4])
    if not selected:
        return

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for c in selected:
        ax.plot(df[xcol], df[c], label=c)
    ax.set_xlabel(xcol)
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Sinogram Sanity Viewer", layout="wide")
    st.title("Sinogram Sanity Viewer")

    st.sidebar.header("Output folder")
    output_dir = Path(st.sidebar.text_input(
        "Folder",
        value="sanity_outputs/overfit_same_patient_attention_in_agg/patient_324181",
    )).expanduser()

    if not output_dir.exists():
        st.error(f"Output folder does not exist: {output_dir}")
        return

    config = read_json(output_dir / "config.json") or {}
    if isinstance(config, dict):
        with st.sidebar.expander("config.json"):
            st.json(config)
    else:
        config = {}

    arr_dir = final_array_dir(output_dir)
    st.sidebar.caption(f"Array dir: {arr_dir}")
    paretos = discover_paretos(arr_dir)

    if not paretos:
        st.error("No pareto arrays found. Expected pareto_*_target.npy and pareto_*_pred_prob.npy.")
        return

    keys = sorted(paretos.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    selected_pareto = st.sidebar.selectbox("Pareto", keys)

    patient_id = str(config.get("patient_id", ""))
    if not patient_id:
        manifest = read_json(output_dir / "selected_samples.json")
        if isinstance(manifest, list) and manifest:
            patient_id = str(manifest[0].get("patient_id", "unknown"))
        else:
            patient_id = st.sidebar.text_input("Patient ID", value="unknown")
    st.sidebar.write(f"Patient: `{patient_id}`")

    paths = paretos[selected_pareto]
    gt = squeeze2d(load_npy(str(paths["target"])))
    pred = squeeze2d(load_npy(str(paths["pred"]))) if "pred" in paths else np.zeros_like(gt)

    st.caption(f"Output: `{output_dir}`")
    st.caption(f"Pred: `{paths.get('pred')}`")
    st.caption(f"GT: `{paths.get('target')}`")

    show_metric_cards(metrics(pred, gt))

    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "Sinograms 2D",
        "Sinograms 3D",
        "Histograms",
        "CT + RTDose + Struct",
        "DVH",
        "Train/Val",
        "Tables",
    ])

    with t1:
        st.pyplot(fig_sino(pred, gt, f"Patient {patient_id} | Pareto {selected_pareto}"))

    with t2:
        cp_stride = st.slider("CP stride", 1, 20, 4)
        leaf_stride = st.slider("Leaf stride", 1, 8, 1)
        view = st.radio("View", ["Prediction", "GT", "Absolute diff"], horizontal=True)
        arr = pred if view == "Prediction" else gt if view == "GT" else np.abs(pred - gt)
        show_surface(arr, view, cp_stride, leaf_stride)

    with t3:
        bins = st.slider("Bins", 20, 200, 80)
        st.pyplot(fig_hist(pred, gt, bins))

    with t4:
        ct_dose_struct_tab(config, patient_id, selected_pareto)

    with t5:
        dvh_tab(pred, gt)

    with t6:
        train_val_multi_folder_plot(output_dir)

    with t7:
        st.subheader("Metrics / selected samples")
        for name in ["per_sample_metrics_final.csv", "per_sample_metrics_latest.csv"]:
            df = read_csv(output_dir / name)
            if df is not None:
                st.write(f"`{name}`")
                st.dataframe(df, use_container_width=True)
                break
        manifest = read_json(output_dir / "selected_samples.json")
        if manifest is not None:
            st.write("`selected_samples.json`")
            st.json(manifest)


if __name__ == "__main__":
    main()
