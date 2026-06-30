#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_sinogram_run_viewer.py

One-file Streamlit dashboard for sinogram experiment runs.

Run:
    streamlit run streamlit_sinogram_run_viewer.py

Load options:
    1) local run folders, one per line
    2) uploaded ZIP files containing run folders
    3) loose uploaded CSV/JSON/PNG files

Expected run files, when available:
    config.json
    split_manifest.json
    test_metrics.json
    loss_history.csv
    baseline_val_per_sample.csv
    best_val_per_sample.csv
    val_per_sample_latest.csv
    test_per_sample.csv
    visualizations/*.png
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import torch
except Exception:
    torch = None


CACHE_DIR = Path(".streamlit_sino_cache")


@dataclass
class RunData:
    name: str
    root: Path
    source: str
    config: dict[str, Any]
    split: dict[str, Any]
    test_metrics: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    images: list[Path]


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_first(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if direct.exists():
        return direct
    hits = sorted(root.rglob(filename), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def find_images(root: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(root.rglob(pattern))
    return sorted(images)


def load_run(root: Path, name: str | None = None, source: str = "local") -> RunData | None:
    root = root.expanduser()
    if not root.exists():
        return None

    table_names = [
        "loss_history.csv",
        "baseline_val_per_sample.csv",
        "best_val_per_sample.csv",
        "val_per_sample_latest.csv",
        "test_per_sample.csv",
        "per_sample_metrics_final.csv",
    ]
    tables: dict[str, pd.DataFrame] = {}
    for table_name in table_names:
        table = read_csv(find_first(root, table_name))
        if table is not None:
            tables[table_name] = table

    config = read_json(find_first(root, "config.json"))
    split = read_json(find_first(root, "split_manifest.json"))
    test_metrics = read_json(find_first(root, "test_metrics.json"))
    images = find_images(root)

    if not (config or split or test_metrics or tables or images):
        return None

    return RunData(
        name=name or root.name,
        root=root,
        source=source,
        config=config,
        split=split,
        test_metrics=test_metrics,
        tables=tables,
        images=images,
    )


def safe_extract_zip(data: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        for member in zf.infolist():
            part = Path(member.filename)
            if part.is_absolute() or ".." in part.parts:
                continue
            target = dest / part
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def candidate_roots(root: Path) -> list[Path]:
    candidates: list[tuple[int, Path]] = []
    markers = [
        "config.json",
        "loss_history.csv",
        "test_metrics.json",
        "best_val_per_sample.csv",
        "visualizations",
    ]
    for path in [root, *root.rglob("*")]:
        if not path.is_dir():
            continue
        score = sum(1 for marker in markers if (path / marker).exists())
        if score:
            candidates.append((score, path))
    candidates.sort(key=lambda item: (-item[0], len(item[1].parts)))
    out: list[Path] = []
    seen: set[str] = set()
    for _, path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def load_uploaded_zips(files) -> list[RunData]:
    runs: list[RunData] = []
    if not files:
        return runs
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for file in files:
        data = file.getvalue()
        digest = hashlib.md5(data).hexdigest()
        dest = CACHE_DIR / digest
        if not dest.exists():
            safe_extract_zip(data, dest)
        for root in candidate_roots(dest):
            run = load_run(root, name=f"{file.name}:{root.name}", source="zip")
            if run:
                runs.append(run)
    return runs


def load_uploaded_loose(files) -> list[RunData]:
    if not files:
        return []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(file.getvalue())
    dest = CACHE_DIR / f"loose_{digest.hexdigest()}"
    dest.mkdir(parents=True, exist_ok=True)
    for file in files:
        (dest / Path(file.name).name).write_bytes(file.getvalue())
    run = load_run(dest, name="uploaded_loose_files", source="files")
    return [run] if run else []


def load_local(text: str) -> list[RunData]:
    runs: list[RunData] = []
    for line in text.splitlines():
        path_text = line.strip()
        if not path_text or path_text.startswith("#"):
            continue
        run = load_run(Path(path_text), source="local")
        if run:
            runs.append(run)
        else:
            st.sidebar.warning(f"No valid run found: {path_text}")
    return runs


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if pd.to_numeric(df[col], errors="coerce").notna().any():
            cols.append(col)
    return cols


def parse_image_name(path: Path) -> dict[str, Any]:
    name = path.name
    epoch = re.search(r"epoch[_-](\d+)", name)
    patient = re.search(r"patient[_-]([A-Za-z0-9]+)", name)
    pareto = re.search(r"pareto[_-]([A-Za-z0-9]+)", name)
    split = ""
    for token in ("val_baseline", "baseline", "train", "val", "test", "final"):
        if token in name:
            split = token
            break
    return {
        "file": name,
        "path": str(path),
        "epoch": int(epoch.group(1)) if epoch else None,
        "patient": patient.group(1) if patient else "",
        "pareto": pareto.group(1) if pareto else "",
        "split": split,
    }


def run_summary(run: RunData) -> dict[str, Any]:
    cfg = run.config
    split = run.split
    test = run.test_metrics
    return {
        "run": run.name,
        "source": run.source,
        "patch_version": cfg.get("patch_version", ""),
        "base_filters": cfg.get("base_filters", ""),
        "delta_scale": cfg.get("delta_scale", ""),
        "adv_weight": cfg.get("adv_weight", ""),
        "selection_metric": cfg.get("selection_metric", ""),
        "train_samples": split.get("n_train_samples", ""),
        "val_samples": split.get("n_val_samples", ""),
        "test_samples": split.get("n_test_samples", ""),
        "test_mae": test.get("mae", ""),
        "test_open_l1": test.get("open_l1", ""),
        "test_closed_abs_pred": test.get("closed_abs_pred", ""),
        "baseline_mae": test.get("baseline_mae", ""),
        "baseline_open_l1": test.get("baseline_open_l1", ""),
        "baseline_closed_abs_pred": test.get("baseline_closed_abs_pred", ""),
        "root": str(run.root),
    }


def improvement_summary(runs: list[RunData]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        test = run.test_metrics
        row: dict[str, Any] = {"run": run.name}
        for metric in ("mae", "open_l1", "closed_abs_pred"):
            refined = test.get(metric)
            baseline = test.get(f"baseline_{metric}")
            row[f"refined_{metric}"] = refined
            row[f"baseline_{metric}"] = baseline
            if refined is not None and baseline is not None:
                refined_f = float(refined)
                baseline_f = float(baseline)
                row[f"improvement_{metric}"] = baseline_f - refined_f
                row[f"relative_improvement_{metric}_%"] = 100.0 * (baseline_f - refined_f) / max(abs(baseline_f), 1e-12)
            else:
                row[f"improvement_{metric}"] = np.nan
                row[f"relative_improvement_{metric}_%"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def sidebar_load() -> list[RunData]:
    st.sidebar.header("Load runs")
    local_paths = st.sidebar.text_area(
        "Local run folders",
        height=140,
        placeholder="runs/delta_residual_gan_full/run_...\nruns/residual_gan_refinement_full/run_...",
    )
    zip_files = st.sidebar.file_uploader("Upload run ZIP(s)", type=["zip"], accept_multiple_files=True)
    loose_files = st.sidebar.file_uploader(
        "Upload loose run files",
        type=["csv", "json", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    runs = []
    runs.extend(load_local(local_paths))
    runs.extend(load_uploaded_zips(zip_files))
    runs.extend(load_uploaded_loose(loose_files))

    unique: list[RunData] = []
    seen: set[tuple[str, str]] = set()
    for run in runs:
        key = (run.name, str(run.root))
        if key not in seen:
            seen.add(key)
            unique.append(run)

    if unique:
        names = [run.name for run in unique]
        selected = st.sidebar.multiselect("Active runs", names, default=names)
        unique = [run for run in unique if run.name in selected]

    return unique


def tab_summary(runs: list[RunData]) -> None:
    st.subheader("Run summary")
    st.dataframe(pd.DataFrame([run_summary(run) for run in runs]), use_container_width=True)

    st.subheader("Test improvement vs baseline")
    st.caption("Positive improvement means refined is better/lower than baseline.")
    st.dataframe(improvement_summary(runs), use_container_width=True)

    run = st.selectbox("Inspect JSON", [r.name for r in runs], key="summary_json")
    selected = next(r for r in runs if r.name == run)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### config.json")
        st.json(selected.config)
    with c2:
        st.markdown("#### split_manifest.json")
        st.json(selected.split)
    with c3:
        st.markdown("#### test_metrics.json")
        st.json(selected.test_metrics)


def tab_curves(runs: list[RunData]) -> None:
    st.subheader("Training curves")
    history_runs = [r for r in runs if "loss_history.csv" in r.tables]
    if not history_runs:
        st.info("No loss_history.csv found.")
        return

    all_cols = sorted(set().union(*(numeric_columns(r.tables["loss_history.csv"]) for r in history_runs)))
    metrics = [c for c in all_cols if c != "epoch"]
    default = [c for c in ["val_open_l1", "val_mae", "val_closed_abs_pred", "train_loss", "val_loss"] if c in metrics]
    selected_metrics = st.multiselect("Metrics", metrics, default=default[:3] or metrics[:1])
    window = st.slider("Rolling mean window", 1, 20, 1)

    for metric in selected_metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        for run in history_runs:
            df = run.tables["loss_history.csv"]
            if metric not in df.columns:
                continue
            x = pd.to_numeric(df["epoch"], errors="coerce") if "epoch" in df.columns else pd.Series(range(1, len(df) + 1))
            y = pd.to_numeric(df[metric], errors="coerce")
            if window > 1:
                y = y.rolling(window, min_periods=1).mean()
            ax.plot(x, y, label=run.name)
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        st.pyplot(fig, clear_figure=True)


def tab_tables(runs: list[RunData]) -> None:
    st.subheader("CSV tables")
    table_names = sorted(set().union(*(r.tables.keys() for r in runs)))
    if not table_names:
        st.info("No CSV tables found.")
        return

    table_name = st.selectbox("Table", table_names)
    patient_filter = st.text_input("Patient filter", "")
    pareto_filter = st.text_input("Pareto filter", "")

    for run in runs:
        if table_name not in run.tables:
            continue
        st.markdown(f"### {run.name}")
        df = run.tables[table_name].copy()
        if patient_filter and "patient_id" in df.columns:
            df = df[df["patient_id"].astype(str).str.contains(patient_filter, case=False, na=False)]
        if pareto_filter and "pareto_index" in df.columns:
            df = df[df["pareto_index"].astype(str).str.contains(pareto_filter, case=False, na=False)]

        st.dataframe(df, use_container_width=True)

        num_cols = numeric_columns(df)
        if num_cols:
            with st.expander("Numeric summary"):
                numeric = df[num_cols].apply(pd.to_numeric, errors="coerce")
                st.dataframe(numeric.describe().T, use_container_width=True)


def tab_visuals(runs: list[RunData]) -> None:
    st.subheader("Visualization browser")
    rows = []
    for run in runs:
        for image in run.images:
            row = parse_image_name(image)
            row["run"] = run.name
            rows.append(row)

    if not rows:
        st.info("No PNG/JPG visualizations found.")
        return

    df = pd.DataFrame(rows)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        run_name = st.selectbox("Run", sorted(df["run"].unique()))
    sub = df[df["run"] == run_name]

    with c2:
        split_options = ["all"] + sorted([x for x in sub["split"].dropna().unique() if x])
        split = st.selectbox("Split", split_options)
    if split != "all":
        sub = sub[sub["split"] == split]

    with c3:
        patient_options = ["all"] + sorted([x for x in sub["patient"].dropna().unique() if x])
        patient = st.selectbox("Patient", patient_options)
    if patient != "all":
        sub = sub[sub["patient"] == patient]

    with c4:
        pareto_options = ["all"] + sorted([x for x in sub["pareto"].dropna().unique() if x])
        pareto = st.selectbox("Pareto", pareto_options)
    if pareto != "all":
        sub = sub[sub["pareto"] == pareto]

    sub = sub.sort_values(["epoch", "file"], na_position="last")
    if sub.empty:
        st.warning("No image matches the filters.")
        return

    choice = st.selectbox("Image", sub["file"].tolist(), index=len(sub) - 1)
    selected = sub[sub["file"] == choice].iloc[0]
    path = Path(selected["path"])
    st.image(Image.open(path), caption=str(path), use_container_width=True)

    with st.expander("Image metadata"):
        st.json(selected.to_dict())


def tab_compare_images(runs: list[RunData]) -> None:
    st.subheader("Side-by-side image comparison")
    rows = []
    for run in runs:
        for image in run.images:
            row = parse_image_name(image)
            row["run"] = run.name
            rows.append(row)

    if not rows:
        st.info("No images found.")
        return

    df = pd.DataFrame(rows)
    labels = (df["run"].astype(str) + " | " + df["file"].astype(str)).tolist()
    default = labels[-2:] if len(labels) >= 2 else labels
    selected = st.multiselect("Images", labels, default=default)

    if not selected:
        return

    cols = st.columns(min(3, len(selected)))
    for i, label in enumerate(selected):
        run_name, file_name = label.split(" | ", 1)
        row = df[(df["run"] == run_name) & (df["file"] == file_name)].iloc[0]
        with cols[i % len(cols)]:
            st.image(Image.open(row["path"]), caption=label, use_container_width=True)


def tab_delta(runs: list[RunData]) -> None:
    st.subheader("Delta diagnostics")
    table_priority = ["test_per_sample.csv", "best_val_per_sample.csv", "val_per_sample_latest.csv"]
    rows = []

    for run in runs:
        for table_name in table_priority:
            if table_name not in run.tables:
                continue
            temp = run.tables[table_name].copy()
            temp["run"] = run.name
            temp["table"] = table_name
            rows.append(temp)

    if not rows:
        st.info("No per-sample tables found.")
        return

    df = pd.concat(rows, ignore_index=True)
    for col in numeric_columns(df):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"baseline_mae", "mae"}.issubset(df.columns):
        df["mae_improvement"] = df["baseline_mae"] - df["mae"]
    if {"baseline_open_l1", "open_l1"}.issubset(df.columns):
        df["open_l1_improvement"] = df["baseline_open_l1"] - df["open_l1"]
    if {"baseline_closed_abs_pred", "closed_abs_pred"}.issubset(df.columns):
        df["closed_improvement"] = df["baseline_closed_abs_pred"] - df["closed_abs_pred"]

    preferred = [
        "run",
        "table",
        "patient_id",
        "pareto_index",
        "mae",
        "baseline_mae",
        "mae_improvement",
        "open_l1",
        "baseline_open_l1",
        "open_l1_improvement",
        "closed_abs_pred",
        "baseline_closed_abs_pred",
        "closed_improvement",
        "delta_mae",
        "mean_abs_delta",
    ]
    cols = [c for c in preferred if c in df.columns]
    st.dataframe(df[cols], use_container_width=True)

    group_cols = [c for c in ["mae_improvement", "open_l1_improvement", "closed_improvement", "delta_mae", "mean_abs_delta"] if c in df.columns]
    if group_cols:
        st.markdown("### Mean diagnostics")
        st.dataframe(df.groupby(["run", "table"])[group_cols].mean().reset_index(), use_container_width=True)

    plot_metric = st.selectbox("Scatter metric", group_cols if group_cols else numeric_columns(df))
    if plot_metric:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = df.dropna(subset=[plot_metric])
        ax.scatter(range(len(plot_df)), plot_df[plot_metric])
        if "improvement" in plot_metric:
            ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(plot_metric)
        ax.set_xlabel("sample")
        ax.set_ylabel(plot_metric)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)


def tab_raw(runs: list[RunData]) -> None:
    st.subheader("Raw files")
    run_name = st.selectbox("Run", [r.name for r in runs], key="raw_run")
    run = next(r for r in runs if r.name == run_name)

    st.markdown("### JSON")
    json_choice = st.selectbox("JSON", ["config", "split", "test_metrics"])
    st.json({"config": run.config, "split": run.split, "test_metrics": run.test_metrics}[json_choice])

    if run.tables:
        st.markdown("### CSV")
        table = st.selectbox("Table", list(run.tables.keys()))
        df = run.tables[table]
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{run.name}_{table}",
            mime="text/csv",
        )



# -------------------------
# Live GPU inference
# -------------------------

def add_project_root(project_root: str) -> None:
    root = str(Path(project_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_inference_device(choice: str):
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but is not available.")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=True)
def cached_rtdataset(
    project_root: str,
    data_path: str,
    cache_dir: str,
    max_dose: float,
    reduction_ratio: int,
    use_cache: bool,
):
    add_project_root(project_root)
    patient_module = importlib.import_module("utils.patient")
    RTDataset = getattr(patient_module, "RTDataset")
    return RTDataset(
        root_dir=data_path,
        augmentation=None,
        max_dose=max_dose,
        reduction_ratio=reduction_ratio,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )


def dataset_patients(dataset) -> list[str]:
    return sorted({str(sample.get("patient_id")) for sample in dataset.samples})


def dataset_paretos(dataset, patient_id: str) -> list[str]:
    values = {
        str(sample.get("pareto_index"))
        for sample in dataset.samples
        if str(sample.get("patient_id")) == str(patient_id)
    }
    return sorted(values, key=lambda x: int(x) if x.isdigit() else x)


def find_sample_index(dataset, patient_id: str, pareto_index: str) -> int | None:
    for idx, sample in enumerate(dataset.samples):
        if str(sample.get("patient_id")) == str(patient_id) and str(sample.get("pareto_index")) == str(pareto_index):
            return idx
    return None


def load_checkpoint_for_inference(path_text: str, device):
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def checkpoint_config(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        return ckpt["config"]
    return {}


def clean_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def state_dict_from_checkpoint(ckpt: Any, candidate_keys: list[str]):
    if not isinstance(ckpt, dict):
        return ckpt

    for key in candidate_keys:
        state = ckpt.get(key)
        if isinstance(state, dict):
            return state

    tensor_like = True
    for value in ckpt.values():
        if not hasattr(value, "shape"):
            tensor_like = False
            break
    if tensor_like:
        return ckpt

    return None


def align_target_live(target):
    if target.ndim == 5 and target.shape[1] == 1 and target.shape[-1] == 1:
        return target[:, 0, :, :, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        return target[:, 0, :, :]
    if target.ndim == 3:
        return target
    raise ValueError(f"Unexpected target shape: {tuple(target.shape)}")


def align_logits_live(output):
    if output.ndim == 5 and output.shape[1] == 1 and output.shape[-1] == 1:
        return output[:, 0, :, :, 0]
    if output.ndim == 4 and output.shape[1] == 1:
        return output[:, 0, :, :]
    if output.ndim == 3:
        return output
    raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}")


def make_condition_maps_live(inputs, reduction: str):
    x = inputs[:, :2]
    if reduction == "mean":
        return x.mean(dim=-1)
    if reduction == "max":
        return x.amax(dim=-1)
    if reduction == "meanmax":
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
    raise ValueError(f"Unknown condition_reduction={reduction}")


def build_generator_for_inference(
    project_root: str,
    ckpt: Any,
    cfg: dict[str, Any],
    device,
    base_filters_override: int | None,
):
    add_project_root(project_root)
    module = importlib.import_module("models.unet_attention_in_agg")
    Model = getattr(module, "DosePredictionAttentionInAgg")

    base_filters = int(base_filters_override or cfg.get("base_filters", 16))
    attention_kernel_size = int(cfg.get("attention_kernel_size", 15))
    detector_width = int(cfg.get("detector_width", 64))

    model = Model(
        base_filters=base_filters,
        in_channel=2,
        attention_kernel_size=attention_kernel_size,
        detector_width=detector_width,
    ).to(device)

    state = state_dict_from_checkpoint(ckpt, ["model_state_dict", "generator_state_dict", "state_dict"])
    if state is None:
        raise ValueError("Could not find generator state dict in checkpoint.")

    model.load_state_dict(clean_state_dict(state), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model


def build_refiner_for_inference(project_root: str, ckpt: Any, cfg: dict[str, Any], device):
    if not isinstance(ckpt, dict) or "refiner_state_dict" not in ckpt:
        return None

    add_project_root(project_root)
    module = importlib.import_module("models.sino_residual_refiner")
    Refiner = getattr(module, "SinoResidualRefiner2D")

    condition_reduction = str(cfg.get("condition_reduction", "mean"))
    condition_channels = 2 if condition_reduction in {"mean", "max"} else 4

    refiner = Refiner(
        condition_channels=condition_channels,
        base_channels=int(cfg.get("refiner_base_channels", 32)),
        delta_scale=float(cfg.get("delta_scale", 0.10)),
    ).to(device)

    refiner.load_state_dict(clean_state_dict(ckpt["refiner_state_dict"]), strict=True)
    refiner.eval()
    for param in refiner.parameters():
        param.requires_grad_(False)

    return refiner


def run_checkpoint_inference(
    project_root: str,
    checkpoint_path: str,
    dataset,
    patient_id: str,
    pareto_index: str,
    device_choice: str,
    base_filters_override: int | None,
):
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")

    device = resolve_inference_device(device_choice)
    ckpt = load_checkpoint_for_inference(checkpoint_path, device)
    cfg = checkpoint_config(ckpt)

    generator = build_generator_for_inference(project_root, ckpt, cfg, device, base_filters_override)
    refiner = build_refiner_for_inference(project_root, ckpt, cfg, device)

    sample_idx = find_sample_index(dataset, patient_id, pareto_index)
    if sample_idx is None:
        raise ValueError(f"Patient={patient_id}, pareto={pareto_index} not found.")

    sample = dataset[sample_idx]
    inputs = sample["input"].unsqueeze(0).float().to(device)
    target = sample["target"].unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = align_logits_live(generator(inputs))
        baseline = torch.sigmoid(logits)
        gt = align_target_live(target)

        if refiner is None:
            pred = baseline
            delta = None
        else:
            reduction = str(cfg.get("condition_reduction", "mean"))
            condition = make_condition_maps_live(inputs, reduction)
            refined_dict = refiner(condition, baseline.unsqueeze(1))
            pred = refined_dict["refined"][:, 0]
            delta = refined_dict["delta"][:, 0]

    return {
        "input": inputs[0].detach().cpu().float().numpy(),
        "target": gt[0].detach().cpu().float().numpy(),
        "baseline": baseline[0].detach().cpu().float().numpy(),
        "pred": pred[0].detach().cpu().float().numpy(),
        "delta": None if delta is None else delta[0].detach().cpu().float().numpy(),
        "config": cfg,
        "device": str(device),
        "has_refiner": refiner is not None,
        "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
    }


def sinogram_metrics_np(pred: np.ndarray, gt: np.ndarray, baseline: np.ndarray | None = None) -> dict[str, float]:
    diff = np.abs(pred - gt)
    open_mask = gt > 1e-6
    closed_mask = ~open_mask

    metrics = {
        "mae": float(diff.mean()),
        "open_l1": float(diff[open_mask].mean()) if open_mask.any() else float("nan"),
        "closed_abs_pred": float(pred[closed_mask].mean()) if closed_mask.any() else float("nan"),
        "max_abs": float(diff.max()),
        "pred_mean": float(pred.mean()),
        "target_mean": float(gt.mean()),
    }

    if baseline is not None:
        base_diff = np.abs(baseline - gt)
        metrics["baseline_mae"] = float(base_diff.mean())
        metrics["baseline_open_l1"] = float(base_diff[open_mask].mean()) if open_mask.any() else float("nan")
        metrics["baseline_closed_abs_pred"] = float(baseline[closed_mask].mean()) if closed_mask.any() else float("nan")
        metrics["mae_improvement"] = metrics["baseline_mae"] - metrics["mae"]
        metrics["open_l1_improvement"] = metrics["baseline_open_l1"] - metrics["open_l1"]
        metrics["closed_improvement"] = metrics["baseline_closed_abs_pred"] - metrics["closed_abs_pred"]

    return metrics


def array_to_npy_bytes(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return buffer.getvalue()


def _imshow_with_colorbar_live(
    fig,
    ax,
    img: np.ndarray,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    aspect: str = "equal",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    im = ax.imshow(
        img,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
        aspect=aspect,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)


def plot_live_inference_result(
    result: dict[str, Any],
    max_dose: float,
    title: str,
    display_mode: str = "Tall clinical sinograms",
    sino_cmap: str = "hot",
    absdiff_cmap: str = "magma",
    signed_cmap: str = "coolwarm",
) -> list[plt.Figure]:
    """
    Display sinograms without horizontal/vertical distortion.

    Tall mode preserves pixel geometry:
        y-axis = control points
        x-axis = leaves
    """
    baseline = result["baseline"]
    pred = result["pred"]
    gt = result["target"]
    delta = result["delta"]

    abs_base_diff = np.abs(baseline - gt)
    abs_pred_diff = np.abs(pred - gt)
    signed_pred_diff = pred - gt

    inp = result["input"]
    mid = inp.shape[1] // 2
    ct = inp[0, mid]
    dose = inp[1, mid] * float(max_dose)

    figures: list[plt.Figure] = []

    if display_mode == "Compact overview":
        if delta is None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
        else:
            fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)

        _imshow_with_colorbar_live(fig, axes[0, 0], baseline, "Frozen baseline", sino_cmap, 0, 1, "auto")
        _imshow_with_colorbar_live(fig, axes[0, 1], pred, "Prediction / refined", sino_cmap, 0, 1, "auto")
        _imshow_with_colorbar_live(fig, axes[0, 2], gt, "GT", sino_cmap, 0, 1, "auto")

        _imshow_with_colorbar_live(fig, axes[1, 0], abs_base_diff, "|Baseline - GT|", absdiff_cmap, 0, 0.5, "auto")
        _imshow_with_colorbar_live(fig, axes[1, 1], abs_pred_diff, "|Pred - GT|", absdiff_cmap, 0, 0.5, "auto")
        _imshow_with_colorbar_live(fig, axes[1, 2], signed_pred_diff, "Pred - GT", signed_cmap, -0.3, 0.3, "auto")

        if delta is not None:
            delta_target = gt - baseline
            vmax = max(
                float(np.nanpercentile(np.abs(delta), 99)),
                float(np.nanpercentile(np.abs(delta_target), 99)),
                1e-3,
            )
            vmax = min(vmax, 0.15)
            _imshow_with_colorbar_live(fig, axes[2, 0], delta, "Predicted delta", signed_cmap, -vmax, vmax, "auto")
            _imshow_with_colorbar_live(fig, axes[2, 1], delta_target, "Target delta = GT - baseline", signed_cmap, -vmax, vmax, "auto")
            _imshow_with_colorbar_live(fig, axes[2, 2], dose, "Dose Berlingo mid CP", "inferno", 0, max_dose, "equal")

        fig.suptitle(title, fontsize=13)
        figures.append(fig)
        return figures

    n_cp, _ = gt.shape
    height = float(np.clip(n_cp / 70.0, 12.0, 28.0))
    n_cols = 5 if delta is None else 6
    width = 2.25 * n_cols

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(width, height),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1] * n_cols},
    )
    axes = np.atleast_1d(axes)

    _imshow_with_colorbar_live(fig, axes[0], baseline, "Baseline", sino_cmap, 0, 1, "equal", "Leaf", "Control point")
    _imshow_with_colorbar_live(fig, axes[1], pred, "Prediction", sino_cmap, 0, 1, "equal", "Leaf", "")
    _imshow_with_colorbar_live(fig, axes[2], gt, "Ground truth", sino_cmap, 0, 1, "equal", "Leaf", "")
    _imshow_with_colorbar_live(fig, axes[3], abs_pred_diff, "|Pred - GT|", absdiff_cmap, 0, 0.5, "equal", "Leaf", "")
    _imshow_with_colorbar_live(fig, axes[4], signed_pred_diff, "Pred - GT", signed_cmap, -0.3, 0.3, "equal", "Leaf", "")

    if delta is not None:
        delta_target = gt - baseline
        vmax = max(
            float(np.nanpercentile(np.abs(delta), 99)),
            float(np.nanpercentile(np.abs(delta_target), 99)),
            1e-3,
        )
        vmax = min(vmax, 0.15)
        _imshow_with_colorbar_live(fig, axes[5], delta, "Predicted delta", signed_cmap, -vmax, vmax, "equal", "Leaf", "")

    fig.suptitle(f"{title} | Preserved sinogram geometry", fontsize=13)
    figures.append(fig)

    if delta is not None:
        context_fig, context_axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        delta_target = gt - baseline
        vmax = max(float(np.nanpercentile(np.abs(delta_target), 99)), 1e-3)
        vmax = min(vmax, 0.15)
        _imshow_with_colorbar_live(context_fig, context_axes[0], delta_target, "Target delta = GT - baseline", signed_cmap, -vmax, vmax, "auto")
        _imshow_with_colorbar_live(context_fig, context_axes[1], ct, "CT Berlingo mid CP", "gray", 0, 1, "equal")
        _imshow_with_colorbar_live(context_fig, context_axes[2], dose, "Dose Berlingo mid CP", "inferno", 0, max_dose, "equal")
    else:
        context_fig, context_axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        _imshow_with_colorbar_live(context_fig, context_axes[0], ct, "CT Berlingo mid CP", "gray", 0, 1, "equal")
        _imshow_with_colorbar_live(context_fig, context_axes[1], dose, "Dose Berlingo mid CP", "inferno", 0, max_dose, "equal")

    context_fig.suptitle(f"{title} | CT/Dose context", fontsize=13)
    figures.append(context_fig)

    return figures


def tab_live_inference() -> None:
    st.subheader("Live GPU inference from checkpoint")

    if torch is None:
        st.error("PyTorch is not installed.")
        return

    st.caption(
        "This loads RTDataset, reads the .pth/.pt checkpoint, selects any patient/pareto, "
        "runs inference on CUDA, and displays pred/GT/diff/delta."
    )

    c1, c2 = st.columns(2)
    with c1:
        project_root = st.text_input("Project root", value=str(Path.cwd()))
        data_path = st.text_input("DATA_PATH", value="/mnt/data/shared/tomo_data/")
        cache_dir = st.text_input("CACHE_DIR", value="/mnt/data/shared/tomo_data/cache_sino")
        checkpoint_path = st.text_input(
            "Checkpoint path",
            value="",
            placeholder="/home/oussama/Desktop/Project/checkpoints/.../best_model.pth",
        )
    with c2:
        max_dose = st.number_input("MAX_DOSE", value=70.0)
        reduction_ratio = st.number_input("REDUCTION_RATIO", min_value=1, value=8)
        use_cache = st.checkbox("Use RTDataset cache", value=True)
        device_choice = st.selectbox(
            "Device",
            ["cuda", "auto", "cpu"],
            index=0 if torch.cuda.is_available() else 1,
            help=f"CUDA available: {torch.cuda.is_available()}",
        )
        override_enabled = st.checkbox("Override base_filters", value=False)
        base_filters_override = (
            st.number_input("base_filters override", min_value=1, value=16)
            if override_enabled
            else None
        )

    if st.button("Load dataset / refresh patient list"):
        st.session_state["live_dataset_loaded"] = True

    if not st.session_state.get("live_dataset_loaded"):
        st.info("Click 'Load dataset / refresh patient list'.")
        return

    try:
        dataset = cached_rtdataset(
            project_root=project_root,
            data_path=data_path,
            cache_dir=cache_dir,
            max_dose=float(max_dose),
            reduction_ratio=int(reduction_ratio),
            use_cache=bool(use_cache),
        )
    except Exception as exc:
        st.error(f"Dataset loading failed: {exc}")
        return

    patients = dataset_patients(dataset)
    if not patients:
        st.error("No patients found.")
        return

    s1, s2 = st.columns(2)
    with s1:
        patient_id = st.selectbox("Patient", patients)
    with s2:
        paretos = dataset_paretos(dataset, patient_id)
        pareto_index = st.selectbox("Pareto", paretos)

    if not checkpoint_path.strip():
        st.warning("Enter a checkpoint path.")
        return

    if not st.button("Run inference on selected patient/pareto", type="primary"):
        return

    try:
        with st.spinner("Running CUDA inference..."):
            result = run_checkpoint_inference(
                project_root=project_root,
                checkpoint_path=checkpoint_path,
                dataset=dataset,
                patient_id=str(patient_id),
                pareto_index=str(pareto_index),
                device_choice=device_choice,
                base_filters_override=int(base_filters_override) if base_filters_override is not None else None,
            )
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    metrics = sinogram_metrics_np(result["pred"], result["target"], result["baseline"])

    st.success(
        f"Inference done on {result['device']} | "
        f"checkpoint epoch={result['checkpoint_epoch']} | "
        f"refiner={result['has_refiner']}"
    )

    metric_keys = [
        "mae",
        "open_l1",
        "closed_abs_pred",
        "baseline_mae",
        "baseline_open_l1",
        "mae_improvement",
    ]
    cols = st.columns(len(metric_keys))
    for col, key in zip(cols, metric_keys):
        value = metrics.get(key, float("nan"))
        col.metric(key, f"{value:.6f}" if np.isfinite(value) else "NA")

    st.markdown("### Display options")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        display_mode = st.selectbox(
            "Display mode",
            ["Tall clinical sinograms", "Compact overview"],
            index=0,
        )
    with d2:
        sino_cmap = st.selectbox("Sinogram colormap", ["hot", "inferno", "magma", "viridis"], index=0)
    with d3:
        absdiff_cmap = st.selectbox("Absolute diff colormap", ["magma", "inferno", "viridis", "turbo"], index=0)
    with d4:
        signed_cmap = st.selectbox("Signed/delta colormap", ["coolwarm", "seismic", "bwr", "PiYG"], index=0)

    figures = plot_live_inference_result(
        result,
        max_dose=float(max_dose),
        title=f"Patient {patient_id} | Pareto {pareto_index}",
        display_mode=display_mode,
        sino_cmap=sino_cmap,
        absdiff_cmap=absdiff_cmap,
        signed_cmap=signed_cmap,
    )
    for fig in figures:
        st.pyplot(fig, clear_figure=True)

    with st.expander("Checkpoint config"):
        st.json(result["config"])

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download pred.npy",
        data=array_to_npy_bytes(result["pred"]),
        file_name=f"patient_{patient_id}_pareto_{pareto_index}_pred.npy",
        mime="application/octet-stream",
    )
    d2.download_button(
        "Download target.npy",
        data=array_to_npy_bytes(result["target"]),
        file_name=f"patient_{patient_id}_pareto_{pareto_index}_target.npy",
        mime="application/octet-stream",
    )
    if result["delta"] is not None:
        d3.download_button(
            "Download delta.npy",
            data=array_to_npy_bytes(result["delta"]),
            file_name=f"patient_{patient_id}_pareto_{pareto_index}_delta.npy",
            mime="application/octet-stream",
        )


def main() -> None:
    st.set_page_config(page_title="Sinogram Run Viewer", page_icon="🧠", layout="wide")
    st.title("🧠 Sinogram Run Viewer")
    st.caption("Saved-run comparison + live CUDA inference for any patient/pareto.")

    runs = sidebar_load()

    tabs = st.tabs([
        "Live Inference",
        "Summary",
        "Curves",
        "Tables",
        "Visualizations",
        "Compare Images",
        "Delta Diagnostics",
        "Raw Files",
    ])

    with tabs[0]:
        tab_live_inference()

    if not runs:
        for tab in tabs[1:]:
            with tab:
                st.info("Load saved run folders/ZIPs from the sidebar to use this tab.")
        return

    with tabs[1]:
        tab_summary(runs)
    with tabs[2]:
        tab_curves(runs)
    with tabs[3]:
        tab_tables(runs)
    with tabs[4]:
        tab_visuals(runs)
    with tabs[5]:
        tab_compare_images(runs)
    with tabs[6]:
        tab_delta(runs)
    with tabs[7]:
        tab_raw(runs)


if __name__ == "__main__":
    main()
