#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_ct_rtdose_viewer.py

Streamlit DICOM viewer for CT anatomy + RTDose visualization.

Run:
    streamlit run streamlit_ct_rtdose_viewer.py

Dependencies:
    pip install streamlit pydicom SimpleITK plotly matplotlib numpy pandas
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import streamlit as st

try:
    import SimpleITK as sitk
except Exception:
    sitk = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None


APP_TITLE = "CT + RTDose Ground-Truth Viewer"


def require_simpleitk() -> None:
    if sitk is None:
        raise RuntimeError(
            "SimpleITK is required for DICOM geometry and RTDose-to-CT resampling. "
            "Install it with: pip install SimpleITK"
        )


def dicom_header(path: Path) -> pydicom.Dataset | None:
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def find_dicom_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file()])


def find_rtdose_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    candidates: list[Path] = []
    for file_path in find_dicom_files(path):
        ds = dicom_header(file_path)
        if ds is not None and str(getattr(ds, "Modality", "")).upper() == "RTDOSE":
            candidates.append(file_path)

    if candidates:
        return sorted(candidates)

    return sorted(
        [
            p for p in find_dicom_files(path)
            if "dose" in p.name.lower() or p.name.upper().startswith("RD")
        ]
    )


def list_ct_series(ct_dir: Path) -> list[dict[str, Any]]:
    require_simpleitk()

    if not ct_dir.exists():
        return []

    series_ids = list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(ct_dir)) or [])
    infos: list[dict[str, Any]] = []

    for series_id in series_ids:
        files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(ct_dir), series_id))
        if not files:
            continue

        modality = ""
        description = ""
        patient_id = ""
        ds = dicom_header(Path(files[0]))
        if ds is not None:
            modality = str(getattr(ds, "Modality", ""))
            description = str(getattr(ds, "SeriesDescription", ""))
            patient_id = str(getattr(ds, "PatientID", ""))

        if modality.upper() == "CT":
            infos.append(
                {
                    "series_id": series_id,
                    "n_files": len(files),
                    "description": description,
                    "patient_id": patient_id,
                    "label": f"{description or 'CT'} | files={len(files)} | id={series_id[:12]}",
                }
            )

    if infos:
        return sorted(infos, key=lambda x: int(x["n_files"]), reverse=True)

    for series_id in series_ids:
        files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(ct_dir), series_id))
        infos.append(
            {
                "series_id": series_id,
                "n_files": len(files),
                "description": "Unknown series",
                "patient_id": "",
                "label": f"Unknown | files={len(files)} | id={series_id[:12]}",
            }
        )

    return sorted(infos, key=lambda x: int(x["n_files"]), reverse=True)


def read_ct_series(ct_dir: Path, series_id: str | None) -> tuple[Any, np.ndarray, dict[str, Any]]:
    require_simpleitk()

    if series_id is None:
        series_infos = list_ct_series(ct_dir)
        if not series_infos:
            raise FileNotFoundError(f"No CT DICOM series found in: {ct_dir}")
        series_id = str(series_infos[0]["series_id"])

    files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(ct_dir), series_id))
    if not files:
        raise FileNotFoundError(f"No DICOM files found for CT series: {series_id}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    arr = sitk.GetArrayFromImage(image).astype(np.float32)

    ds0 = dicom_header(Path(files[0]))
    meta = {
        "series_id": series_id,
        "n_slices": int(arr.shape[0]),
        "shape_zyx": tuple(int(v) for v in arr.shape),
        "spacing_xyz": tuple(float(v) for v in image.GetSpacing()),
        "origin_xyz": tuple(float(v) for v in image.GetOrigin()),
        "direction": tuple(float(v) for v in image.GetDirection()),
        "patient_id": str(getattr(ds0, "PatientID", "")) if ds0 is not None else "",
        "study_date": str(getattr(ds0, "StudyDate", "")) if ds0 is not None else "",
        "series_description": str(getattr(ds0, "SeriesDescription", "")) if ds0 is not None else "",
    }

    return image, arr, meta


def read_scaled_rtdose_image(dose_path: Path) -> tuple[Any, np.ndarray, dict[str, Any]]:
    require_simpleitk()

    if not dose_path.exists():
        raise FileNotFoundError(f"RTDose file does not exist: {dose_path}")

    ds = pydicom.dcmread(str(dose_path), force=True)
    raw = ds.pixel_array.astype(np.float32)
    scaling = float(getattr(ds, "DoseGridScaling", 1.0))
    dose = raw * scaling

    dose_img_raw = sitk.ReadImage(str(dose_path))
    dose_img = sitk.GetImageFromArray(dose.astype(np.float32))

    try:
        dose_img.CopyInformation(dose_img_raw)
    except Exception:
        dose_img.SetSpacing(dose_img_raw.GetSpacing())
        dose_img.SetOrigin(dose_img_raw.GetOrigin())
        dose_img.SetDirection(dose_img_raw.GetDirection())

    meta = {
        "path": str(dose_path),
        "shape_zyx": tuple(int(v) for v in dose.shape),
        "spacing_xyz": tuple(float(v) for v in dose_img.GetSpacing()),
        "origin_xyz": tuple(float(v) for v in dose_img.GetOrigin()),
        "direction": tuple(float(v) for v in dose_img.GetDirection()),
        "dose_grid_scaling": scaling,
        "dose_units": str(getattr(ds, "DoseUnits", "")),
        "dose_type": str(getattr(ds, "DoseType", "")),
        "max_dose": float(np.nanmax(dose)),
        "mean_dose_nonzero": float(np.nanmean(dose[dose > 0])) if np.any(dose > 0) else 0.0,
    }

    return dose_img, dose.astype(np.float32), meta


def resample_dose_to_ct(dose_img: Any, ct_img: Any) -> tuple[Any, np.ndarray]:
    require_simpleitk()

    resampled = sitk.Resample(
        dose_img,
        ct_img,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    return resampled, sitk.GetArrayFromImage(resampled).astype(np.float32)


@st.cache_data(show_spinner=True)
def load_ct_dose_cached(
    ct_dir_text: str,
    ct_series_id: str | None,
    dose_path_text: str,
) -> dict[str, Any]:
    ct_dir = Path(ct_dir_text).expanduser()
    dose_path = Path(dose_path_text).expanduser()

    ct_img, ct_arr, ct_meta = read_ct_series(ct_dir, ct_series_id)
    dose_img, dose_arr_original, dose_meta = read_scaled_rtdose_image(dose_path)
    _, dose_arr_ct_grid = resample_dose_to_ct(dose_img, ct_img)

    return {
        "ct": ct_arr,
        "dose": dose_arr_ct_grid,
        "dose_original": dose_arr_original,
        "ct_meta": ct_meta,
        "dose_meta": dose_meta,
    }


def window_ct(ct_slice: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    if hi <= lo:
        return np.zeros_like(ct_slice, dtype=np.float32)
    return np.clip((ct_slice - lo) / (hi - lo), 0.0, 1.0)


def get_plane(arr: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "Axial":
        return arr[index, :, :]
    if plane == "Coronal":
        return arr[:, index, :]
    if plane == "Sagittal":
        return arr[:, :, index]
    raise ValueError(f"Unknown plane: {plane}")


def plane_max_index(shape_zyx: tuple[int, int, int], plane: str) -> int:
    z, y, x = shape_zyx
    if plane == "Axial":
        return z - 1
    if plane == "Coronal":
        return y - 1
    if plane == "Sagittal":
        return x - 1
    raise ValueError(f"Unknown plane: {plane}")


def default_plane_index(shape_zyx: tuple[int, int, int], plane: str) -> int:
    return plane_max_index(shape_zyx, plane) // 2


def dose_masked(dose_slice: np.ndarray, threshold: float) -> np.ma.MaskedArray:
    return np.ma.masked_where(dose_slice < threshold, dose_slice)


def plot_2d_slice(
    ct: np.ndarray,
    dose: np.ndarray,
    plane: str,
    index: int,
    window_center: float,
    window_width: float,
    dose_max: float,
    dose_threshold: float,
    alpha: float,
    contour: bool,
) -> plt.Figure:
    ct_slice = get_plane(ct, plane, index)
    dose_slice = get_plane(dose, plane, index)

    ct_display = window_ct(ct_slice, window_center, window_width)
    dose_display = dose_masked(dose_slice, dose_threshold)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].imshow(ct_display, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"CT anatomy - {plane} slice {index}")
    axes[0].axis("off")

    im1 = axes[1].imshow(dose_slice, cmap="hot", vmin=0, vmax=dose_max)
    axes[1].set_title("RTDose on CT grid")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Gy")

    axes[2].imshow(ct_display, cmap="gray", vmin=0, vmax=1)
    im2 = axes[2].imshow(dose_display, cmap="hot", vmin=0, vmax=dose_max, alpha=alpha)
    if contour and np.nanmax(dose_slice) > dose_threshold:
        levels = [0.25 * dose_max, 0.5 * dose_max, 0.75 * dose_max]
        levels = [level for level in levels if dose_threshold < level < float(np.nanmax(dose_slice))]
        if levels:
            axes[2].contour(dose_slice, levels=levels, linewidths=0.8)
    axes[2].set_title("CT + RTDose overlay")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Gy")

    return fig


def plot_mip_projection(
    ct: np.ndarray,
    dose: np.ndarray,
    axis: int,
    window_center: float,
    window_width: float,
    dose_max: float,
    dose_threshold: float,
    alpha: float,
    title: str,
) -> plt.Figure:
    ct_mip = np.max(ct, axis=axis)
    dose_mip = np.max(dose, axis=axis)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].imshow(window_ct(ct_mip, window_center, window_width), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"CT MIP - {title}")
    axes[0].axis("off")

    im1 = axes[1].imshow(dose_mip, cmap="hot", vmin=0, vmax=dose_max)
    axes[1].set_title(f"Dose MIP - {title}")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Gy")

    axes[2].imshow(window_ct(ct_mip, window_center, window_width), cmap="gray", vmin=0, vmax=1)
    im2 = axes[2].imshow(dose_masked(dose_mip, dose_threshold), cmap="hot", vmin=0, vmax=dose_max, alpha=alpha)
    axes[2].set_title(f"Overlay MIP - {title}")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Gy")

    return fig


def plot_histograms(ct: np.ndarray, dose: np.ndarray, dose_threshold: float) -> plt.Figure:
    ct_vals = ct[np.isfinite(ct)]
    dose_vals = dose[np.isfinite(dose)]
    dose_nonzero = dose_vals[dose_vals > dose_threshold]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].hist(ct_vals.ravel(), bins=120, range=(-1200, 1200))
    axes[0].set_title("CT HU distribution")
    axes[0].set_xlabel("HU")
    axes[0].set_ylabel("Voxel count")

    if dose_nonzero.size:
        axes[1].hist(dose_nonzero.ravel(), bins=120)
    axes[1].set_title(f"RTDose distribution > {dose_threshold:.2f} Gy")
    axes[1].set_xlabel("Dose [Gy]")
    axes[1].set_ylabel("Voxel count")

    return fig


def downsample_volume(arr: np.ndarray, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    return arr[::stride, ::stride, ::stride]


def plot_3d_volume(
    ct: np.ndarray,
    dose: np.ndarray,
    window_center: float,
    window_width: float,
    dose_max: float,
    dose_threshold: float,
    stride: int,
    show_ct_body: bool,
) -> None:
    if go is None:
        st.warning("Plotly is not installed. Install with: pip install plotly")
        return

    ct_ds = downsample_volume(ct, stride)
    dose_ds = downsample_volume(dose, stride)

    zz, yy, xx = np.indices(ct_ds.shape)
    fig = go.Figure()

    if show_ct_body:
        ct_norm = window_ct(ct_ds, window_center, window_width)
        body_value = float(np.clip(window_ct(np.array([-500.0]), window_center, window_width)[0], 0.05, 0.95))
        fig.add_trace(
            go.Isosurface(
                x=xx.flatten(),
                y=yy.flatten(),
                z=zz.flatten(),
                value=ct_norm.flatten(),
                isomin=body_value,
                isomax=1.0,
                surface_count=1,
                opacity=0.12,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name="CT body",
                colorscale="Gray",
                showscale=False,
            )
        )

    dose_value = np.clip(dose_ds, 0.0, dose_max)
    if float(np.nanmax(dose_value)) > dose_threshold:
        fig.add_trace(
            go.Volume(
                x=xx.flatten(),
                y=yy.flatten(),
                z=zz.flatten(),
                value=dose_value.flatten(),
                isomin=float(dose_threshold),
                isomax=float(dose_max),
                opacity=0.18,
                surface_count=14,
                colorscale="Hot",
                name="RTDose",
                colorbar=dict(title="Gy"),
            )
        )

    fig.update_layout(
        title=f"3D CT anatomy + RTDose volume, stride={stride}",
        height=800,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def metadata_table(ct_meta: dict[str, Any], dose_meta: dict[str, Any], ct: np.ndarray, dose: np.ndarray) -> pd.DataFrame:
    rows = [
        {"field": "Patient ID", "value": ct_meta.get("patient_id", "")},
        {"field": "CT shape [z,y,x]", "value": str(tuple(ct.shape))},
        {"field": "Dose shape on CT grid [z,y,x]", "value": str(tuple(dose.shape))},
        {"field": "CT spacing [x,y,z] mm", "value": str(ct_meta.get("spacing_xyz", ""))},
        {"field": "CT origin [x,y,z]", "value": str(ct_meta.get("origin_xyz", ""))},
        {"field": "RTDose original shape [z,y,x]", "value": str(dose_meta.get("shape_zyx", ""))},
        {"field": "RTDose original spacing [x,y,z] mm", "value": str(dose_meta.get("spacing_xyz", ""))},
        {"field": "DoseGridScaling", "value": str(dose_meta.get("dose_grid_scaling", ""))},
        {"field": "Dose units", "value": str(dose_meta.get("dose_units", ""))},
        {"field": "Max dose on CT grid [Gy]", "value": f"{float(np.nanmax(dose)):.4f}"},
        {
            "field": "Mean nonzero dose on CT grid [Gy]",
            "value": f"{float(np.nanmean(dose[dose > 0])):.4f}" if np.any(dose > 0) else "0",
        },
        {"field": "CT HU min/max", "value": f"{float(np.nanmin(ct)):.1f} / {float(np.nanmax(ct)):.1f}"},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("DICOM inputs")

        ct_dir_text = st.text_input("CT DICOM series folder", value="/mnt/data/shared/tomo_data/")
        ct_dir = Path(ct_dir_text).expanduser()

        series_infos: list[dict[str, Any]] = []
        selected_series_id = None

        if sitk is not None and ct_dir.exists():
            try:
                series_infos = list_ct_series(ct_dir)
            except Exception as exc:
                st.warning(f"Could not list CT series: {exc}")

        if series_infos:
            selected_label = st.selectbox(
                "CT series",
                options=[info["label"] for info in series_infos],
                index=0,
            )
            selected_series_id = next(info["series_id"] for info in series_infos if info["label"] == selected_label)
        else:
            st.info("No CT series listed yet. Check the folder or install SimpleITK.")

        dose_input_text = st.text_input("RTDose file or folder", value="/mnt/data/shared/tomo_data/")
        dose_input = Path(dose_input_text).expanduser()

        dose_files = find_rtdose_files(dose_input) if dose_input.exists() else []
        if dose_files:
            dose_label_to_path = {str(p): p for p in dose_files}
            selected_dose_label = st.selectbox("RTDose file", options=list(dose_label_to_path.keys()), index=0)
            selected_dose_path = dose_label_to_path[selected_dose_label]
        else:
            selected_dose_path = None
            st.warning("No RTDose file found. Provide a file path or a folder containing RTDose DICOM.")

        load_clicked = st.button("Load CT + RTDose", type="primary")

        st.header("Display")
        window_center = st.number_input("CT window center", value=40.0)
        window_width = st.number_input("CT window width", min_value=1.0, value=400.0)
        overlay_alpha = st.slider("Dose overlay alpha", 0.0, 1.0, 0.45)
        dose_threshold = st.number_input("Dose display threshold [Gy]", min_value=0.0, value=1.0, step=0.5)

    if load_clicked:
        if selected_dose_path is None:
            st.error("No RTDose file selected.")
            return
        try:
            st.session_state["ct_dose_data"] = load_ct_dose_cached(
                str(ct_dir),
                selected_series_id,
                str(selected_dose_path),
            )
        except Exception as exc:
            st.error(f"Loading failed: {exc}")
            return

    data = st.session_state.get("ct_dose_data")
    if not data:
        st.info("Select CT folder and RTDose, then click **Load CT + RTDose**.")
        return

    ct = np.asarray(data["ct"], dtype=np.float32)
    dose = np.asarray(data["dose"], dtype=np.float32)
    ct_meta = dict(data["ct_meta"])
    dose_meta = dict(data["dose_meta"])

    dose_max_default = float(np.nanpercentile(dose[dose > 0], 99.5)) if np.any(dose > 0) else 1.0
    dose_max_default = max(dose_max_default, float(np.nanmax(dose)), 1.0)

    with st.sidebar:
        dose_max = st.number_input(
            "Dose color max [Gy]",
            min_value=0.1,
            value=float(round(dose_max_default, 2)),
            step=1.0,
        )
        contour = st.checkbox("Show dose contours on 2D overlay", value=True)

    st.caption(f"CT shape: `{ct.shape}` | Dose on CT grid: `{dose.shape}` | Max dose: `{np.nanmax(dose):.3f} Gy`")

    tab_2d, tab_3d_mip, tab_3d_volume, tab_hist, tab_meta = st.tabs(
        ["2D slice overlay", "3D projections", "3D volume", "Histograms", "Metadata"]
    )

    with tab_2d:
        col_a, col_b = st.columns([1, 3])

        with col_a:
            plane = st.radio("Plane", ["Axial", "Coronal", "Sagittal"], horizontal=False)
            max_idx = plane_max_index(tuple(ct.shape), plane)
            default_idx = default_plane_index(tuple(ct.shape), plane)
            slice_idx = st.slider("Slice index", 0, max_idx, default_idx)

            ct_slice = get_plane(ct, plane, slice_idx)
            dose_slice = get_plane(dose, plane, slice_idx)

            st.metric("Slice max dose [Gy]", f"{float(np.nanmax(dose_slice)):.3f}")
            st.metric(
                "Slice mean nonzero dose [Gy]",
                f"{float(np.nanmean(dose_slice[dose_slice > 0])):.3f}" if np.any(dose_slice > 0) else "0.000",
            )

        with col_b:
            st.pyplot(
                plot_2d_slice(
                    ct=ct,
                    dose=dose,
                    plane=plane,
                    index=slice_idx,
                    window_center=window_center,
                    window_width=window_width,
                    dose_max=dose_max,
                    dose_threshold=dose_threshold,
                    alpha=overlay_alpha,
                    contour=contour,
                )
            )

    with tab_3d_mip:
        st.subheader("Maximum-intensity projections")
        projection = st.selectbox(
            "Projection direction",
            ["Axial projection over Z", "Coronal projection over Y", "Sagittal projection over X"],
        )
        axis_map = {
            "Axial projection over Z": 0,
            "Coronal projection over Y": 1,
            "Sagittal projection over X": 2,
        }
        st.pyplot(
            plot_mip_projection(
                ct=ct,
                dose=dose,
                axis=axis_map[projection],
                window_center=window_center,
                window_width=window_width,
                dose_max=dose_max,
                dose_threshold=dose_threshold,
                alpha=overlay_alpha,
                title=projection,
            )
        )

    with tab_3d_volume:
        st.subheader("Interactive 3D rendering")
        if go is None:
            st.warning("Plotly is not installed. Install with: pip install plotly")
        else:
            col1, col2 = st.columns(2)
            with col1:
                stride = st.slider("3D downsample stride", 1, 10, 4)
            with col2:
                show_ct_body = st.checkbox("Show approximate CT body isosurface", value=True)

            st.info("Use stride 4-8 for large CT volumes. Lower stride gives better detail but can be slow.")
            plot_3d_volume(
                ct=ct,
                dose=dose,
                window_center=window_center,
                window_width=window_width,
                dose_max=dose_max,
                dose_threshold=dose_threshold,
                stride=stride,
                show_ct_body=show_ct_body,
            )

    with tab_hist:
        st.pyplot(plot_histograms(ct, dose, dose_threshold=dose_threshold))

    with tab_meta:
        st.dataframe(metadata_table(ct_meta, dose_meta, ct, dose), use_container_width=True)

        with st.expander("Raw CT metadata", expanded=False):
            st.json(json.loads(json.dumps(ct_meta, default=str)))

        with st.expander("Raw RTDose metadata", expanded=False):
            st.json(json.loads(json.dumps(dose_meta, default=str)))


if __name__ == "__main__":
    main()
