#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_sanity_existing_metrics_boxplots.py

Create visually clean boxplots for sanity-check model comparison using existing metrics.

Default metrics:
    Structure-related metric:
        closed_abs_pred
        -> false opening / background leakage.

    Modulation/amplitude metric:
        open_l1
        -> error on open-leaf regions.

Supported CSV names:
    per_sample_metrics_final.csv
    best_val_per_sample.csv
    val_per_sample_latest.csv
    test_per_sample.csv
    loss_history.csv

Run:
    python plot_sanity_existing_metrics_boxplots.py \
      --inputs \
        sanity_outputs/overfit_same_patient_attention_in_agg/patient_324181 \
        sanity_outputs/overfit_same_patient_attention_in_agg_BF8/patient_324181 \
        sanity_outputs/overfit_same_patient_coord_twohead/patient_324181 \
      --output figures/sanity_existing_metric_boxplots.png \
      --pretty-names
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_CANDIDATES = (
    "per_sample_metrics_final.csv",
    "best_val_per_sample.csv",
    "val_per_sample_latest.csv",
    "test_per_sample.csv",
    "loss_history.csv",
)


COLUMN_ALIASES = {
    "closed_abs_pred": (
        "closed_abs_pred",
        "val_closed_abs_pred",
        "eval_closed_abs_pred",
        "test_closed_abs_pred",
        "baseline_closed_abs_pred",
    ),
    "open_l1": (
        "open_l1",
        "val_open_l1",
        "eval_open_l1",
        "test_open_l1",
        "baseline_open_l1",
    ),
    "mae": (
        "mae",
        "val_mae",
        "eval_mae",
        "test_mae",
        "baseline_mae",
    ),
    "loss": (
        "loss",
        "val_loss",
        "eval_loss",
        "test_loss",
    ),
    "train_loss": (
        "train_loss",
        "loss",
    ),
    "eval_loss": (
        "eval_loss",
        "val_loss",
        "loss",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sanity-check existing metrics as boxplots."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Sanity run folders or CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/sanity_existing_metric_boxplots.png"),
    )
    parser.add_argument(
        "--structure-metric",
        type=str,
        default="closed_abs_pred",
        help="Existing metric for structure/background control.",
    )
    parser.add_argument(
        "--modulation-metric",
        type=str,
        default="open_l1",
        help="Existing metric for open-region amplitude/modulation.",
    )
    parser.add_argument(
        "--pretty-names",
        action="store_true",
        help="Shorten run labels.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=0,
        help="Use only last N rows per CSV. 0 means use all rows.",
    )
    return parser.parse_args()


def find_csvs(inputs: list[str]) -> list[Path]:
    csvs: list[Path] = []

    for item in inputs:
        path = Path(item).expanduser().resolve()

        if path.is_file() and path.suffix.lower() == ".csv":
            csvs.append(path)
            continue

        if not path.is_dir():
            print(f"[WARN] Missing input: {path}")
            continue

        for name in CSV_CANDIDATES:
            candidate = path / name
            if candidate.exists():
                csvs.append(candidate)
                break
        else:
            nested_csvs = sorted(path.rglob("*.csv"))
            if nested_csvs:
                csvs.append(nested_csvs[0])
            else:
                print(f"[WARN] No CSV found in: {path}")

    if not csvs:
        raise FileNotFoundError("No CSV files found from --inputs.")

    return csvs


def infer_run_name(csv_path: Path) -> str:
    parent = csv_path.parent

    if parent.name.startswith("patient_"):
        return f"{parent.parent.name}/{parent.name}"

    return parent.name


def pretty_name(name: str) -> str:
    replacements = {
        "overfit_same_patient_attention_in_agg_BF8": "Attention+Agg BF8",
        "overfit_same_patient_attention_in_agg": "Attention+Agg BF16",
        "overfit_same_patient_attention": "Attention",
        "overfit_same_patient_coord_twohead": "Coord+TwoHead",
        "overfit_same_patient": "Baseline",
        "patient_324181": "",
    }

    out = name
    for old, new in replacements.items():
        out = out.replace(old, new)

    out = out.replace("/", " ").replace("__", "_").strip()
    return out or name


def find_metric_column(df: pd.DataFrame, requested_metric: str) -> str | None:
    lower_to_original = {col.lower(): col for col in df.columns}
    aliases = COLUMN_ALIASES.get(requested_metric, (requested_metric,))

    for alias in aliases:
        if alias.lower() in lower_to_original:
            return lower_to_original[alias.lower()]

    requested = requested_metric.lower()
    for col in df.columns:
        if requested in col.lower():
            return col

    return None


def extract_metric_rows(
    csv_path: Path,
    structure_metric: str,
    modulation_metric: str,
    use_pretty_names: bool,
    last_n: int,
) -> list[dict[str, float | str]]:
    df = pd.read_csv(csv_path)

    if last_n > 0:
        df = df.tail(last_n)

    structure_col = find_metric_column(df, structure_metric)
    modulation_col = find_metric_column(df, modulation_metric)

    if structure_col is None and modulation_col is None:
        print(
            f"[WARN] No requested metrics found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
        return []

    run_name = infer_run_name(csv_path)
    if use_pretty_names:
        run_name = pretty_name(run_name)

    rows: list[dict[str, float | str]] = []

    for _, row in df.iterrows():
        item: dict[str, float | str] = {
            "run": run_name,
            "source_csv": str(csv_path),
        }

        if structure_col is not None and pd.notna(row[structure_col]):
            item["structure_value"] = float(row[structure_col])

        if modulation_col is not None and pd.notna(row[modulation_col]):
            item["modulation_value"] = float(row[modulation_col])

        if "structure_value" in item or "modulation_value" in item:
            rows.append(item)

    return rows


def collect_data(
    csvs: list[Path],
    structure_metric: str,
    modulation_metric: str,
    use_pretty_names: bool,
    last_n: int,
) -> pd.DataFrame:
    all_rows: list[dict[str, float | str]] = []

    for csv_path in csvs:
        all_rows.extend(
            extract_metric_rows(
                csv_path=csv_path,
                structure_metric=structure_metric,
                modulation_metric=modulation_metric,
                use_pretty_names=use_pretty_names,
                last_n=last_n,
            )
        )

    if not all_rows:
        raise RuntimeError("No usable metrics found.")

    return pd.DataFrame(all_rows)


def grouped_values(df: pd.DataFrame, column: str, runs: list[str]) -> list[np.ndarray]:
    values = []

    for run in runs:
        if column not in df.columns:
            values.append(np.asarray([], dtype=float))
            continue
        series = df.loc[df["run"] == run, column].dropna()
        values.append(series.to_numpy(dtype=float))

    return values


def make_boxplot(
    ax: plt.Axes,
    values: list[np.ndarray],
    labels: list[str],
    title: str,
    ylabel: str,
    better_text: str,
) -> None:
    box = ax.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.55,
        medianprops={"linewidth": 2.0},
        meanprops={"linewidth": 2.0, "linestyle": "--"},
        whiskerprops={"linewidth": 1.3},
        capprops={"linewidth": 1.3},
        boxprops={"linewidth": 1.3},
    )

    for patch in box["boxes"]:
        patch.set_alpha(0.55)

    rng = np.random.default_rng(7)
    for idx, arr in enumerate(values, start=1):
        if arr.size == 0:
            continue
        jitter = rng.normal(idx, 0.035, size=arr.size)
        ax.scatter(jitter, arr, s=28, alpha=0.68, edgecolors="black", linewidths=0.35)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.30)
    ax.tick_params(axis="x", labelrotation=20)
    ax.text(
        0.99,
        0.97,
        better_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "alpha": 0.12},
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def save_summary(df: pd.DataFrame, output: Path) -> None:
    summary_rows = []

    for run, group in df.groupby("run"):
        row: dict[str, float | str | int] = {
            "run": run,
            "n": int(len(group)),
        }

        if "structure_value" in group:
            values = group["structure_value"].dropna()
            row["structure_mean"] = float(values.mean()) if len(values) else np.nan
            row["structure_std"] = float(values.std()) if len(values) > 1 else 0.0
            row["structure_min"] = float(values.min()) if len(values) else np.nan

        if "modulation_value" in group:
            values = group["modulation_value"].dropna()
            row["modulation_mean"] = float(values.mean()) if len(values) else np.nan
            row["modulation_std"] = float(values.std()) if len(values) > 1 else 0.0
            row["modulation_min"] = float(values.min()) if len(values) else np.nan

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = output.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


def make_plot(
    df: pd.DataFrame,
    output: Path,
    structure_metric: str,
    modulation_metric: str,
) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    runs = list(dict.fromkeys(df["run"].astype(str).tolist()))

    has_structure = "structure_value" in df.columns and df["structure_value"].notna().any()
    has_modulation = "modulation_value" in df.columns and df["modulation_value"].notna().any()

    if not has_structure and not has_modulation:
        raise RuntimeError("No plottable structure or modulation values found.")

    n_cols = int(has_structure) + int(has_modulation)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), constrained_layout=True)

    if n_cols == 1:
        axes = [axes]

    ax_idx = 0

    if has_structure:
        make_boxplot(
            ax=axes[ax_idx],
            values=grouped_values(df, "structure_value", runs),
            labels=runs,
            title="Structural leakage control",
            ylabel=f"{structure_metric} ↓",
            better_text="Lower is better",
        )
        ax_idx += 1

    if has_modulation:
        make_boxplot(
            ax=axes[ax_idx],
            values=grouped_values(df, "modulation_value", runs),
            labels=runs,
            title="Open-leaf modulation accuracy",
            ylabel=f"{modulation_metric} ↓",
            better_text="Lower is better",
        )

    fig.suptitle("Sanity-check model comparison", fontsize=15, fontweight="bold")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    print(f"Saved figure: {output}")

    save_summary(df, output)


def main() -> None:
    args = parse_args()

    csvs = find_csvs(args.inputs)

    print("Using CSV files:")
    for csv_path in csvs:
        print(f"  - {csv_path}")

    df = collect_data(
        csvs=csvs,
        structure_metric=args.structure_metric,
        modulation_metric=args.modulation_metric,
        use_pretty_names=args.pretty_names,
        last_n=args.last_n,
    )

    make_plot(
        df=df,
        output=args.output,
        structure_metric=args.structure_metric,
        modulation_metric=args.modulation_metric,
    )


if __name__ == "__main__":
    main()
