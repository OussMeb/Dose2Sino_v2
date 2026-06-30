#!/usr/bin/env python3
"""
Lit X RT-PLAN Tomo, compte chaque valeur de lame (arrondie à 0.1) dans un
Counter, puis trace le décompte.

Le sinogramme est dans le tag privé Tomo (300D,10A7) de chaque ControlPoint
du premier Beam : chaque ligne = 64 lames (leaf open time normalisé ∈ [0,1]),
valeurs séparées par '\\'.

Usage:
    python plot_sinogram_distribution.py RTPLAN.dcm [RTPLAN2.dcm ...]
    python plot_sinogram_distribution.py --dir DATA_DIR [--max-plans N]
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SINO_TAG = (0x300D, 0x10A7)  # tag privé Tomo "Sinogram Data"
N_LEAVES = 64


def extract_leaf_values(plan):
    """Génère toutes les valeurs de lame ∈ [0,1] d'un RT-PLAN."""
    beam = plan[(0x300A, 0x00B0)][0]          # BeamSequence[0]
    cps = beam[(0x300A, 0x0111)].value        # ControlPointSequence
    for cp in cps:
        if SINO_TAG not in cp:
            continue
        val = cp[SINO_TAG].value
        if isinstance(val, (bytes, bytearray)):
            parts = val.decode(errors="ignore").split("\\")
            for x in parts:
                if x.strip() != "":
                    yield float(x)


def plot_counter(counter: Counter, n_plans: int, out_path: Path):
    """Trace le décompte des valeurs de lame arrondies à 0.1."""
    bins = sorted(counter.keys())
    counts = [counter[b] for b in bins]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(bins, counts, width=0.08, color="steelblue", edgecolor="black")
    ax.set_xlabel("Valeur de lame (arrondie à 0.1)")
    ax.set_ylabel("Décompte")
    ax.set_title(f"Distribution des valeurs de lame — {n_plans} RT-PLAN, N={total:,}")
    ax.set_xticks(np.round(np.arange(0.0, 1.01, 0.1), 1))

    # annotate les % au-dessus des barres
    for b, c in zip(bins, counts):
        ax.text(b, c, f"{c / total:.1%}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    logging.info(f"Figure sauvegardée: {out_path}")
    print("\nCounter (valeur -> décompte):")
    for b in bins:
        print(f"  {b:.1f} : {counter[b]:>10,}  ({counter[b] / total:.1%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rtplans", nargs="*", help="Fichiers RT-PLAN (RP*.dcm).")
    parser.add_argument("--dir", help="Répertoire à scanner récursivement pour les RP*.dcm.")
    parser.add_argument("--max-plans", type=int, default=None, help="Limite le nombre de plans lus.")
    parser.add_argument("--out", default="figures/sinogram_leaf_counter.png", help="PNG de sortie.")
    args = parser.parse_args()

    plan_files = [Path(p) for p in args.rtplans]
    if args.dir:
        plan_files += sorted(Path(args.dir).glob("**/RP*.dcm"))
    if args.max_plans:
        plan_files = plan_files[:args.max_plans]

    if not plan_files:
        logging.error("Aucun RT-PLAN fourni (passe des fichiers ou --dir DATA_DIR).")
        return

    counter = Counter()
    n_ok = 0
    for pf in plan_files:
        try:
            plan = pydicom.dcmread(str(pf))
            for v in extract_leaf_values(plan):
                counter[round(v, 1)] += 1
            n_ok += 1
        except Exception as e:
            logging.warning(f"Échec {pf.name}: {e}")

    if not counter:
        logging.error("Aucune valeur de lame extraite.")
        return

    logging.info(f"{n_ok} RT-PLAN lu(s), {sum(counter.values()):,} valeurs de lame.")
    plot_counter(counter, n_ok, Path(args.out))


if __name__ == "__main__":
    main()
