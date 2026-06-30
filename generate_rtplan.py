"""
generate_rtplan.py
==================
Génère un fichier DICOM RT-Plan Tomo/Radixact en remplaçant le sinogramme
par la prédiction du modèle DosePrediction.

Usage
-----
    python generate_rtplan.py \
        --patient_dir /mnt/data/shared/tomo_data/<PATIENT_ID>/Tomo_FB_copy \
        --pareto_index 0 \
        --checkpoint checkpoints/20260518_140141/best_model_new_session_session_0_.pth \
        --output predicted_rtplan.dcm

Le script :
  1. Charge le RT-Plan DICOM de référence (pour les métadonnées).
  2. Construit le tenseur d'entrée berlingo (CT + dose) via RTDataset en mode
     inference.
  3. Fait l'inférence avec DosePrediction.
  4. Réinjecte le sinogramme prédit (tag privé 300D,10A7) dans chaque
     ControlPointSequence du RT-Plan de référence.
  5. Sauvegarde le nouveau fichier DICOM RT-Plan.
"""

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import pydicom
import torch

from models.unet import DosePrediction
from utils.patient import RTDataset
from utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sino_row_to_bytes(row: np.ndarray) -> bytes:
    """
    Encode une ligne du sinogramme (64 valeurs float) en bytes DICOM Tomo.

    Format observé dans les fichiers réels (VR=UN) :
      - valeurs séparées par un seul backslash '\\' (0x5C)
      - pas de zéros en fin de décimale : '0.4247242' pas '0.42472420'
      - longueur totale **paire** obligatoire (DICOM spec) → padding espace (0x20) si impair
      - les zéros exacts sont encodés '0' (pas '0.000000')

    Exemple : b'0\\0\\0.4247242\\0.4910204\\0\\...\\0 '
    """
    parts = []
    for v in row:
        if v == 0.0:
            parts.append("0")
        else:
            # Supprimer les zéros trailing après la virgule, comme le fait le TPS
            s = f"{v:.7g}"
            parts.append(s)

    raw = "\\".join(parts).encode("ascii")

    # Padding DICOM : longueur paire (UN VR)
    if len(raw) % 2 != 0:
        raw += b" "

    return raw


def _get_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Retourne le checkpoint le plus récent dans le répertoire."""
    pths = sorted(checkpoint_dir.glob("**/*.pth"))
    if not pths:
        raise FileNotFoundError(f"Aucun .pth trouvé dans {checkpoint_dir}")
    return pths[-1]


# ---------------------------------------------------------------------------
# Inférence
# ---------------------------------------------------------------------------

def run_inference(
    patient_dir: Path,
    pareto_index: int,
    checkpoint_path: Path,
    base_filters: int = 8,
    device: torch.device = None,
) -> np.ndarray:
    """
    Charge le patient, fait l'inférence et retourne le sinogramme prédit.

    Returns
    -------
    sino_pred : np.ndarray, shape [N_CP, 64], valeurs dans [0, 1]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device : {device}")

    # --- Trouver les fichiers du patient ---
    tomo_dir = patient_dir if patient_dir.name == "Tomo_FB_copy" else patient_dir
    ct_files = sorted(tomo_dir.glob("CT*.dcm"))
    rs_files  = list(tomo_dir.glob("RS*.dcm"))
    pareto_dir = tomo_dir / f"pareto_{pareto_index}"
    plan_files = list(pareto_dir.glob("RP*.dcm"))
    dose_files = list(pareto_dir.glob("RD*.dcm"))

    if not ct_files:
        raise FileNotFoundError(f"Pas de fichiers CT dans {tomo_dir}")
    if not plan_files:
        raise FileNotFoundError(f"Pas de RT-Plan dans {pareto_dir}")
    if not dose_files:
        raise FileNotFoundError(f"Pas de RT-Dose dans {pareto_dir}")

    co_json_path = pareto_dir / "co.json"

    logging.info(f"CT : {len(ct_files)} slices | Plan : {plan_files[0].name} | Dose : {dose_files[0].name}")

    # --- Construire le co_inference minimal à partir du RT-Plan ---
    plan_ds = pydicom.dcmread(str(plan_files[0]))
    beam = plan_ds[(0x300A, 0x00B0)][0]
    cps  = beam[(0x300A, 0x0111)].value

    SINO_TAG = (0x300D, 0x10A7)
    co_inference = {}
    for i, cp in enumerate(cps):
        if SINO_TAG in cp:
            val = cp[SINO_TAG].value
            if isinstance(val, (bytes, bytearray)):
                parts = val.decode(errors="ignore").split("\\")
                row = [float(x) for x in parts if x.strip() != ""]
                if len(row) == 64:
                    co_inference[i] = row  # sinogramme original (non utilisé en inférence)

    # --- Dataset en mode inférence ---
    config = Config()
    config.REDUCTION_RATIO = 1

    dataset = RTDataset(
        root_dir=str(tomo_dir.parent.parent),   # remonte à la racine des patients
        reduction_ratio=config.REDUCTION_RATIO,
        use_cache=False,
        inference=True,
        co_inference={"_inference_": True},      # marqueur minimal requis
        debug=tomo_dir.parent.name,              # patient_id = nom du dossier patient
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset vide — vérifiez le chemin patient_dir.")

    # Trouver l'index du pareto souhaité
    sample_idx = None
    for i, s in enumerate(dataset.samples):
        if s["pareto_index"] == pareto_index:
            sample_idx = i
            break
    if sample_idx is None:
        raise ValueError(f"pareto_index={pareto_index} introuvable dans le dataset.")

    sample = dataset[sample_idx]
    inp = sample["input"].unsqueeze(0).to(device)   # [1, 2, N_CP, 64, 64]
    logging.info(f"Tenseur d'entrée : {inp.shape}")

    # --- Modèle ---
    model = DosePrediction(base_filters=base_filters, in_channel=2).to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    logging.info(f"Checkpoint chargé depuis {checkpoint_path}")

    # --- Prédiction ---
    with torch.no_grad():
        pred = model(inp)                        # [1, 1, N_CP, 64, 1]

    # Mise en forme → [N_CP, 64]
    pred_np = pred.squeeze().cpu().numpy()       # [N_CP, 64]
    if pred_np.ndim == 1:
        pred_np = pred_np[np.newaxis, :]         # sécurité pour N_CP=1

    # Le trainer flip les targets lors de l'entraînement → on re-flip la sortie
    pred_np = pred_np[::-1].copy()

    # Clamp dans [0, 1] (les lames ne peuvent pas dépasser 100 % d'ouverture)
    pred_np = np.clip(pred_np, 0.0, 1.0)

    logging.info(f"Sinogramme prédit : {pred_np.shape}, min={pred_np.min():.4f}, max={pred_np.max():.4f}")
    return pred_np, plan_files[0]


# ---------------------------------------------------------------------------
# Écriture DICOM
# ---------------------------------------------------------------------------

def create_rtplan_from_prediction(
    reference_plan_path: Path,
    sino_pred: np.ndarray,
    output_path: Path,
) -> pydicom.Dataset:
    """
    Crée un nouveau fichier DICOM RT-Plan en remplaçant le sinogramme
    par `sino_pred` dans chaque ControlPoint.

    Parameters
    ----------
    reference_plan_path : Path
        Chemin vers le RT-Plan DICOM original (métadonnées conservées).
    sino_pred : np.ndarray, shape [N_CP, 64]
        Sinogramme prédit par le modèle, valeurs dans [0, 1].
    output_path : Path
        Chemin de sortie du nouveau DICOM RT-Plan.

    Returns
    -------
    ds : pydicom.Dataset
        Le dataset DICOM modifié.
    """
    import copy
    from datetime import datetime
    from pydicom.uid import generate_uid

    SINO_TAG = (0x300D, 0x10A7)

    # Deep-copy pour ne pas modifier l'original en mémoire
    ds = copy.deepcopy(pydicom.dcmread(str(reference_plan_path)))

    beam = ds[(0x300A, 0x00B0)][0]
    cps  = beam[(0x300A, 0x0111)].value

    # Filtre : seulement les CPs qui portent déjà le tag sinogramme
    sino_cps = [cp for cp in cps if SINO_TAG in cp]

    n_pred = sino_pred.shape[0]
    n_sino = len(sino_cps)

    if n_pred != n_sino:
        logging.warning(
            f"Taille sinogramme prédit ({n_pred}) ≠ nombre de CPs avec sinogramme ({n_sino}). "
            "On tronque/répète pour aligner."
        )
        if n_pred > n_sino:
            sino_pred = sino_pred[:n_sino]
        else:
            # Répétition du dernier vecteur pour compléter
            pad = np.tile(sino_pred[-1:], (n_sino - n_pred, 1))
            sino_pred = np.vstack([sino_pred, pad])

    # --- Injection du sinogramme prédit ---
    for i, cp in enumerate(sino_cps):
        new_bytes = _sino_row_to_bytes(sino_pred[i])
        cp[SINO_TAG].value = new_bytes
        logging.debug(f"CP {i:04d} : sinogramme mis à jour ({len(new_bytes)} bytes)")

    # --- Mise à jour des UIDs et métadonnées ---
    now = datetime.now()
    ds.SOPInstanceUID = generate_uid()
    ds.StudyDate  = now.strftime("%Y%m%d")
    ds.StudyTime  = now.strftime("%H%M%S")
    ds.SeriesDate = now.strftime("%Y%m%d")
    ds.SeriesTime = now.strftime("%H%M%S")

    # Ajout d'un commentaire pour traçabilité
    ds.PlanDescription = getattr(ds, "PlanDescription", "") + " [SinoGram AI prediction]"

    # --- Sauvegarde ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path), write_like_original=False)
    logging.info(f"RT-Plan DICOM sauvegardé → {output_path}")

    return ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Génère un RT-Plan DICOM avec le sinogramme prédit par le modèle."
    )
    parser.add_argument(
        "--patient_dir",
        type=Path,
        required=True,
        help="Chemin vers le dossier Tomo_FB_copy du patient "
             "(ex: /mnt/data/shared/tomo_data/PATIENT_ID/Tomo_FB_copy)",
    )
    parser.add_argument(
        "--pareto_index",
        type=int,
        default=0,
        help="Index du dossier pareto à utiliser (défaut: 0)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Chemin vers le fichier .pth du modèle. "
             "Si absent, utilise le dernier checkpoint du dossier ./checkpoints/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predicted_rtplan.dcm"),
        help="Chemin de sortie du fichier DICOM RT-Plan (défaut: predicted_rtplan.dcm)",
    )
    parser.add_argument(
        "--base_filters",
        type=int,
        default=8,
        help="Nombre de filtres de base du modèle (défaut: 8, identique à main.py)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device PyTorch : 'cuda', 'cpu', etc. Auto-détecté si absent.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Résolution du checkpoint
    if args.checkpoint is None:
        checkpoint_path = _get_latest_checkpoint(Path("./checkpoints"))
        logging.info(f"Checkpoint auto-détecté : {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint

    device = torch.device(args.device) if args.device else None

    # 1. Inférence
    sino_pred, reference_plan_path = run_inference(
        patient_dir=args.patient_dir,
        pareto_index=args.pareto_index,
        checkpoint_path=checkpoint_path,
        base_filters=args.base_filters,
        device=device,
    )

    # 2. Création du fichier DICOM RT-Plan
    ds = create_rtplan_from_prediction(
        reference_plan_path=reference_plan_path,
        sino_pred=sino_pred,
        output_path=args.output,
    )

    logging.info("Terminé.")
    return ds


if __name__ == "__main__":
    main()
