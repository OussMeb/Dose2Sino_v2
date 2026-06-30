#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Splitter - Gestion déterministe du split train/val/test

Crée ou met à jour un split_manifest.json pour garantir la reproductibilité.

Usage:
    python preprocessing/utils/splitter.py create --data-dir /path/to/data --ratios 0.90 0.07 0.03
    python preprocessing/utils/splitter.py check --data-dir /path/to/data
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Optional


def create_split(
    data_dir: Path,
    output_file: Path,
    seed: int = 44,
    train_ratio: float = 0.90,
    val_ratio: float = 0.07,
    test_ratio: float = 0.03,
    max_test: Optional[int] = None,
) -> dict:
    """
    Crée ou met à jour un split train/val/test déterministe.

    Args:
        data_dir: Dossier contenant les patients (dossiers)
        output_file: Fichier de sortie JSON
        seed: Seed aléatoire pour reproductibilité
        train_ratio: Ratio train (défaut 0.90)
        val_ratio: Ratio val (défaut 0.07)
        test_ratio: Ratio test (défaut 0.03)
        max_test: Maximum de patients en test (override le ratio si plus restrictif)

    Returns:
        dict: Split créé {train: [...], val: [...], test: [...]}
    """
    random.seed(seed)

    # Valider les ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 < total_ratio < 1.01):  # Tolérance pour les arrondis
        print(f"⚠️  Attention: somme des ratios = {total_ratio:.3f} (attendu ~1.0)")

    # Récupérer tous les patients (dossiers)
    all_ids = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    n_total = len(all_ids)

    print(f"📊 Dataset: {n_total} patients trouvés")

    # Charger le split existant si présent
    if output_file.exists():
        with open(output_file) as f:
            split = json.load(f)
        already = set(split.get('train', []) + split.get('val', []) + split.get('test', []))
        print(f"📂 Split existant trouvé: {len(already)} patients déjà assignés")
    else:
        split = {'train': [], 'val': [], 'test': []}
        already = set()

    # Nouveaux patients à assigner
    new_ids = [pid for pid in all_ids if pid not in already]
    if new_ids:
        print(f"✨ {len(new_ids)} nouveaux patients à assigner")
        new_ids_sorted = sorted(new_ids, key=lambda x: hashlib.sha1(x.encode()).hexdigest())
    else:
        print("✅ Tous les patients sont déjà assignés")
        return split

    # Calculer les cibles
    # Si max_test est spécifié, on le respecte
    if max_test and max_test < n_total * test_ratio:
        n_test_target = max_test
        # Redistribuer le remainder entre train et val
        remaining = n_total - n_test_target
        n_train_target = int(round(train_ratio * remaining))
        n_val_target = remaining - n_train_target
        print(f"🔧 max_test={max_test} appliqué (vs ratio {test_ratio:.0%} = {n_total * test_ratio:.0f})")
    else:
        n_train_target = int(round(train_ratio * n_total))
        n_val_target = int(round(val_ratio * n_total))
        n_test_target = n_total - n_train_target - n_val_target

    # Combien ajouter pour atteindre les cibles
    n_train_add = max(0, n_train_target - len(split['train']))
    n_val_add = max(0, n_val_target - len(split['val']))
    n_test_add = max(0, n_test_target - len(split['test']))

    print(f"\n🎯 Cibles calculées:")
    print(f"   Train: {n_train_target} ({n_train_target/n_total*100:.1f}%)")
    print(f"   Val:   {n_val_target} ({n_val_target/n_total*100:.1f}%)")
    print(f"   Test:  {n_test_target} ({n_test_target/n_total*100:.1f}%)")

    # Assigner les nouveaux patients
    assign = ['train'] * n_train_add + ['val'] * n_val_add + ['test'] * n_test_add

    # Padding si nécessaire
    if len(new_ids_sorted) > len(assign):
        assign += ['train'] * (len(new_ids_sorted) - len(assign))

    for pid, bucket in zip(new_ids_sorted[:len(assign)], assign):
        split[bucket].append(pid)

    # Trier pour lisibilité
    for k in split:
        split[k] = sorted(split[k])

    # Sauvegarder
    with open(output_file, 'w') as f:
        json.dump(split, f, indent=2, sort_keys=True)

    print(f"\n✅ Split sauvegardé: {output_file}")
    print(f"   Train: {len(split['train'])} ({len(split['train'])/n_total*100:.1f}%)")
    print(f"   Val:   {len(split['val'])} ({len(split['val'])/n_total*100:.1f}%)")
    print(f"   Test:  {len(split['test'])} ({len(split['test'])/n_total*100:.1f}%)")

    return split


def check_split(data_dir: Path, split_file: Path) -> bool:
    """
    Vérifie la cohérence du split.

    Returns:
        bool: True si cohérent, False sinon
    """
    with open(split_file) as f:
        split = json.load(f)

    all_in_split = set(split['train'] + split['val'] + split['test'])
    all_in_dir = set(d.name for d in data_dir.iterdir() if d.is_dir())

    missing = all_in_dir - all_in_split
    extra = all_in_split - all_in_dir

    # Vérifier duplicatas
    all_flat = split['train'] + split['val'] + split['test']
    duplicates = [x for x in set(all_flat) if all_flat.count(x) > 1]

    print("🔍 Vérification du split:")
    print(f"   Patients dans dataset: {len(all_in_dir)}")
    print(f"   Patients dans split:   {len(all_in_split)}")

    errors = False

    if missing:
        print(f"   ❌ Manquants dans split: {len(missing)}")
        print(f"       Exemples: {list(missing)[:5]}")
        errors = True

    if extra:
        print(f"   ❌ En trop dans split: {len(extra)}")
        print(f"       Exemples: {list(extra)[:5]}")
        errors = True

    if duplicates:
        print(f"   ❌ Duplicatas: {duplicates}")
        errors = True

    if not errors:
        print("   ✅ Split cohérent!")
        print(f"\n📊 Répartition:")
        print(f"   Train: {len(split['train'])} ({len(split['train'])/len(all_in_split)*100:.1f}%)")
        print(f"   Val:   {len(split['val'])} ({len(split['val'])/len(all_in_split)*100:.1f}%)")
        print(f"   Test:  {len(split['test'])} ({len(split['test'])/len(all_in_split)*100:.1f}%)")

    return not errors


def main():
    parser = argparse.ArgumentParser(
        description='Gestion du split train/val/test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Créer un split 90/7/3 avec max 10 en test
  python splitter.py create --data-dir /path/to/data --ratios 0.90 0.07 0.03 --max-test 10
  
  # Vérifier le split existant
  python splitter.py check --data-dir /path/to/data
  
  # Régénérer sur seed différent (pour debug)
  python splitter.py create --data-dir /path/to/data --seed 123
        """
    )

    parser.add_argument('action', choices=['create', 'check'],
                        help='Action à effectuer')
    parser.add_argument('--data-dir', required=True, type=Path,
                        help='Dossier racine des données')
    parser.add_argument('--output', type=Path,
                        help='Fichier split JSON (défaut: data_dir/split_manifest.json)')
    parser.add_argument('--seed', type=int, default=44,
                        help='Seed aléatoire (défaut: 44)')
    parser.add_argument('--ratios', nargs=3, type=float, default=[0.90, 0.07, 0.03],
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='Ratios train/val/test (défaut: 0.90 0.07 0.03)')
    parser.add_argument('--max-test', type=int, default=None,
                        help='Maximum de patients en test (override le ratio si plus restrictif)')

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"❌ Dossier non trouvé: {args.data_dir}")
        return 1

    if args.output is None:
        args.output = args.data_dir / 'split_manifest.json'

    if args.action == 'create':
        create_split(
            args.data_dir,
            args.output,
            seed=args.seed,
            train_ratio=args.ratios[0],
            val_ratio=args.ratios[1],
            test_ratio=args.ratios[2],
            max_test=args.max_test,
        )
    elif args.action == 'check':
        ok = check_split(args.data_dir, args.output)
        return 0 if ok else 1

    return 0


if __name__ == '__main__':
    exit(main())

