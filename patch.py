#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch.py

Fix generate_rtplan_attention_in_agg.py all-paretos export error.

Problem:
    --save-npy tries to save into:
        raystation_exports/<patient>_all_paretos/*.npy
    before the output directory exists.

Fix:
    Create output_path.parent before saving .npy or preview .png.

Run from Project folder:
    python patch.py
"""

from __future__ import annotations

from pathlib import Path


TARGET = Path("generate_rtplan_attention_in_agg.py")


def patch_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path.resolve()}")

    text = path.read_text(encoding="utf-8")
    original = text

    old_save_optional = """def save_optional_artifacts(sino: np.ndarray, output_path: Path, args: argparse.Namespace) -> None:
    if args.save_npy:
        npy_path = output_path.with_suffix(".npy")
        np.save(str(npy_path), sino)
        logging.info("Saved .npy: %s", npy_path)

    if args.save_preview:
        save_preview(sino, output_path)
"""

    new_save_optional = """def save_optional_artifacts(sino: np.ndarray, output_path: Path, args: argparse.Namespace) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_npy:
        npy_path = output_path.with_suffix(".npy")
        np.save(str(npy_path), sino)
        logging.info("Saved .npy: %s", npy_path)

    if args.save_preview:
        save_preview(sino, output_path)
"""

    if old_save_optional in text:
        text = text.replace(old_save_optional, new_save_optional)
    elif "def save_optional_artifacts" in text and "output_path.parent.mkdir(parents=True, exist_ok=True)" in text:
        print("save_optional_artifacts already patched.")
    else:
        raise RuntimeError("Could not patch save_optional_artifacts automatically.")

    old_save_preview = """def save_preview(sino: np.ndarray, output_path: Path) -> None:
    preview = output_path.with_suffix(".png")
    fig, ax = plt.subplots(figsize=(10, 6))
"""

    new_save_preview = """def save_preview(sino: np.ndarray, output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = output_path.with_suffix(".png")
    fig, ax = plt.subplots(figsize=(10, 6))
"""

    if old_save_preview in text:
        text = text.replace(old_save_preview, new_save_preview)
    elif "def save_preview" in text and "preview.parent.mkdir" in text:
        print("save_preview already patched.")
    elif "def save_preview" in text and "output_path.parent.mkdir(parents=True, exist_ok=True)" in text:
        print("save_preview already patched.")
    else:
        raise RuntimeError("Could not patch save_preview automatically.")

    if text == original:
        print("No changes needed. File already patched.")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")

    compile(text, str(path), "exec")
    print(f"Patched: {path}")
    print(f"Backup:  {backup}")


def main() -> None:
    patch_file(TARGET)


if __name__ == "__main__":
    main()
