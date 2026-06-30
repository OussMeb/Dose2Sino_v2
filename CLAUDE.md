# CLAUDE.md

## Project

Predict a **tomotherapy Leaf-Open-Time (LOT) sinogram** from patient anatomy/dose.

- **Input**: 3D "berlingo" tensor `[2, N_CP, H, W]` — channel 0 = CT, channel 1 = dose,
  each control-point slice is the CT/dose rotated by that gantry angle
  (`apply_tomo_transform_to_stack` in [utils/mask_utils.py](utils/mask_utils.py)).
- **Output**: 2D sinogram `[N_CP, 64]` — LOT per control point × 64 MLC leaves,
  continuous values in `[0, 1]`.
- **Core logic**: input and output are **spatially aligned**; the model reduces the
  3D volume to 2D by **collapsing one axis (the ray / beam-depth, `W`)**. Physically
  the sinogram is ~the **projection of the PTV**: a leaf opens where its ray passes
  through PTV. "Put 1s where there is PTV along the ray."

## Key shapes & config

- `N_CP` (control points) = the volume **depth** axis; preserved everywhere (Down/UpSampling
  use stride `(1,2,2)` — only H,W are downsampled).
- `H` = leaf axis → must reduce to **64** (fixed MLC leaf count).
- `W` = ray/beam axis → **collapsed to 1** (the projection).
- `REDUCTION_RATIO` downsamples H **and** W of the input. ratio=3 → ~170×170; ratio=8 → ~64×64.
  The model resamples in-plane axes to the canonical 64-leaf grid at its output stage,
  so any ratio works. Project uses **ratio=3** (more data).
- Entry point: [main_attention_in_agg_full.py](main_attention_in_agg_full.py);
  model in [models/unet_attention_in_agg.py](models/unet_attention_in_agg.py);
  trainer in [utils/trainer_supervised_logits.py](utils/trainer_supervised_logits.py).

### Cache gotcha
Cache files are keyed `{patient_id}_{pareto_index}.pt.gz` and **do NOT encode REDUCTION_RATIO**.
A cache built at one ratio is silently reused at another. `cache_sino` holds ratio-3 (~150 MB/file).
Regenerate with [generate_cache.py](generate_cache.py) into a ratio-specific dir if you change ratio.

## Loss — [utils/losses.py](utils/losses.py) `SinogramLoss`

Data is **~82% exact zeros**, the rest spread continuously over `(0,1]` — sparse continuous regression.
- **No BCE** (it forced binarization of continuous leaf values → over-painting/blur).
- **Charbonnier** on `sigmoid(logits)`, weighted to up-weight open leaves (`pos_weight`, recall).
- **Soft false-positive penalty** `fp_weight * mean(closed * p²)` — punishes opening a leaf that
  should be closed (precision), squared so it's gentle near 0. Additive (not a reweight of a mean),
  verified monotonic with a numerical unit test.
- Knobs: `pos_weight=8`, `fp_weight=4`.

## Sanity workflow — [sanity_overfit.py](sanity_overfit.py)

A correct model **must** overfit ONE sample to ~0 loss. Run before trusting any change:
```
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u sanity_overfit.py --patient <IPP> --steps 300 --lr 1e-3 \
  --pos-weight 8 --fp-weight 4 --cosine --viz-every 1 --viz-dir sanity_viz_test
```
Knobs: `--lr --steps --pos-weight --fp-weight --cosine/--no-cosine --min-lr --grad-clip
--base-filters --patient --viz-every`. Writes pred-vs-GT PNGs + `latest.pt` to `--viz-dir`.
Watch `open_l1` (error on open leaves), `closed_pred` (precision), `pred_mean` vs `target_mean`.

**ALWAYS** drop a `README.txt` (or `experiment.txt`) into the `--viz-dir` of every sanity run
explaining what that run is testing: the hypothesis, the exact config (arch/ratio/reduce_h/refine/
loss/base_filters/steps/patient), and what result would confirm or refute it. Past sanity dirs are
otherwise undocumented and impossible to interpret later (e.g. `sanity_viz_long` reached loss 0.005
/ open_l1 0.061 but its config was lost). Write the txt at launch.

## Architecture insight (why early versions could not overfit)

The reduction over the ray axis is a **MAX/OR** ("PTV anywhere along the ray → open"), but the
original head collapsed it with a **learned linear *sum*** (`Conv3d` kernel `(1,1,W)`) — the wrong
inductive bias, which a single sample could not overfit (width/LR didn't help → structural).

Grounded in DL-radiotherapy literature (Beam's-Eye-View → fluence maps; MIP 3D→2D projection nets):
- **#1 (current):** replace the linear ray-collapse with a **soft-max (LogSumExp) projection** along
  the ray axis. Right inductive bias for "open if any PTV along ray", differentiable, dense gradients.
- **#2 (next if needed):** **project-then-2D-refine** — 3D encoder → soft-max ray projection → **2D
  U-Net** on the `[N_CP, leaf]` plane.
- Refs: arXiv 2502.03360 (BEV→fluence), arXiv 1902.00347 (projection 2.5D U-Net),
  arXiv 2407.08655 (MIP-as-loss), PMC8099762 (fluence prediction).

## 2.5D rethink — current best model ([models/sinogram_2p5d.py](models/sinogram_2p5d.py))

The 3D V-Net family is hard to optimize here (deeper/wider/residual variants stall or
collapse; it overfits one sample only to `open_l1 ~0.11`). The task is intrinsically
per-control-point, so the **2.5D model processes each CP slice `[2,H,W]` with a shared 2D
CNN -> a 64-leaf row**, stacked over N_CP. Use `sanity_overfit.py --arch 2p5d`.
- **Trains reliably, ~15x faster (~2.3 s/step vs 33), fits memory with headroom to scale.**

### The leaf-readout finding (how the ~0.11 floor was broken)
A free `[N_CP,64]` tensor fits the target to `open_l1 ~0.0001` -> **data & loss are fine**;
the floor is the **leaf reduction**. Spatial pooling to 64 makes adjacent leaves *correlated*
(shared/overlapping windows) -> can't set 64 independent values -> identical ~0.11 floor on
EVERY architecture. The fix is a readout that is **both H-aligned (localization) AND per-leaf
independent**: `AdaptiveMaxPool((64,1))` then a **per-leaf linear** (`einsum('ncj,jc->nj')`
with a `[64, base]` weight). This broke the floor: **`open_l1` 0.11 -> 0.055**, diff ~all green.
- Dead ends for the readout: global Linear -> collapses to all-zero (82%-majority trap);
  64 conv filters + GLOBAL maxpool -> stalls (loses leaf localization).

## Hardware / ops notes

- GPU: **Quadro M6000 24 GB** (Maxwell, **no Tensor Cores** → AMP barely speeds up). One ratio-3
  run ≈ **19 GB** and ~**33 s/step** / ~1h40 per training epoch. Only **one** run fits per GPU;
  `pkill -f sanity_overfit.py` before relaunching. After kill, wait for VRAM to free before relaunch.
- Always run on `CUDA_VISIBLE_DEVICES=0` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Trainer uses `ReduceLROnPlateau(factor=0.9, patience=3)` but early-stop patience 5 → it barely
  decays before stopping. Cosine LR decay visibly polishes fine details in the sanity; consider
  switching the trainer to cosine or `factor=0.5` + higher early-stop patience.
