# Berlingo geometry-fix follow-up experiments

Methodical A/B on the geometry-fixed pipeline (symmetric 40cm crop + `z_iso - tables`).
Splits fixed (seed 42): train 810 / val 108 (6 patients) / test 144. Metric: mean val
`open_l1` (error on open leaves). Change ONE variable per run.

## Baseline (committed `cdf77f2`, run `20260620_121840_2p5d`)
- TARGET_HW=64, BASE_FILTERS=24, reduce_h=False, cosine 1e-3→1e-5/20ep, flip-aug on.
- Trajectory: e1 0.246 → e6 0.207 → e12 0.162 → **plateau ~0.158** (best val_loss 0.0868 @ e23).
- vs the OLD geometry-buggy wall: 0.31. → geometry fixes ≈ halved the error.
- train ≈ val (underfit, no overfit) → motivates lever 2 (capacity).

## Lever 2 — capacity
- **`exp_base48.py` ABANDONED**: base_filters=48 ran ~99 s/it (~22 h/epoch) on the
  M6000 — a cudnn pathological-algo / occupancy cliff (benchmark=False, Maxwell, 4×
  channels). ~2 weeks to plateau, untenable. Killed at epoch 1.
- **`exp_base32.py` (started 2026-06-22 23:16)**: BASE_FILTERS 24 → 32 (~735k params).
  Runs ~1.7 s/it / **~23 min/epoch** (5× faster than the baseline's logged 8.5 s/it —
  that baseline was data-bound on a cold NFS cache; warm cache is now compute-bound).
  Hypothesis: model underfits (sanity hits 0.03) → more capacity lowers the 0.158 floor.
- **Result: small real gain. base32 plateau ~0.153 vs baseline 0.158** (e12 0.155, e15
  0.153; best val_loss 0.0851 vs 0.0868). ~3% lower open_l1 — capacity helps a little
  but is NOT the main floor (consistent with data/inverse-problem limit). Killed at e16
  (converged, flat 0.153-0.155 since e12).

## Lever 1 — input resolution (`exp_hw128.py`, cache regen started 2026-06-22 14:32)
- Change: TARGET_HW 64 → 128 (reduce_h=True keeps 1:1 leaf readout). New cache_sino_hw128.
- Hypothesis: floor is partly lossy low-res dose input → finer berlingo lowers it.
- Compute turned out tractable (warm cache is compute-bound): ~45 min/epoch expected.
- Launched 2026-06-23 after lever 2.
- Result: _pending_.

### Lever 1 interim (epoch 8)
hw128 tracks ~0.02 ABOVE baseline at every epoch (e6 0.226 vs 0.207, e8 0.212 vs
base24 0.191 / base32 0.179). Higher input resolution is NOT helping — consistent
with the dose-space finding (floor = degeneracy, not lost input info). Confound:
hw128 uses reduce_h=True (different readout regime). Verdict pending e12.

### Dose-space eval (surrogate g, base16)
g VAL dose MAE 1.5% of peak; g IS sinogram-pattern-driven (zero/shuffle sinogram ->
8% dose change) BUT amplitude-INSENSITIVE (0.5xLOT -> only 0.66% vs expected ~50%).
=> the "0.52% dose impact of the predicted sinogram" is reliable for pattern errors
but under-reports amplitude errors. Retraining g with amplitude augmentation
(surrogate_dose_ampaug.pth) to make the dose-space verdict trustworthy.

### Lever 1 FINAL (resolution) — NEGATIVE
hw128 e12 = 0.202 vs base24 0.162 / base32 0.155 — consistently ~0.04-0.05 WORSE at
every epoch. Doubling input resolution does not help (confound: reduce_h=True regime).
Killed at e12. Confirms the floor is NOT lost input information.

### Surrogate dose-eval — UNTRUSTWORTHY (abandoned as in-silico proxy)
Amplitude augmentation a~U[0,2] (incl. zero-sinogram->zero-dose cases) did NOT fix it:
g(ZERO) 11.5% (was 12%), g(0.5xLOT) 6.1% (was 7%). Root cause: (1) InstanceNorm3d
normalizes away input amplitude -> architecturally scale-invariant; (2) CT-shortcut:
plans conform to anatomy so CT->dose is ~deterministic and g free-rides on the CT,
barely using the sinogram. => the in-silico dose-space verdict cannot be trusted as
built. Definitive dose validation needs the real TPS (export RTPlan -> recompute),
offline. A norm-free surrogate MIGHT fix amplitude but the CT identifiability remains.

## STATUS / next levers
- Geometry fixes: 0.31 -> 0.153 (the big win, banked).
- Capacity (base32): 0.158 -> 0.153 (marginal).
- Resolution (hw128): negative.
- In-silico dose proxy: not trustworthy here.
=> Model-side in-silico levers are largely exhausted. Decision point is OUTSIDE this
repo: export the predicted plan and validate DOSE on the Accuray TPS. Generative /
forward-consistency only makes sense once we can measure dose faithfully.

## Lever 3 — Fourier coupling along the control-point (gantry) axis

Motivation: the 2.5D model predicts every control point INDEPENDENTLY; the only
cross-CP coupling is the conv `refine_head`, which sees only +/-1-2 CP. But N_CP ~1300
and a structure traces a slow sinusoid across that whole axis (it is a *sino*gram).

### Structure check (measured, 1062 RTPLANs) — POSITIVE
FFT of GT sinograms along N_CP (DC removed, per-leaf normalized) vs a CP-shuffled
null (same sparsity, angular order destroyed):
- low-freq power (lowest 5 non-DC bins): GT 6.4% vs null 0.6% -> **10x**, the two
  distributions do not even overlap. ~50% of angular power in the lowest ~9% of bins,
  ~90% in the lowest ~40% -> low-freq-dominant but with a real structured HF tail
  (the sparse on/off leaf transitions). => global low-freq angular structure is real;
  a Fourier mixing op along N_CP is justified, but keep the real-domain residual path
  (a pure low-pass would lose the tail / ring on the sparse zeros).
  Script: `analyze_angular_fft.py`; fig `fft_angular_structure.png`.

### Implementation
- `models/sinogram_2p5d.py::SpectralRefine1D` — a 1D FNO layer on the [N_CP,64] output
  plane: lift 64->hidden, rfft along N_CP, learned complex mix on the lowest `fno_modes`
  freqs, irfft, + real 1x1 path, zero-init output proj (identity at init). Gated by
  `refine_mode in {conv, fno, both}` (+ `fno_modes`).
- `utils/losses.py::AngularSpectralLoss` — additive aux term matching the low-freq
  angular magnitude spectrum (skip DC, lowest `modes` freqs, length-normalized).
- M6000 (SM_52) gotcha: cuFFT has NO half support -> both modules force fp32 inside an
  `autocast(enabled=False)` block; einsum has no complex support on this torch build ->
  the spectral channel-mix is done via real/imag einsums. Unit tests:
  `test_fourier.py` (identity-at-init, global reach, grads, end-to-end all modes).
- Flags wired into `sanity_overfit.py`: --refine-mode --fno-modes --spectral-weight
  --spectral-modes.

### Sanity (single sample 321322, r3 reduce_h=True, 150 steps, matched budget)
NOTE: overfit rewards LOCAL memorization, so it is the wrong test for a global low-freq
REGULARIZER (fno) — informative only as a "does it fit / is it stable" gate.
- conv (local)          : mae 0.0427  open_l1 0.288  pred_mean 0.049
- fno  (global)         : mae 0.0537  open_l1 0.414  pred_mean 0.030  (fits slower, as
  expected for a regularizer; monotone, no block)
- both + spectral(w0.5) : mae 0.0325  open_l1 0.239  pred_mean 0.0616 (= target 0.0608)
  => conv handles local detail, fno+spectral add angular structure and calibrate the
  mean almost exactly. Stable, no NaN.

### Full train A/B (exp_fourier.py: base32 + refine_mode=both + spectral w0.5) — NEGATIVE
Ran 54 epochs (~22 h; never early-stopped because CosineAnnealingLR T_max=20 RESTARTS
-> LR oscillates 1e-3<->1e-5 every 20 ep and each LR change resets the early-stop
counter -> AUTO_STOP neutralized).
- best val open_l1 = **0.162 @ e51** vs base32 baseline plateau **0.153** -> ~6% WORSE,
  never caught up, then overfit (train loss 0.065 vs val 0.090). (NB: its val_loss
  includes the spectral term so it is NOT comparable to the baseline's 0.0851.)
- => the Fourier coupling does NOT help open_l1. Not surprising given the degeneracy
  finding below: the baseline 0.153 is ALREADY at the cross-plan floor, so no
  architecture lever can go meaningfully lower; the extra params + spectral term only
  divert from raw open_l1. Killed.

### THE DECISIVE CONTEXT — degeneracy floor (anatomy -> sinogram is one-to-many)
Measured on 59 patients with >=2 Pareto plans (same CT, different optimizer tradeoff),
9027 plan pairs: **cross-plan open_l1 = mean 0.150 / median 0.121**. The model floor is
**0.153 == the intrinsic plan-to-plan spread**. So the model already predicts a sinogram
as well as one valid plan predicts another for the same anatomy: open_l1 vs a SINGLE
chosen plan is a saturated metric, and ~0.15 is an irreducible (aleatoric / optimizer
null-space) ceiling that more same-distribution data CANNOT lower. Script:
`analyze_pareto_degeneracy.py`.
=> Stop tuning architecture against open_l1 (Fourier, capacity, resolution all chase a
saturated metric). Real levers: (1) richer CONDITIONING (Pareto index / objective
weights / modulation factor / pitch / prescription — already in the DICOM; keep all
Pareto plans, do NOT drop to one-per-patient = data starvation with only ~6 train
patients), and (2) a DOSE-EQUIVALENT target via a PHYSICS-linear differentiable dose
operator (D@fluence, exportable Dij from RayStation), NOT the learned CNN surrogate
(CT-shortcut + InstanceNorm amplitude-invariance already made it untrustworthy).
DIBH/FB: keep both (real anatomical diversity, not degeneracy); split is leakage-safe
(get_patient_based_splits strips _DIBH). Fourier code (SpectralRefine1D /
AngularSpectralLoss / refine_mode) is left in place, off by default.

## Lever B prototype — differentiable simplified dose operator (the dose-consistency path)

Scalar conditioning (pitch/field-width) is DEAD: confirmed all Pareto plans of a patient
share the SAME gantry angles + couch positions (only LOT differs). The per-plan signal is
the DOSE, already an input channel — so the task is "dose(+CT) -> sino", learn the
optimizer inverse. The lever is a PHYSICS forward dose model used as a loss (never the
learned CNN surrogate: CT-shortcut + amplitude-invariance killed it).

### Built it ourselves (no RayStation Dij needed) — the berlingo makes it cheap
The berlingo already gives CT rotated to each gantry angle -> ray axis W IS beam depth.
- `prototype_dose_forward.py::dose_forward`: BEV primary dose TERMA[i,h,w] =
  LOT[i,h]·mu·exp(-cumsum_w mu), mu from normalized-HU electron density. Differentiable,
  LINEAR in sino. Optional Gaussian scatter (scatter_h/scatter_w) = CCC 'lite'.
- `prototype_dose_accumulate.py::accumulate`: un-rotate each CP's BEV dose by -(90-theta)
  (differentiable grid_sample) and index_add into the z-bin of z_iso-table -> 3D dose.

### Validation (patient 183040)
- **Amplitude: PERFECT** (linear): 0.5x LOT -> 0.500 dose change, 2x -> 1.000. vs the
  learned surrogate's 0.66% (amplitude-blind). The surrogate's fatal flaw is gone by
  construction. Gradient finite, dosimetrically steered.
- **Accumulation -> real dose**: corr per-CP primary 0.47 -> accumulated **0.873** (best
  scatter_h=2 -> 0.882). Hotspot co-located with the real planned dose (viz
  `prototype_dose_accumulate.png`). Scatter barely helps -> the ~0.12 residual is deeper
  crudeness (no real depth kernel, crude mu(HU), nearest-z binning, no field-width z-
  spread), NOT lateral scatter. ~0.88 is the ceiling of this toy; >0.9 needs real CCC.
  0.88 is good enough to PROTOTYPE the loss.

### The cross-Pareto "degeneracy relief" test — DID NOT pan out (honest)
Two real Pareto plans (shared geometry): open_l1(sino) 0.107 vs accumulated dose-rel
**0.171** — dose varies as much / more, NOT << as hoped. Why: (1) metrics not comparable
(masked-MAE vs relative-L2); (2) **different Pareto plans are genuinely dose-DIFFERENT**
(different tradeoffs) -> cross-Pareto is real dose variation, not pure null-space; (3)
primary-only dose amplifies differences scatter would smooth.
=> REFRAME the lever's mechanism: the dose loss is best understood NOT as "erase the
degeneracy" but as **"force the predicted sino to reproduce the INPUT dose"** (anti
CT-shortcut; make the model actually USE the dose channel it currently underuses). Part
of the 0.15 floor is real dose variation that IS in principle predictable from the dose
input. Pure null-space (same dose, different sino) remains unprobed (needs SVD of the
operator or repeated TPS re-optimizations).

### Wired + calibrated (sanity, patient 183040)
`utils/dose_operator.py` (dose_forward + accumulate + geometry) and
`utils/losses.py::DoseConsistencyLoss` (scale-invariant normalized-dose MSE + sinogram
anchor); flags in `sanity_overfit.py --loss dose`.
- Single-sample overfit WORKS: L_dose 0.78 -> 0.019 (driven BELOW GT's 0.188 by finding a
  dose-equivalent sino). open_l1 stayed ~0.29 -> the loss admits dose-equivalent solutions
  far from GT, as intended. Predicted sino looks banded/physical, not noise.
- The model exploits the operator null-space if under-anchored. sino_weight sweep
  (dose_weight=1, 100 steps): sw 0.1 -> open_l1 0.29 / L_dose 0.019; 0.5 -> 0.22/0.036;
  1.0 -> 0.18/0.053; **2.0 -> 0.144/0.077**; 4.0 -> 0.13/0.11. **sino_weight=2.0 is the
  operating point**: open_l1 ~0.15 (as close to GT as another valid plan) while L_dose
  stays ~2.4x below GT. (Single-sample calibration; reconfirm in full training.)

### Operator fidelity improved by PHYSICS (not blur): z field-width
The per-CP single-slice accumulation omitted the jaw FIELD WIDTH in z. Adding a Gaussian
z-spread (field_z, ~real 2.5cm jaw) + light lateral scatter (scatter_xy): corr vs real
dose **0.873 -> 0.94** (183040). This is the CORRECT null-space (real field width), not
cosmetic smoothing. Baked into `accumulate(field_z, scatter_xy)` / DoseConsistencyLoss
(defaults field_z=2.5, scatter_xy=1.0). Script: `analyze_dose_operator_fidelity.py`.
- **Fidelity distribution over 25 acquisitions: median 0.92, mean 0.88, 88% >= 0.90.**
- Low tail (3): 187591 0.55, 223696_DIBH 0.63, 229221_DIBH 0.50 — a per-ACQUISITION
  geometry issue, NOT an operator limit: FB vs DIBH of the same patient diverge wildly
  (187591 0.55 vs 187591_DIBH 0.94; 229221 0.90 vs _DIBH 0.50). Smells like residual
  [[berlingo-z-slice-couch-bug]] on those couch tracks. Identify/fix or down-weight before
  a full run.
- Sanity re-run at sino_weight=2.0 with the improved operator confirms a SMALLER, more
  physical null-space: at the same anchor the model lands CLOSER to GT (open_l1
  0.144 -> 0.088 @ matched steps) with better dose match (L_dose 0.077 -> 0.055); pred-vs-GT
  diff is now nearly all green (sanity_viz_dose_improved). The fidelity gain translated
  directly into a better-behaved loss.

### Next
Trainer plumbing (surface angles/tables per sample) + exp_dose.py + full run at
dose_weight=1/sino_weight=2.0; report a DOSE metric (not open_l1). Optionally fix the
low-fidelity acquisitions first.

## DoseCUDA — the REAL collapsed-cone engine as the gold validation judge
User's own CCC engine (git@github.com:ArthurRochette/DoseCUDA, cloned to /mnt/data/DoseCUDA):
real Tomo 6MV FFF collapsed-cone with an EGSnrc-Monte-Carlo dose kernel. Built + installed
into the project venv (`uv pip install /mnt/data/DoseCUDA`; M6000 sm_52 via CMAKE native).
**Side effect: numpy pinned to 1.26.4** (DoseCUDA requires numpy<2) -- project still imports
fine and np.ndarray.ptp works again.
- API: `TomoPlan.readPlanDicom(plan)` reads the LOT sinogram (tag 300D,10A7, floats in [0,1]);
  `TomoDoseGrid.computeTomoPlan(plan)` -> 3D dose [Gy]. Input = our exact sinogram format.
- Verified on patient 183040 GT plan: realistic dose (max 4.48 Gy, mean(>0) 0.70 Gy for a
  15-fx breast plan). Script `validate_dose_ccc.py` (inject predicted sinogram into a copy of
  the plan DICOM -> CCC -> compare to GT CCC). RTStruct in this build has only getCentroid/
  getStructureNames (no getBoundingBox) -> set the dose ROI manually around the isocenter.
- **CRITICAL CONSTRAINT: ~14 min/plan on the M6000** (826 s, 2.5 mm grid). Implications:
  CCC in the training loss is INFEASIBLE (14 min x thousands of steps); brute-force Dij
  precompute is INFEASIBLE (~28k beamlets/patient). It IS usable as an OFFLINE validation
  judge for a few cases.
- **Strategy (settled): cheap ray-tracer (0.91) = differentiable training GRADIENT; DoseCUDA
  = exact validation JUDGE.** The gradient only needs to point roughly right; CCC judges the
  destination. Self-correcting: if the model exploits ray-tracer null-space that CCC
  penalises, the CCC validation catches it -> only then upgrade the loss (e.g. precomputed
  Dij on a faster GPU). cuve-a-eau clean-floor test available (tests/test_tomo_cuve.py).

## Lever B full-train PROBE (exp_dose.py, base32 + DoseConsistencyLoss) — NEGATIVE / loss too insensitive
Amplitude term added (l_amp = per-projection mean-LOT match) — FIXES the +8% drift: sanity
pred_mean 0.124 -> 0.115 (= GT) while L_dose stays low. Trainer plumbed (`_compute_loss` +
`_dose_geometry`: per-sample angles/tables, cached, low-fidelity acquisitions excluded via
config.DOSE_EXCLUDE, sino-only fallback). Probe at dose_w=1/sino_w=2/amp_w=5:
- val open_l1 DESCENDS but ~0.02 BEHIND baseline (e8 0.198 vs base32 0.179) — the dose+amp
  terms slow the sinogram learning without breaking it.
- **VAL L_dose FLAT / slightly up** (0.0598 e1 -> 0.0643 e8): the dose loss provides NO useful
  generalization gradient. It hits its floor at epoch 1 (already BELOW GT's operator L_dose
  ~0.12-0.16 -> the model instantly finds operator-exploiting solutions) and stops moving.
- Root cause: the scale-invariant COSINE dose loss is **insensitive** — the large smooth dose
  blob dominates the cosine, so any reasonable prediction scores ~0.06. At the realistic
  operating point (sino_w=2) the loss is 70% sinogram-dominated -> effectively a (slightly
  worse) sino run; lowering sino_w lets the dose term dominate but then open_l1 blows up to
  ~0.29 (operator exploitation). No balance drives dose-correctness AND stays realistic.
=> The CONCEPT (dose-equivalent objective) is fine and CCC-validated on the overfit (0.995),
but this LOSS FORMULATION is the bottleneck. Killed at e8.
- Masking to the high-dose region was tried (dose_mask_frac) and REFUTED: a noise sweep shows
  the masked L_dose DECREASES as the sinogram degrades (anti-sensitive), while the UNMASKED
  L_dose rises monotonically (0.096->0.312, it IS sensitive). Mask reverted to off (0.0).
- Reframe: the flat VAL L_dose is not insensitivity but OPERATOR EXPLOITATION (the model drives
  L_dose 0.058 BELOW GT's 0.10-0.14 by exploiting the operator's 8% error, hits that floor at
  epoch 1).

## THE CLARIFYING RESULT — a generalized model is ALREADY dosimetrically correct
CCC-validated the dose-trained model (base32, epoch 7, val open_l1 ~0.20) on UNSEEN val patient
183040 (DoseCUDA real CCC vs the GT plan's CCC dose):
- **corr in dose-region 0.991** (full 0.995); mean region dose pred 1.770 vs gt 1.755 Gy
  (+0.9%, identical); max +10% (slight hotspot); mean |Δ| 7.4%. Amplitude well-calibrated
  (pred mean 0.116 vs 0.1145, the amp term held in generalization). Fig ccc_valmodel_vs_gt.png.
=> **The ~0.153/0.20 open_l1 "floor" is a METRIC RED HERRING, not a dosimetric deficiency.** A
realistically-generalized model delivers the right real dose (0.991 on an unseen patient, ~as
good as the overfit's 0.995). This reconciles everything: degenerate plans (cross-Pareto 0.15)
ALL give the right dose, so a model at the degeneracy floor IS dose-correct; the negative levers
(Fourier, capacity, dose loss) never beat open_l1 because open_l1 was already at a dosimetrically
fine point -- we were optimizing the wrong metric. The dose loss had no dosimetric headroom to
gain (baseline ~0.99 already); its real value was building the CCC JUDGE that proved the model
was already good. PRAGMATIC PATH: ship the simpler sinogram baseline, validate by offline CCC.
CONFIRMED on 3 UNSEEN val patients (validate_dose_ccc_general.py): 183040 corr-region 0.991
(mean +0.9%), 229221 0.986 (+1.6%), 297768 0.987 (+3.1%). Solid + reproducible. The ONE
consistent imperfection is a **+9-10% max hotspot** across all three (systematic slight peak
over-prediction -> a real, minor refinement target; mean dose & spatial distribution excellent).
this shows the model is dose-correct, not that dose-loss > baseline (both ~0.99).
Scripts: validate_dose_ccc_valmodel.py / validate_dose_ccc_general.py. RACE-CONDITION LESSON:
a stale full-grid CCC run clobbered ccc_gt_183040.npy (shared filename) -> distinct names/run.

## Lever C — Dmax / hotspot penalty (2026-06-29)

### Motivation
The +9-10% Dmax hotspot is systematic across all 3 val patients. Post-hoc sharpening
makes it WORSE (+11%); mild spatial blur helps slightly (+8%) at correlation cost.
A training-time one-sided Dmax penalty was the clean fix.

### CCC kernel for differentiable Dmax (ccc_kernel_test.py)
Real Tomo 6MV FFF collapsed-cone kernel from DoseCUDA/lookuptables/photons/Tomo/6MV/kernel.csv:
6 collapsed cones at theta=[1.875, 20.625, 43.125, 61.875, 88.125, 106.875] deg, each with
double-exponential Am*exp(-am*r)+Bm*exp(-bm*r) (primary A-term sub-cm; scatter B-term long-
range 8-16cm). Angle-weighted isotropic average -> build_ccc_kernel(dz, dxy) in dose_operator.py.
**corr(KERNEL Dmax, CCC Dmax) = 0.893 vs 0.605 for old Gaussian.** Monotone: more blur = lower
Dmax (correct direction). Kernel [1,1,21,17,17] at dz=0.5cm/dxy=0.625cm/rmax=5cm;
vectorised numpy build 0.1s (cached); conv3d forward 0.09s CPU, negligible GPU overhead.

### DoseConsistencyLoss — dmax_weight added (utils/losses.py)
One-sided hinge: l_dmax = ((dmax_pred - dmax_gt) / dmax_gt).clamp(min=0)
dmax_pred = apply_ccc_kernel(accumulate(dose_forward(pred_sino, ct))).max()
dmax_gt   = apply_ccc_kernel(accumulate(dose_forward(target_sino, ct))).max()  [no_grad]
Sanity probe (183040, 100 steps, dmax_weight=1.0): l_dmax 0.71 -> 0.00 by step 10,
briefly 0.046 at step 50 (shifted dose), then 0. Grad flows, backward OK, no NaN.
Wired in trainer (Ldmax log column) and sanity_overfit.py (--dmax-weight).

### Fine-tune run (exp_dmax.py, 2026-06-29 16:20 -> 2026-06-30 09:21, 37 epochs)
Fine-tuned dose checkpoint (20260626_200610_2p5d_dose) with dmax_weight=1.0, lr=2e-4,
cosine 15ep. LR/early-stop interaction bug (ReduceLROnPlateau resets counter on each LR
change; cosine changes LR every epoch -> early-stop neutralized, same as exp_fourier).
Val loss: 0.387 -> 0.322 over 37 epochs. BEST checkpoint: epoch 36, val 0.322405
(20260629_161955_2p5d_dmax/best_model_new_session_session_0_.pth).
Ldmax fires on multiple train patients per epoch (0.007-0.09 range), confirming the
penalty targets real hotspot-candidates. Killed at e38 (plateau since e31).
### CCC validation — epoch 36 checkpoint vs dose baseline (patient 183040)
validate_dose_ccc_general.py with --ckpt on both checkpoints, same GT CCC run:

| metric              | dose baseline | dmax e36   | delta          |
|---------------------|---------------|------------|----------------|
| corr region         | 0.9910        | **0.9960** | +0.5%          |
| mean region pred    | 1.770 Gy +0.9%| 1.744 −0.6%| better calib   |
| **maxhot ratio**    | **+10.4%**    | **+10.1%** | **≈ unchanged**|
| mean |Δ| dose       | 7.4%          | **4.4%**   | −3% ✅         |
| pred mean sino      | 0.1163        | 0.1122     |                |

VERDICT: overall dose quality improved materially (corr +0.5%, mean error −3%), but the
+10% Dmax hotspot is UNCHANGED (10.4% -> 10.1%, within CCC noise). The scalar dmax_weight=1.0
hinge was too weak to restructure the spatial sinogram pattern — the model minimized the loss
by slightly reducing the global mean (0.1163->0.1122) rather than opening peripheral leaves.
The hotspot is structural (regression-to-the-mean -> dose convergence at isocenter), not
a scalar amplitude problem.

OPTIONS: (a) dmax_weight >>1 (5-10), (b) Dmax/Dmean ratio penalty (concentration-specific),
(c) accept the +10% as clinically within D2% tolerance (mean dose + corr both excellent).
The general dose improvement (7.4%->4.4% error, 0.991->0.996 corr) is a real gain from the
fine-tuning, likely from the dmax kernel providing additional spatial dose gradient signal
even when the max penalty itself is satisfied.
