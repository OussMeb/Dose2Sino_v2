#!/usr/bin/env python3
"""
Confirm (or refute) that GT LOT sinograms have low-frequency structure along the
control-point (gantry-angle) axis. If a Fourier mixing layer along N_CP is going
to help, the angular spectrum must be concentrated at low frequencies (a strong,
compressible prior). If the spectrum is flat/white, Fourier buys nothing.

Loads all RP*.dcm RTPLANs under the data dir, extracts the [N_CP,64] sinogram
(private Tomo tag 300D,10A7), and measures, per leaf, how the angular power
spectrum concentrates: fraction of power in the lowest k frequencies, plus the
analogous numbers for a same-marginal phase-randomized surrogate (the null:
"same sparsity, no angular structure").
"""
from pathlib import Path
import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SINO_TAG = (0x300D, 0x10A7)
DATA = Path("/mnt/data/tomo_data")


def extract_sino(plan_path):
    ds = pydicom.dcmread(plan_path, force=True)
    try:
        beam = ds[(0x300A, 0x00B0)][0]
        cps = beam[(0x300A, 0x0111)].value
    except Exception:
        return None
    rows = []
    for cp in cps:
        if SINO_TAG not in cp:
            continue
        val = cp[SINO_TAG].value
        if isinstance(val, (bytes, bytearray)):
            parts = val.decode(errors="ignore").split("\\")
            r = [float(x) for x in parts if x.strip() != ""]
            if len(r) == 64:
                rows.append(r)
    if len(rows) < 16:
        return None
    return np.asarray(rows, dtype=np.float32)  # [N_CP, 64]


def angular_spectrum(sino):
    """Mean (over leaves) normalized power spectrum along the N_CP axis.
    DC removed (we care about the *varying* structure, not the mean level)."""
    s = sino - sino.mean(axis=0, keepdims=True)        # remove per-leaf DC
    F = np.fft.rfft(s, axis=0)                          # [F, 64]
    P = (np.abs(F) ** 2)                                # power per (freq, leaf)
    tot = P.sum(axis=0, keepdims=True)
    tot[tot == 0] = 1.0
    Pn = P / tot                                        # normalize per leaf
    return Pn.mean(axis=1)                              # [F]  mean over leaves


def lowfreq_fraction(spec, k):
    """Fraction of (non-DC) angular power in the lowest k frequency bins."""
    return spec[:k].sum() / spec.sum()


def phase_random_surrogate(sino, rng):
    """Same per-leaf marginal magnitude spectrum, randomized phase -> destroys
    cross-frequency/temporal structure while keeping the power distribution.
    This is NOT the right null for 'is power low-freq' (it preserves the spectrum).
    Instead we use a column-shuffle null: shuffle CP order per leaf -> whitens the
    angular spectrum but keeps the value distribution (sparsity) identical."""
    out = np.empty_like(sino)
    for j in range(sino.shape[1]):
        out[:, j] = sino[rng.permutation(sino.shape[0]), j]
    return out


def main():
    plans = sorted(DATA.glob("*/**/pareto_*/RP*.dcm"))
    print(f"found {len(plans)} RTPLAN files")
    specs, frac_lo, frac_lo_null, ncps = [], [], [], []
    K = 5  # "low freq" = first 5 non-DC bins
    rng = np.random.default_rng(0)
    used = 0
    for p in plans:
        sino = extract_sino(p)
        if sino is None:
            continue
        used += 1
        ncps.append(sino.shape[0])
        spec = angular_spectrum(sino)
        # resample spectrum to common length for averaging (freq axis varies w/ N_CP)
        specs.append(np.interp(np.linspace(0, 1, 64), np.linspace(0, 1, len(spec)), spec))
        frac_lo.append(lowfreq_fraction(spec, K))
        null = phase_random_surrogate(sino, rng)
        frac_lo_null.append(lowfreq_fraction(angular_spectrum(null), K))
    print(f"used {used} plans | N_CP range {min(ncps)}..{max(ncps)} median {int(np.median(ncps))}")
    frac_lo = np.array(frac_lo); frac_lo_null = np.array(frac_lo_null)
    print(f"\nLow-freq power (first {K} non-DC bins / total non-DC power):")
    print(f"  GT sinograms : mean {frac_lo.mean():.3f}  median {np.median(frac_lo):.3f}  "
          f"[{np.percentile(frac_lo,10):.3f}..{np.percentile(frac_lo,90):.3f}]")
    print(f"  shuffled null: mean {frac_lo_null.mean():.3f}  median {np.median(frac_lo_null):.3f}")
    print(f"  => ratio GT/null = {frac_lo.mean()/frac_lo_null.mean():.2f}x")

    # cumulative: how many low-freq bins to reach 90% of angular power
    mean_spec = np.array(specs).mean(axis=0)
    cum = np.cumsum(mean_spec) / mean_spec.sum()
    bins_90 = int(np.searchsorted(cum, 0.90)) + 1
    print(f"\nBins to reach 90% of angular power (resampled to 64): {bins_90}/64")
    print(f"Bins to reach 50%: {int(np.searchsorted(cum,0.50))+1}/64")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(mean_spec, lw=2)
    ax[0].set_title("Mean angular power spectrum (DC removed, per-leaf norm)")
    ax[0].set_xlabel("frequency bin along N_CP (0 = slowest)")
    ax[0].set_ylabel("normalized power"); ax[0].set_yscale("log"); ax[0].grid(alpha=.3)
    ax[1].hist(frac_lo, bins=20, alpha=.7, label="GT", color="C0")
    ax[1].hist(frac_lo_null, bins=20, alpha=.7, label="CP-shuffled null", color="C3")
    ax[1].set_title(f"Power fraction in first {K} bins"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    out = Path(__file__).parent / "fft_angular_structure.png"
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
