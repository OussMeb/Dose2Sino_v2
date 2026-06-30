import torch
import torch.nn as nn

from utils.dose_operator import dose_forward, accumulate, build_zbin, build_ccc_kernel, apply_ccc_kernel


class CharbonnierLoss(nn.Module):
	"""Robust L1-like loss."""

	def __init__(self, eps: float = 1e-3):
		super().__init__()
		self.eps2 = eps ** 2

	def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class SinogramLoss(nn.Module):
	"""
	Regression loss for sparse MLC sinograms.

	- No BCE (avoids binarization pressure)
	- Charbonnier on sigmoid output
	- Soft penalty for unwanted openings
	"""

	def __init__(
		self,
		eps: float = 1e-3,
		pos_weight: float = 8.0,
		fp_weight: float = 4.0,
		open_threshold: float = 1e-3,
	):
		super().__init__()
		self.eps2 = eps ** 2
		self.pos_weight = pos_weight
		self.fp_weight = fp_weight
		self.open_threshold = open_threshold

	def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		p = torch.sigmoid(pred)

		open_mask = target > self.open_threshold
		closed = (~open_mask).float()

		weight = torch.where(
			open_mask,
			1.0 + (self.pos_weight - 1.0) * target,
			torch.ones_like(target),
		)

		denom = weight.sum().clamp_min(1.0)

		# Main regression term (robust, non-binary)
		char = torch.sqrt((p - target) ** 2 + self.eps2)
		match = (weight * char).sum() / denom

		# Soft false-positive penalty (avoids hard binarization)
		fp = (closed * p.pow(2)).sum() / closed.sum().clamp_min(1.0)

		return match + self.fp_weight * fp


class AngularSpectralLoss(nn.Module):
    """Match the low-frequency angular (control-point axis) magnitude spectrum of
    sigmoid(pred) to the target.

    Auxiliary, additive term. GT LOT sinograms provably carry low-frequency
    structure along the gantry-angle axis (~10x more low-freq power than a
    CP-shuffled null). This term encourages the model to reproduce that
    slowly-varying angular content directly, WITHOUT forcing structure in the
    sparse real domain (where matching a dense Fourier target would create
    ringing / false positives in the 82%-closed background).

    Only the lowest `modes` non-DC frequencies are matched (the structured band);
    the sparse high-frequency tail is left to the real-domain SinogramLoss. DC is
    skipped so this term is about the *variation* along angle, not the mean level
    (already handled by the regression term). Magnitudes are length-normalized so
    the weight is stable across patients with different N_CP.
    """

    def __init__(self, modes: int = 32, weight: float = 1.0):
        super().__init__()
        self.modes = int(modes)
        self.weight = float(weight)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred_logits, target: [B, N_CP, 64]  (N_CP on dim=1)
        p = torch.sigmoid(pred_logits)
        n = p.shape[1]
        # cuFFT has no half support on pre-SM_53 GPUs (the M6000 is SM_52); run the
        # FFT in float32 even under AMP autocast.
        with torch.autocast(device_type=p.device.type, enabled=False):
            p = p.float()
            tgt = target.float()
            pf = torch.fft.rfft(p - p.mean(dim=1, keepdim=True), dim=1)
            tf = torch.fft.rfft(tgt - tgt.mean(dim=1, keepdim=True), dim=1)
        m = min(self.modes, pf.shape[1] - 1)          # skip DC (index 0)
        # eps-safe complex magnitude: |z|=sqrt(re^2+im^2) is non-differentiable at
        # 0 (NaN grad when a mode is exactly flat); the eps keeps it smooth.
        eps = 1e-12
        dp = (pf[:, 1:1 + m].real ** 2 + pf[:, 1:1 + m].imag ** 2 + eps).sqrt() / n
        dt = (tf[:, 1:1 + m].real ** 2 + tf[:, 1:1 + m].imag ** 2 + eps).sqrt() / n
        return self.weight * (dp - dt).abs().mean()


class DoseConsistencyLoss(nn.Module):
    """Lever B: force the predicted sinogram to reproduce the planned DOSE, via a
    self-built differentiable dose operator (utils/dose_operator), + a small sinogram
    anchor. Rationale (EXPERIMENTS "Lever B"): the open_l1-vs-one-plan metric is
    saturated at the ~0.15 anatomy->sinogram degeneracy; the model also underuses the
    dose input (CT-shortcut). Matching the accumulated dose makes the model USE the
    dose channel and admits dose-equivalent solutions.

    Dose term is SCALE-INVARIANT (the operator's absolute gain mu_water is arbitrary):
    compares L2-normalized 3D doses == 2*(1 - cosine), so it constrains the dose SHAPE
    / relative amplitude, not the arbitrary global gain.

    forward(pred_logits, target_sino, ct, real_dose_bev, alpha_deg, tables):
      pred_logits/target_sino [N,64]; ct/real_dose_bev [N,H,W]; alpha_deg=(90-gantry)
      [N]; tables [N]. (B=1: squeeze before calling.)
    """

    def __init__(self, n_z: int = 48, dose_weight: float = 1.0, sino_weight: float = 0.1,
                 amp_weight: float = 1.0, dmax_weight: float = 0.0,
                 mu_water: float = 0.03, scatter_h: float = 0.0, sign: float = 1.0,
                 field_z: float = 2.5, scatter_xy: float = 1.0, dose_mask_frac: float = 0.0,
                 pos_weight: float = 8.0, fp_weight: float = 4.0,
                 dz_cm: float = 0.5, dxy_cm: float = 0.625, dmax_rmax_cm: float = 5.0):
        super().__init__()
        self.n_z = int(n_z)
        self.dose_weight = float(dose_weight)
        self.sino_weight = float(sino_weight)
        # amp_weight constrains ABSOLUTE amplitude (MU): the dose term is scale-
        # invariant (normalized) so it lets the global level drift (+8% in the CCC
        # validation). Matching the per-projection mean leaf-open-time pins the MU.
        self.amp_weight = float(amp_weight)
        self.mu_water = float(mu_water)
        self.scatter_h = float(scatter_h)
        self.sign = float(sign)
        # field_z/scatter_xy = jaw field width + lateral scatter on the PRIMARY side
        # (operator fidelity 0.88 -> ~0.94; the missing real-geometry factor).
        self.field_z = float(field_z)
        self.scatter_xy = float(scatter_xy)
        # dose_mask_frac: restrict the dose loss to voxels with real dose >
        # frac*max (target + penumbra). The global cosine is dominated by the large
        # smooth low-dose bath -> insensitive (probe: VAL L_dose flat ~0.06). Masking
        # to where the dose matters makes the gradient discriminating.
        self.dose_mask_frac = float(dose_mask_frac)
        self.dmax_weight = float(dmax_weight)
        # Pre-build the CCC kernel (lazy: only when dmax_weight > 0).
        # Cached in dose_operator._kernel3d_cache so rebuilds across instances are free.
        self._dz_cm = float(dz_cm)
        self._dxy_cm = float(dxy_cm)
        self._dmax_rmax_cm = float(dmax_rmax_cm)
        self._ccc_kernel = None  # built on first forward with the right device
        if dmax_weight > 0:
            # Try to pre-build on CPU so first forward has no build delay.
            k = build_ccc_kernel(dz_cm, dxy_cm, dmax_rmax_cm)
            if k is None:
                import warnings
                warnings.warn("dmax_weight>0 but kernel CSV not found; Dmax term disabled")
                self.dmax_weight = 0.0
            else:
                self._ccc_kernel = k          # cpu tensor; moved to device on first use
        self.sino = SinogramLoss(pos_weight=pos_weight, fp_weight=fp_weight)

    @staticmethod
    def _ndose(v: torch.Tensor) -> torch.Tensor:
        return v / (v.norm() + 1e-8)

    def forward(self, pred_logits, target_sino, ct, real_dose_bev, alpha_deg, tables):
        p = torch.sigmoid(pred_logits)
        zbin = build_zbin(tables.detach().cpu().numpy(), self.n_z, device=p.device)
        # run the dose operator in fp32 (grid_sample/cumsum + kernel conv stability under AMP)
        with torch.autocast(device_type=p.device.type, enabled=False):
            pf, ctf, rdf, af = p.float(), ct.float(), real_dose_bev.float(), alpha_deg.float()
            bev_pred = dose_forward(pf, ctf, self.mu_water, scatter_h=self.scatter_h)
            dose_pred = accumulate(bev_pred, af, zbin, self.n_z, self.sign, reduce="sum",
                                   field_z=self.field_z, scatter_xy=self.scatter_xy)
            dose_real = accumulate(rdf, af, zbin, self.n_z, self.sign, reduce="mean")
            if self.dose_mask_frac > 0:
                mask = dose_real > self.dose_mask_frac * dose_real.max()
                dp, dr = dose_pred[mask], dose_real[mask]
            else:
                dp, dr = dose_pred, dose_real
            l_dose = ((self._ndose(dp) - self._ndose(dr)) ** 2).sum()

            # Dmax penalty: one-sided hinge — penalise when pred Dmax exceeds GT Dmax.
            # Uses the real Tomo CCC scatter kernel (double-exponential, corr=0.89 vs CCC).
            if self.dmax_weight > 0 and self._ccc_kernel is not None:
                kernel = self._ccc_kernel.to(p.device)
                dmax_pred = apply_ccc_kernel(dose_pred, kernel).max()
                with torch.no_grad():
                    bev_gt = dose_forward(target_sino.float(), ctf, self.mu_water)
                    acc_gt = accumulate(bev_gt, af, zbin, self.n_z, self.sign, reduce="sum",
                                       field_z=self.field_z, scatter_xy=self.scatter_xy)
                    dmax_gt = apply_ccc_kernel(acc_gt, kernel).max()
                l_dmax = ((dmax_pred - dmax_gt) / (dmax_gt + 1e-8)).clamp(min=0)
            else:
                l_dmax = torch.zeros(1, device=p.device)

        l_sino = self.sino(pred_logits, target_sino)
        l_amp = (p.mean(dim=-1) - target_sino.mean(dim=-1)).abs().mean()
        total = (self.dose_weight * l_dose + self.sino_weight * l_sino
                 + self.amp_weight * l_amp + self.dmax_weight * l_dmax)
        return total, l_dose.detach(), l_sino.detach(), l_amp.detach(), l_dmax.detach()


class SinogramSpectralLoss(nn.Module):
    """SinogramLoss (real-domain regression) + AngularSpectralLoss (low-freq angular
    spectrum match) as one additive loss with the trainer's (logits, target)
    signature. Lever 3: the real term keeps the sparse output honest; the spectral
    term injects the global gantry-angle structure the per-CP encoder cannot see."""

    def __init__(self, eps: float = 1e-3, pos_weight: float = 8.0, fp_weight: float = 4.0,
                 spectral_weight: float = 0.5, spectral_modes: int = 32):
        super().__init__()
        self.base = SinogramLoss(eps=eps, pos_weight=pos_weight, fp_weight=fp_weight)
        self.spectral = AngularSpectralLoss(modes=spectral_modes, weight=spectral_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.base(pred, target) + self.spectral(pred, target)


class SinogramL1RuleLoss(nn.Module):
	"""
	L1 sinogram loss + physical sinogram rules from the DICOM export path.

	Goal: predictions *extremely close* to the ground-truth leaf-open-time (LOT)
	sinogram, while respecting the rules enforced when writing the RTPLAN back to
	DICOM (see generate_rtplan.py / inspect_repair_rtplan_closed_rows.py):

	  R1. Values in [0, 1]                       -> guaranteed by sigmoid.
	  R2. Fully-closed control points must be    -> closed-row penalty (drives the
	      EXACTLY zero (RayStation rejects tiny      whole CP row to 0; this is the
	      sigmoid leakage in closed rows).           rule that gets plans rejected).
	  R3. Tiny values floor to 0 (no negligible  -> L1 on closed leaves has a
	      openings).                                 constant gradient toward exact 0,
	                                                 unlike a squared penalty.

	Terms (additive):
	  - Weighted L1 on sigmoid(p) vs target for the "extremely close" objective.
	    Open leaves are weighted 1 + (pos_weight-1)*target (fit the values that
	    matter precisely); closed leaves keep weight 1 so the open-leaf gradient is
	    not diluted by the 82%-closed majority.
	  - fp_weight * mean(closed * p^2): SQUARED closed penalty -> exact zero but
	    GENTLE near 0. (An L1 closed penalty has a constant grad toward 0 that, with
	    the sigmoid saturating the open-leaf L1 grad near p=0, collapses ALL leaves
	    to zero -- verified in sanity. Squared keeps closed->0 without that trap.)
	  - row_weight * mean over fully-closed target rows of (max p)^2: squared
	    closed-control-point rule (R2), also gentle.
	"""

	def __init__(
		self,
		pos_weight: float = 8.0,
		fp_weight: float = 4.0,
		row_weight: float = 2.0,
		open_threshold: float = 1e-3,
	):
		super().__init__()
		self.pos_weight = pos_weight
		self.fp_weight = fp_weight
		self.row_weight = row_weight
		self.open_threshold = open_threshold

	def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		p = torch.sigmoid(pred)

		open_mask = target > self.open_threshold
		closed = (~open_mask).float()

		# Weighted L1: open leaves scaled by target, closed weight 1 (balanced denom).
		weight = torch.where(
			open_mask,
			1.0 + (self.pos_weight - 1.0) * target,
			torch.ones_like(target),
		)
		l1 = (p - target).abs()
		match = (weight * l1).sum() / weight.sum().clamp_min(1.0)

		# R3: squared precision on closed leaves (drives ->0, gentle near 0).
		fp = (closed * p.pow(2)).sum() / closed.sum().clamp_min(1.0)

		# R2: control points fully closed in the target must be all-zero.
		# A row is closed if its whole target row sums to ~0; penalize its max p
		# (squared, gentle).
		row_open = target.sum(dim=-1) > self.open_threshold          # [..., N_CP]
		row_closed = (~row_open).float()
		row_pen = (row_closed * p.amax(dim=-1).pow(2)).sum() / row_closed.sum().clamp_min(1.0)

		return match + self.fp_weight * fp + self.row_weight * row_pen