import json
import logging
import os
import re
from typing import Any, Any, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
from pathlib import Path
from torch.nn import functional as F
import os
import gzip
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# from .densities import TableDensities

from utils import (
    apply_tomo_transform_to_stack,
)

torch.multiprocessing.set_sharing_strategy('file_system')


def file_exists(path, timeout=2):
	def check_exists():
		return os.path.exists(path)

	with ThreadPoolExecutor(max_workers=1) as executor:
		try:
			return executor.submit(check_exists).result(timeout=timeout)
		except TimeoutError:
			logging.warning("Dead ass NFS")
			return False


class RTDataset(Dataset):
	"""
	Advanced PyTorch Dataset for loading radiotherapy treatment plans.
	
	This dataset loads:
	- CT images (3D volume)
	- RT Plan (treatment parameters)
	- RT Dose (3D dose distribution)
	- JSON data from co.json if available
	
	Each sample is a complete treatment plan with all associated data.
	"""
	
	def __init__(self, root_dir, augmentation=None,  max_dose=80.0, reduction_ratio=4, ban_co_json=[], use_cache=False, cache_dir=None, debug=None, inference=False, co_inference={}, ptv_importances={}, preloaded_raw=None, target_hw=64):
		"""
		Args:
			root_dir (str): Directory with the NASRP structure
			transform (callable, optional): Optional transform to be applied to samples
		"""
		self.root_dir = Path(root_dir)
		self.augmentation = augmentation
		self.samples = []
		self.max_dose = max_dose
		self.reduction_ration = reduction_ratio
		self.target_hw = int(target_hw)   # in-plane size of the berlingo (H=leaf, W=ray)
		self.use_cache = use_cache
		self.cache_dir = Path(cache_dir) if cache_dir else Path(root_dir) / "cache/"	
		logging.info(f"Loading dataset from {self.root_dir}, reduction ratio: {self.reduction_ration}")
		if (self.use_cache):
			os.makedirs(self.cache_dir, exist_ok=True)
		self.debug = debug
		self.debug_iteration = 0
		self.inference = inference
		self.co_inference = co_inference
		self.ptv_importances = ptv_importances
		self.ban_co_json = ban_co_json
		self.preloaded_raw = preloaded_raw
		if co_inference == {} and inference:
			raise ValueError("co_inference data must be provided in inference mode.")

		# empty cache directory
		# if self.use_cache and self.cache_dir.exists():
		# 	for file in self.cache_dir.glob("*.pt"):
		# 		try:
		# 			file.unlink()
		# 		except Exception as e:
		# 			logging.warning(f"Failed to delete cache file {file}: {e}")
  
		# Scan for all patient/plan pairs
		if inference:
			self._open_single_patient()
		else:
			self._scan_directory()
		
	def _get_cache_path(self, idx):
		"""Génère un chemin de cache unique pour un échantillon"""
		sample_info = self.samples[idx]
		# Créer une clé de hachage basée sur les métadonnées du patient et les paramètres de traitement
		hash_key = f"{sample_info['patient_id']}_{sample_info['pareto_index']}"
		return self.cache_dir / f"{hash_key}.pt.gz"

	def _scan_directory(self):
		"""Scan directory and find all patient/plan pairs"""
		# Find all patient folders
		patient_folders = [p for p in self.root_dir.glob("*") if p.is_dir()]
		
		for patient_folder in patient_folders:
			patient_id = patient_folder.name
			if self.debug and patient_id != self.debug:
				continue # debug mode, only load one patient
			
			# Find the tomo folder: free-breathing (Tomo_FB_copy, in "<id>/" dirs)
			# OR deep-inspiration breath-hold (Tomo_DIBH, in "<id>_DIBH/" dirs).
			# Each patient dir holds exactly one; patient_id stays the folder name so
			# FB and DIBH get distinct cache keys ("317396" vs "317396_DIBH").
			# get_patient_based_splits normalizes the id (strips _DIBH) so the two
			# acquisitions of the same person never split across train/val/test.
			tomo_folders = list(patient_folder.glob("**/Tomo_FB_copy")) \
				+ list(patient_folder.glob("**/Tomo_DIBH"))

			if not tomo_folders:
				continue

			tomo_folder = tomo_folders[0]
			
			# Get CT files
			ct_files = sorted([f for f in tomo_folder.glob("CT*.dcm")])
			
			if not ct_files:
				continue
			
			# Get the RS structure file if it exists
			rs_files = list(tomo_folder.glob("RS*.dcm"))
			rs_file = rs_files[0] if rs_files else None
   
			# Find all pareto folders
			pareto_folders = sorted([p for p in tomo_folder.glob("pareto_*") if p.is_dir()])
			
			for pareto_folder in pareto_folders:
				pareto_index = int(pareto_folder.name.split("_")[1])
				
				# Find plan and dose files
				plan_files = list(pareto_folder.glob("RP*.dcm"))
				dose_files = list(pareto_folder.glob("RD*.dcm"))
				
				if not plan_files or not dose_files:
					continue
				
				co_json_path = pareto_folder / "co.json"
				self.samples.append({
					'patient_id': patient_id,
					'pareto_index': pareto_index,
					'ct_files': ct_files,
					'rs_file': rs_file,
					'plan_file': plan_files[0],
					'dose_file': dose_files[0],
					'co_json_path': co_json_path if co_json_path.exists() else None
				})

	def __len__(self):
		"""Retourne le nombre total d'échantillons ou de slices"""

		return len(self.samples)
	
	def __getitem__(self, idx):
		"""Get a sample from the dataset"""
		if torch.is_tensor(idx):
			idx = idx.tolist()

		if self.use_cache and not self.inference:
			cache_path = self._get_cache_path(idx)

			try:
				cache_exists = file_exists(cache_path)
				if cache_exists:
					with gzip.open(cache_path, "rb") as f:
						sample = torch.load(f, weights_only=False)
					# Invalidate old format (bare tensor saved before dict format)
					if not isinstance(sample, dict):
						raise ValueError("stale cache format (bare tensor)")
					logging.debug(f"Loaded sample {idx} from cache")
					return sample
			except Exception as e:
				logging.warning(f"Failed to load {cache_path} from cache: {e}")
				try:
					Path(cache_path).unlink()
					logging.info(f"Deleted corrupted cache file {cache_path}")
				except Exception as e2:
					logging.warning(f"Failed to delete corrupted cache file {cache_path}: {e2}")	

		# Mode volume 3D normal
		full_sample = self._load_full_sample(idx)
		
		# ---- CROP-BEFORE-REDUCE ----
		# Old order downsampled the FULL FOV by reduction_ratio FIRST, then cropped
		# -> the 40 cm field ended up sampled by only ~44 coarse (~9 mm) voxels, and W
		# varied per patient (40cm/(ps*ratio)). New order: crop the 40 cm field around
		# the isocenter at FULL resolution on X, THEN resample the cropped region to a
		# FIXED TARGET_HW. Result: constant [.,TARGET_HW,TARGET_HW] in-plane, finer
		# (~6.25 mm) sampling inside the field, identical scale for every patient.
		TARGET_HW = self.target_hw
		ct_vol = full_sample['ct_volume']                              # [1,1,Z,Y,X] full-res
		Z, Yf, Xf = ct_vol.shape[2], ct_vol.shape[3], ct_vol.shape[4]
		ps_y = float(full_sample['pixel_spacing'][0])
		ps_x = float(full_sample['pixel_spacing'][1])

		# Put dose on the CT full-res grid so the X crop indices line up.
		dose_vol = full_sample['dose_data']['dose_grid']
		if dose_vol.dim() == 4:
			dose_vol = dose_vol.unsqueeze(0)
		dose_vol = F.interpolate(dose_vol, size=(Z, Yf, Xf), mode='trilinear', align_corners=True)

		# Crop X to 40 cm around the isocenter, at full resolution.
		x_iso_mm = float(full_sample['plan_data']['cps'][0].IsocenterPosition[0])
		ct_origin_x_mm = float(full_sample['ct_origin_x'])
		tw = int(round(400.0 / ps_x))                                  # 40 cm in full-res voxels
		iso_vox_x = int(round((x_iso_mm - ct_origin_x_mm) / ps_x))
		w_start = max(0, iso_vox_x - tw // 2)
		w_end = w_start + tw
		if w_end > Xf:
			w_end = Xf
			w_start = max(0, w_end - tw)
		ct_vol = ct_vol[:, :, :, :, w_start:w_end]
		dose_vol = dose_vol[:, :, :, :, w_start:w_end]
		crop_x_mm = (w_end - w_start) * ps_x
		full_sample['ct_origin_x'] = ct_origin_x_mm + w_start * ps_x

		# Crop Y to 40 cm around the isocenter too, MIRRORING the X crop, so the
		# in-plane grid is ISOTROPIC (~6.25 mm) and spans exactly the MLC field on
		# BOTH axes. Previously only X was cropped (Y kept full extent): the grid was
		# anisotropic, so the pixel-space rotation in apply_tomo_transform_to_stack
		# sheared the anatomy by an angle-dependent amount, and the leaf axis (Y) was
		# mis-scaled vs the 40 cm / 64-leaf sinogram -> per-control-point
		# misregistration of the berlingo against the LOT sinogram.
		y_iso_mm = float(full_sample['plan_data']['cps'][0].IsocenterPosition[1])
		ct_origin_y_mm = float(full_sample['ct_origin_y'])
		thh = int(round(400.0 / ps_y))                                 # 40 cm in full-res voxels
		iso_vox_y = int(round((y_iso_mm - ct_origin_y_mm) / ps_y))
		h_start = max(0, iso_vox_y - thh // 2)
		h_end = h_start + thh
		if h_end > Yf:
			h_end = Yf
			h_start = max(0, h_end - thh)
		ct_vol = ct_vol[:, :, :, h_start:h_end, :]
		dose_vol = dose_vol[:, :, :, h_start:h_end, :]
		crop_y_mm = (h_end - h_start) * ps_y
		full_sample['ct_origin_y'] = ct_origin_y_mm + h_start * ps_y

		# Resample the cropped volume to fixed (Z, TARGET_HW, TARGET_HW).
		ct = F.interpolate(ct_vol, size=(Z, TARGET_HW, TARGET_HW), mode='trilinear', align_corners=True)
		dose = F.interpolate(dose_vol, size=(Z, TARGET_HW, TARGET_HW), mode='trilinear', align_corners=True).squeeze(0)

		# Normaliser
		ct = (ct - (-1024)) / (3071 - (-1024))
		dose = dose / self.max_dose

		# Effective in-plane spacings AFTER crop+resample (X and Y both cover the
		# 40 cm field around the iso -> isotropic grid). The transform needs these
		# to place rays.
		x_iso = x_iso_mm
		y_iso = y_iso_mm
		z_iso = float(full_sample['plan_data']['cps'][0].IsocenterPosition[2])
		spacing_zyx = (
			full_sample['ct_dz'],            # Z: slice thickness, unchanged
			crop_y_mm / TARGET_HW,           # Y: cropped 40cm field resampled to TARGET_HW
			crop_x_mm / TARGET_HW,           # X: cropped 40cm field resampled to TARGET_HW
		)

		## TURN TO BERLINGO
		dose_berlingo = apply_tomo_transform_to_stack(
			mask_zyx=dose.squeeze(),
			angles=full_sample['plan_data']['angles'],
			tables=full_sample['plan_data']['tables'],
			x_iso=x_iso, y_iso=y_iso, z_iso=z_iso,
			spacing_zyx=spacing_zyx,
			origin_zyx=(full_sample['ct_origin_z'], full_sample['ct_origin_y'], full_sample['ct_origin_x']),
            is_label=False
		)

		ct_berlingo = apply_tomo_transform_to_stack(
			mask_zyx=ct.squeeze(),
			angles=full_sample['plan_data']['angles'],
			tables=full_sample['plan_data']['tables'],
			x_iso=x_iso, y_iso=y_iso, z_iso=z_iso,
			spacing_zyx=spacing_zyx,
			origin_zyx=(full_sample['ct_origin_z'], full_sample['ct_origin_y'], full_sample['ct_origin_x']),
            is_label=False
		)
  
		# go back to tensor 
		dose_berlingo = torch.tensor(dose_berlingo, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]
		ct_berlingo = torch.tensor(ct_berlingo, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]
  
   
		# Combiner les 2 canaux
		sample_input = torch.cat([
			ct_berlingo,
			dose_berlingo
		])  # [2, D, H, W]
  
		logging.debug(f"Sample input shape: {sample_input.shape}")
  
		result = {
			'input': sample_input,  # [2, D, H, W]
			'target': torch.tensor(full_sample['plan_data']['sino'], dtype=torch.float32),  # [N_CP, 64]
			'patient_id': full_sample['patient_id'],
			'pareto_index': full_sample['pareto_index'],
		}

		if self.use_cache and not self.inference:
			tmp_path = cache_path.with_suffix(".tmp")
			try:
				with gzip.open(tmp_path, "wb") as f:
					torch.save(result, f)
				tmp_path.rename(cache_path)
				logging.debug(f"Saved sample {idx} to cache")
			except Exception as e:
				logging.warning(f"Failed to save to cache: {e} KeyboardInterrupt ?")
				if tmp_path.exists():
					tmp_path.unlink(missing_ok=True)

		return result

	def _extract_sinogram(self, cps) -> np.ndarray:
		"""Retourne le sinogramme normalisé [N_CP, 64] à partir du ControlPointSequence Tomo."""
		lines = []
		for cp in cps:
			tag = (0x300D, 0x10A7)  # Tomo: Private tag "Sinogram Data" (bytes séparés par '\')
			if tag in cp:
				val = cp[tag].value
				if isinstance(val, (bytes, bytearray)):
					parts = val.decode(errors="ignore").split("\\")
					row = [float(x) for x in parts if x.strip() != ""]
					if len(row) == 64:
						lines.append(row)
					else:
						logging.warning(f"Expected 64 values in sinogram line, got {len(row)}.")
		if not lines:
			raise RuntimeError("Sinogramme introuvable dans le RT-PLAN (tag privé 300D,10A7 manquant).")
		return np.array(lines, dtype=np.float32)  # ∈[0,1]

	def _extract_beam_meterset_minutes(self, ds: pydicom.Dataset) -> float:
		"""
		Tomo/Radixact: Beam Meterset (= temps planifié en minutes) se trouve
		dans FractionGroupSequence / ReferencedBeamSequence / BeamMeterset (300A,0086).
		Fallback sur BeamSequence[0].BeamMeterset si jamais présent.
		"""
		# 1) Chemin "canonique" DICOM pour le meterset par beam
		try:
			if "FractionGroupSequence" in ds:
				fgs = ds.FractionGroupSequence[0]
				if "ReferencedBeamSequence" in fgs and len(fgs.ReferencedBeamSequence) > 0:
					# S'il y a plusieurs beams, on prend le premier (tomothérapie = 1).
					refb = fgs.ReferencedBeamSequence[0]
					if hasattr(refb, "BeamMeterset"):
						return float(refb.BeamMeterset)
		except Exception:
			pass

		# 2) Fallback direct (rarement rempli chez Tomo/Radixact)
		try:
			if "BeamSequence" in ds and len(ds.BeamSequence) > 0:
				beam = ds.BeamSequence[0]
				if hasattr(beam, "BeamMeterset"):
					return float(beam.BeamMeterset)
		except Exception:
			pass

		return 0.0

	def _extract_tomo_private(self, ds: pydicom.Dataset) -> Dict[str, Any]:
		"""
		Extrait quelques tags privés Tomo si présents.
		Retourne un dict avec 'gantry_period_sec', 'treatment_pitch', 'couch_speed_mm_per_s'.
		"""
		vals = {"gantry_period_sec": float("nan"),
				"treatment_pitch": float("nan"),
				"couch_speed_mm_per_s": float("nan")}
		try:
			beam = ds[(0x300A, 0x00B0)][0]
		except Exception:
			beam = None

		def _get_tag(container, tag):
			try:
				if container is not None and tag in container:
					return container[tag].value
			except Exception:
				pass
			try:
				if tag in ds:
					return ds[tag].value
			except Exception:
				pass
			return None

		# Accuray Tomo private creator souvent 'TOMO_HA_01'
		# 300D,1040 : Gantry Period (s)
		v = _get_tag(beam, (0x300D, 0x1040))
		if v is None:
			v = _get_tag(ds, (0x300D, 0x1040))
		if v is not None:
			try:
				vals["gantry_period_sec"] = float(v)
			except Exception:
				pass

		# 300D,1060 : Treatment Pitch (sans unité)
		v = _get_tag(beam, (0x300D, 0x1060))
		if v is None:
			v = _get_tag(ds, (0x300D, 0x1060))
		if v is not None:
			try:
				vals["treatment_pitch"] = float(v)
			except Exception:
				pass

		# 300D,1080 : Couch Speed (mm/s)
		v = _get_tag(beam, (0x300D, 0x1080))
		if v is None:
			v = _get_tag(ds, (0x300D, 0x1080))
		if v is not None:
			try:
				vals["couch_speed_mm_per_s"] = float(v)
			except Exception:
				pass

		return vals

	def _extract_rtplan_scalars(self, ds, cps):
		"""
		Remarques Tomo/Radixact :
		- BeamMeterset (300A,0086) = temps planifié (minutes)
		- CumulativeMetersetWeight (CMW) ∈ [0..1] croissant sur CP
		"""
		beam = ds[(0x300A, 0x00B0)][0]

		# BeamMeterset en minutes (Tomo/Radixact)
		beam_meterset_minutes = self._extract_beam_meterset_minutes(ds)

		# nombre de CP
		number_of_control_points = int(getattr(beam, "NumberOfControlPoints", len(cps)))

		# séquences CP → vecteurs numpy
		gantry_angle = np.array([float(getattr(cp, "GantryAngle", 0.0)) for cp in cps], dtype=np.float32)
		table_top_lateral_position = np.array(
			[float(getattr(cp, "TableTopLateralPosition", 0.0)) for cp in cps], dtype=np.float32
		)
		cumulative_meterset_weight = np.array(
			[float(getattr(cp, "CumulativeMetersetWeight", 0.0)) for cp in cps], dtype=np.float32
		)

		# isocentre depuis CP0
		cp0 = cps[0]
		if (0x300A, 0x012C) in cp0:
			x_iso, y_iso, z_iso = map(float, cp0[(0x300A, 0x012C)].value)
		else:
			x_iso, y_iso, z_iso = map(float, getattr(cp0, "IsocenterPosition"))

		# tags privés Tomo utiles
		tomo_priv = self._extract_tomo_private(ds)

		# stockage attributs instance
		self.beam_meterset_minutes = beam_meterset_minutes
		self.number_of_control_points = number_of_control_points
		self.control_point_seq = cps
		self.gantry_angle = gantry_angle
		self.table_top_lateral_position = table_top_lateral_position
		self.cumulative_meterset_weight = cumulative_meterset_weight
		self.tomo_private = tomo_priv

		return {
			"isocentre": (x_iso, y_iso, z_iso),
			"gantry_angles": gantry_angle,
			"table_positions": table_top_lateral_position,
			"n_cp": number_of_control_points
		}

	def _load_rt_plan(self, plan_file):
		"""Load RT Plan file and extract relevant parameters."""
		if plan_file is None:
			return None

		try:
			plan = pydicom.dcmread(str(plan_file))
   
			beam =  plan[(0x300A, 0x00B0)][0]
			cps = beam[(0x300A, 0x0111)].value
   
			sino = self._extract_sinogram(cps)
   
			rt_scal = self._extract_rtplan_scalars(plan, cps)
			
			plan_info = {
				'sino': sino,
				'cps': cps,
				'angles': np.asarray(rt_scal["gantry_angles"], dtype=np.float32).tolist(),
				'tables': np.asarray(rt_scal["table_positions"], dtype=np.float32).tolist(),
			}
			
			return plan_info

		except Exception as e:
			logging.error(f"Failed to load RT Plan file {plan_file}: {e}")
			return None
		
	def _load_full_sample(self, idx):
		"""Charge un échantillon complet 3D"""
		sample_info = self.samples[idx]
		logging.debug(f"Loading sample {idx}: Patient {sample_info['patient_id']}, Pareto index {sample_info['pareto_index']}")

		raw_data = None


		# Chargement complet depuis le disque
		ct_volume, pixel_spacing, ct_origin_x, ct_origin_y, ct_origin_z, ct_dz = self._load_ct_volume(sample_info['ct_files'])

		# Charger la dose RT
		dose_data = self._load_rt_dose(sample_info['dose_file'], sample_info['ct_files'])

		co_json_data = None
		if sample_info['co_json_path'] and sample_info['co_json_path'].exists():
			with open(sample_info['co_json_path'], 'r') as f:
				co_json_data = json.load(f)


		# charger rt plan
		plan_data = self._load_rt_plan(sample_info['plan_file'])

		raw_data = {
			'ct_volume': ct_volume,
			'pixel_spacing': pixel_spacing,
			'co_json_data': co_json_data,
		}

		# Créer l'échantillon
		sample = {
			'patient_id': sample_info['patient_id'],
			'pareto_index': sample_info['pareto_index'],
			'ct_volume': ct_volume,
			'pixel_spacing': pixel_spacing,
			'ct_origin_x': ct_origin_x,
			'ct_origin_y': ct_origin_y,
			'ct_origin_z': ct_origin_z,
			'ct_dz': ct_dz,
			'raw_data': raw_data,
			'plan_data': plan_data,
		}

		sample['dose_data'] = None
		sample['metadata'] = {
			'co_json_path': None
		}
		sample['dose_data'] = dose_data
		sample['metadata']['co_json_path'] = str(sample_info['co_json_path']) if sample_info['co_json_path'] else None

		return sample
	
	def _center_crop_tensor(self, tensor, target_shape):
		"""
		Center crop a tensor to match a target shape.
		If tensor is smaller than target, it will be returned as is.
		
		Args:
			tensor: Input tensor with shape [B, C, D, H, W]
			target_shape: Tuple (D, H, W) target shape
		
		Returns:
			Cropped tensor of shape [B, C, D, H, W] with spatial dimensions matching target_shape
		"""
		current_shape = tensor.shape[2:]  # Get spatial dimensions [D, H, W]
		
		# Calculate crop amounts for each dimension
		crop_d = max(0, current_shape[0] - target_shape[0])
		crop_h = max(0, current_shape[1] - target_shape[1])
		crop_w = max(0, current_shape[2] - target_shape[2])
		
		# Calculate start and end positions for each dimension
		start_d = crop_d // 2
		end_d = current_shape[0] - (crop_d - start_d)
		
		start_h = crop_h // 2
		end_h = current_shape[1] - (crop_h - start_h)
		
		start_w = crop_w // 2
		end_w = current_shape[2] - (crop_w - start_w)
		
		# Perform the center crop
		cropped_tensor = tensor[:, :, start_d:end_d, start_h:end_h, start_w:end_w]
		
		return cropped_tensor
	
	def _load_ct_volume(self, ct_files):
		"""Load CT files and create a 3D volume"""
		slices = []
		positions = []
		
		# Load all CT slices
		for file in ct_files:
			try:
				ds = pydicom.dcmread(str(file))
				pixel_spacing = ds.PixelSpacing  # assuming square pixels
				slices.append(ds)
				# Get slice position
				pos = ds.ImagePositionPatient[2]  # Z position
				positions.append((pos, ds))
			except Exception as e:
				logging.error(f"Failed to read CT file: {file}")
				raise
		
		# Sort slices by position
		positions.sort(key=lambda x: x[0])
		sorted_slices = [item[1] for item in positions]
		
		# Extract pixel arrays
		volume = np.stack([s.pixel_array for s in sorted_slices])
		
		# Apply rescaling if needed
		if hasattr(sorted_slices[0], 'RescaleSlope') and hasattr(sorted_slices[0], 'RescaleIntercept'):
			slope = sorted_slices[0].RescaleSlope
			intercept = sorted_slices[0].RescaleIntercept
			volume = volume * slope + intercept
		
		# Convert to tensor
		volume_tensor = torch.from_numpy(volume).float()
		volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)

		ct_origin_x = float(sorted_slices[0].ImagePositionPatient[0])
		ct_origin_y = float(sorted_slices[0].ImagePositionPatient[1])
		ct_origin_z = float(sorted_slices[0].ImagePositionPatient[2])
		nz = len(sorted_slices)
		ct_dz = (float(sorted_slices[-1].ImagePositionPatient[2]) - ct_origin_z) / (nz - 1) if nz > 1 else float(sorted_slices[0].SliceThickness)

		return volume_tensor, pixel_spacing, ct_origin_x, ct_origin_y, ct_origin_z, ct_dz
	
	def _load_rt_dose(self, dose_file, ct_files):
		"""Load RT Dose and resample it onto the CT grid using physical DICOM coordinates."""
		try:
			ds = pydicom.dcmread(str(dose_file))

			# --- CT geometry ---
			ct_metas = [pydicom.dcmread(str(f), stop_before_pixels=True) for f in ct_files]
			ct_metas.sort(key=lambda x: float(x.ImagePositionPatient[2]))
			ct_first = ct_metas[0]
			ct_last  = ct_metas[-1]
			ct_nz = len(ct_metas)
			ct_ny = int(ct_first.Rows)
			ct_nx = int(ct_first.Columns)
			ct_ipp = np.array([float(v) for v in ct_first.ImagePositionPatient])
			ct_ps_row = float(ct_first.PixelSpacing[0])   # espacement entre lignes (y)
			ct_ps_col = float(ct_first.PixelSpacing[1])   # espacement entre colonnes (x)
			ct_dz = (float(ct_last.ImagePositionPatient[2]) - float(ct_first.ImagePositionPatient[2])) / (ct_nz - 1) if ct_nz > 1 else float(ct_first.SliceThickness)

			# --- Dose geometry ---
			dose_grid = ds.pixel_array.astype(np.float32)
			if hasattr(ds, 'DoseGridScaling'):
				dose_grid *= float(ds.DoseGridScaling)

			dose_ipp    = np.array([float(v) for v in ds.ImagePositionPatient])
			dose_ps_row = float(ds.PixelSpacing[0])
			dose_ps_col = float(ds.PixelSpacing[1])
			dose_nz, dose_ny, dose_nx = dose_grid.shape

			if hasattr(ds, 'GridFrameOffsetVector'):
				dose_z_offsets = np.array([float(v) for v in ds.GridFrameOffsetVector])
			else:
				dose_dz_fallback = float(ds.SliceThickness) if hasattr(ds, 'SliceThickness') else abs(ct_dz)
				dose_z_offsets = np.arange(dose_nz) * dose_dz_fallback

			dose_z_start = dose_ipp[2] + dose_z_offsets[0]
			dose_dz = (dose_z_offsets[-1] - dose_z_offsets[0]) / (dose_nz - 1) if dose_nz > 1 else 1.0

			# --- Coordonnées physiques des voxels CT ---
			ct_x = ct_ipp[0] + np.arange(ct_nx) * ct_ps_col   # [ct_nx]
			ct_y = ct_ipp[1] + np.arange(ct_ny) * ct_ps_row   # [ct_ny]
			ct_z = ct_ipp[2] + np.arange(ct_nz) * ct_dz        # [ct_nz]

			# --- Conversion en indices dose (espace dose) ---
			dose_coord_x = (ct_x - dose_ipp[0])   / dose_ps_col   # [ct_nx]
			dose_coord_y = (ct_y - dose_ipp[1])   / dose_ps_row   # [ct_ny]
			dose_coord_z = (ct_z - dose_z_start)  / dose_dz        # [ct_nz]

			# --- Normalisation en [-1, 1] pour F.grid_sample (align_corners=True) ---
			norm_x = (dose_coord_x / (dose_nx - 1)) * 2 - 1
			norm_y = (dose_coord_y / (dose_ny - 1)) * 2 - 1
			norm_z = (dose_coord_z / (dose_nz - 1)) * 2 - 1

			# grid_sample attend grid[..., 0]=x(W), [1]=y(H), [2]=z(D)
			gz, gy, gx = np.meshgrid(norm_z, norm_y, norm_x, indexing='ij')  # [ct_nz, ct_ny, ct_nx]
			grid = np.stack([gx, gy, gz], axis=-1).astype(np.float32)        # [ct_nz, ct_ny, ct_nx, 3]
			grid_tensor = torch.from_numpy(grid).unsqueeze(0)                 # [1, ct_nz, ct_ny, ct_nx, 3]

			dose_tensor_5d = torch.from_numpy(dose_grid).unsqueeze(0).unsqueeze(0)  # [1, 1, dose_nz, dose_ny, dose_nx]

			dose_on_ct = F.grid_sample(
				dose_tensor_5d,
				grid_tensor,
				mode='bilinear',
				padding_mode='zeros',   # 0 Gy hors du champ de la dose RT
				align_corners=True
			)  # [1, 1, ct_nz, ct_ny, ct_nx]

			dose_info = {
				'dose_type':             ds.DoseType            if hasattr(ds, 'DoseType')            else None,
				'dose_unit':             ds.DoseUnits           if hasattr(ds, 'DoseUnits')           else None,
				'dose_summation_type':   ds.DoseSummationType   if hasattr(ds, 'DoseSummationType')   else None,
				'dose_grid_scaling':     float(ds.DoseGridScaling) if hasattr(ds, 'DoseGridScaling') else None,
			}

			return {'dose_grid': dose_on_ct, 'info': dose_info}

		except Exception as e:
			print(f"Error loading RT Dose file {dose_file}: {e}")
			return None
	
	
	def _find_slice_index(self, z_pos, ref_ct):
		"""
		Find the corresponding slice index in the CT volume for a given z position.
		
		Args:
			z_pos (float): Z position in world coordinates
			num_slices (int): Number of slices in CT volume
			ref_ct (pydicom.dataset.FileDataset): Reference CT dicom
			
		Returns:
			int: Slice index or None if out of bounds
		"""
		try:
			# Get image position and slice thickness
			image_position = ref_ct.ImagePositionPatient[2]  # Z position of first slice
			slice_thickness = ref_ct.SliceThickness
			
			# Calculate the slice index
			# NOTE: le -1 est un bug historique mais le modèle a été entraîné avec
			slice_idx = int(round((z_pos - image_position) / slice_thickness)) -1
			
			return slice_idx
		except Exception as e:
			print(f"Error finding slice index: {e}")
			return None


	
def get_patient_based_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
	"""
	Split dataset by patient to ensure no patient appears in multiple splits.

	Args:
		dataset (RTDataset): The dataset to split
		train_ratio (float): Ratio of patients for training (default: 0.7)
		val_ratio (float): Ratio of patients for validation (default: 0.15)
		test_ratio (float): Ratio of patients for test (default: 0.15)
		seed (int): Random seed for reproducibility (default: 42)

	Returns:
		tuple: (train_indices, val_indices, test_indices)
	"""
	# Group samples by NORMALIZED patient id (strip the "_DIBH" suffix) so that the
	# free-breathing and breath-hold acquisitions of the same person always land in
	# the SAME split -> no patient-identity leakage between train/val/test.
	patient_to_indices = {}
	for idx, sample_info in enumerate(dataset.samples):
		patient_id = sample_info['patient_id'].replace("_DIBH", "")
		if patient_id not in patient_to_indices:
			patient_to_indices[patient_id] = []
		patient_to_indices[patient_id].append(idx)

	# Get list of unique patients
	patients = list(patient_to_indices.keys())
	np.random.seed(seed)
	np.random.shuffle(patients)

	# Calculate split indices
	n_patients = len(patients)
	n_train = int(n_patients * train_ratio)
	n_val = int(n_patients * val_ratio)

	# Split patients into train/val/test
	train_patients = patients[:n_train]
	val_patients = patients[n_train:n_train + n_val]
	test_patients = patients[n_train + n_val:]

	# Get sample indices for each split
	train_indices = []
	val_indices = []
	test_indices = []

	for patient in train_patients:
		train_indices.extend(patient_to_indices[patient])

	for patient in val_patients:
		val_indices.extend(patient_to_indices[patient])

	for patient in test_patients:
		test_indices.extend(patient_to_indices[patient])

	return train_indices, val_indices, test_indices

