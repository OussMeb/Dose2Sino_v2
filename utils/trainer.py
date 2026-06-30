
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from utils.augmentation import RTDataAugmentation
from torch.optim import Adam
from utils.patient import RTDataset, get_patient_based_splits

from utils.config import Config


class AugmentedSubset(torch.utils.data.Dataset):
	"""Wrapper that applies augmentation to a Subset, so that
	augmentation is only applied to the training split."""
	def __init__(self, subset, augmentation):
		self.subset = subset
		self.augmentation = augmentation

	def __getitem__(self, idx):
		sample = self.subset[idx]
		if self.augmentation:
			sample = self.augmentation(sample)
		return sample

	def __len__(self):
		return len(self.subset)


class Trainer:
	def __init__(self, config: Config, model: torch.nn.Module, device: torch.device, 
				resume_timestamp: str = None, loss_function: callable = None, 
				training_mode: str = 'supervised', phase_suffix: str = ""):
		
		self.config = config
		self.model = model.to(device)
		self.device = device
		self.loss_function = loss_function if loss_function else None
		self.training_mode = training_mode
		
		# AJOUT: Initialiser le GradScaler pour mixed precision
		self.scaler = torch.amp.GradScaler("cuda") if config.USE_MIXED_PRECISION and device.type == "cuda" else None
		
		# Setup directories avec suffixe pour les phases
		if resume_timestamp:
			self.timestamp = resume_timestamp
		else:
			self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			
		# Ajouter le suffixe de phase au répertoire
		if phase_suffix:
			self.checkpoint_dir = Path(config.CHECKPOINT_DIR) / f"{self.timestamp}_{phase_suffix}"
		else:
			self.checkpoint_dir = Path(config.CHECKPOINT_DIR) / self.timestamp
			
		self.visualization_dir = self.checkpoint_dir / "visualizations"
		self._setup_directories()

		self.session_prefix = self._get_next_session_prefix() 
		
		# Setup logging
		self._setup_logging(resume_mode=resume_timestamp is not None)
		# Log configuration
		logging.info(f"Training configuration:\n{config}")
		
		# Log mixed precision status
		if self.config.USE_MIXED_PRECISION:
			logging.info("Mixed precision training enabled")
		
		# Setup data
		self.train_loader, self.val_loader, self.test_loader = self._setup_data(config)
		
		# Setup optimizer and scheduler
		self.optimizer = Adam(
			self.model.parameters(),
			lr=config.LEARNING_RATE,
			weight_decay=getattr(config, "WEIGHT_DECAY", 0.0),
		)
		self.scheduler = self._setup_scheduler()
		
		# Training state
		self.best_val_loss = float('inf')
		self.early_stop_counter = 0
		self.last_loss = float('inf')    

	def train_epoch(self, epoch: int) -> float:
		raise NotImplementedError()

	def validate(self) -> float:
		raise NotImplementedError()

	def test(self, model_path: Optional[str] = None, verbose: bool = False) -> float:
		raise NotImplementedError()

	def train(self):
		raise NotImplementedError()

	def _get_next_session_prefix(self):
		"""Trouve le prochain numéro de session pour les reprises."""
		existing_files = list(self.visualization_dir.glob("session_*.png"))
		if not existing_files:
			return "session_0_"
		
		# Extraire les numéros de session existants
		session_numbers = []
		for file in existing_files:
			if "session_" in file.name:
				try:
					num = int(file.name.split("session_")[1].split("_")[0])
					session_numbers.append(num)
				except:
					pass
		next_session = max(session_numbers, default=0) + 1
		return f"session_{next_session}"

	def _setup_directories(self):
		"""Create necessary directories."""
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
		self.visualization_dir.mkdir(parents=True, exist_ok=True)
	
	def _setup_logging(self, resume_mode=False):
		"""Setup logging configuration."""
		# Mode append si resume, sinon write
		mode = 'a' if resume_mode else 'w'
		
		LEVEL = logging.INFO

		# Récupérer le logger root
		logger = logging.getLogger()
		logger.setLevel(LEVEL)
		
		# Supprimer tous les handlers existants pour éviter les doublons
		for handler in logger.handlers[:]:
			logger.removeHandler(handler)
		
		# Formatter commun
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		
		# Handler pour la console
		console_handler = logging.StreamHandler()
		console_handler.setLevel(LEVEL)
		console_handler.setFormatter(formatter)
		logger.addHandler(console_handler)
		
		# Handler pour le fichier
		file_handler = logging.FileHandler(self.visualization_dir / 'training.log', mode=mode)
		file_handler.setLevel(LEVEL)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

		# Create journal.txt if it doesn't exist
		journal_path = self.checkpoint_dir / 'journal.txt'
		if not journal_path.exists():
			journal_path.touch()
		
		# Write journal content if provided
		if self.config.JOURNAL:
			with open(journal_path, 'a') as f:
				f.write(f"{self.config.JOURNAL.__str__()}\n")
		
		if resume_mode:
			logging.info("=== TRAINING RESUMED ===")
		else:
			logging.info("=== TRAINING STARTED ===")
	
	def _setup_data(self, config:Config) -> tuple:
		"""Setup datasets and dataloaders."""
		logging.info("Loading dataset...")
		
		# Augmentation for training only, gated by USE_AUGMENTATION (default off).
		# Geometry-safe transforms only (see utils/augmentation.py): leaf-axis flip
		# (with target), ray-axis flip (input only), CT intensity jitter. In-plane
		# rotate/zoom are NOT valid on a cached berlingo, so FLIP_PROB drives both
		# flips here. NOTE: leaf-flip assumes top<->bottom leaf symmetry; the
		# 2026-06-17 run showed it HURT val (0.287 vs 0.260 no-aug), so keep off
		# unless re-testing without the leaf flip.
		augmentation = None
		if getattr(self.config, "USE_AUGMENTATION", False):
			augmentation = RTDataAugmentation(
				flip_leaf_prob=self.config.FLIP_PROB,
				flip_ray_prob=self.config.FLIP_PROB,
				ct_jitter_prob=self.config.ZOOM_PROB,
			)
		
		# Create dataset
		dataset = RTDataset(
			self.config.DATA_PATH,
			augmentation=None,
			max_dose=self.config.MAX_DOSE,
			use_cache=config.USE_CACHE,
			cache_dir=os.path.join(self.config.CACHE_DIR),
			reduction_ratio=config.REDUCTION_RATIO,
			target_hw=getattr(self.config, "TARGET_HW", 64),
		)
		
		# Split dataset by patient to ensure no patient appears in multiple splits
		train_indices, val_indices, test_indices = get_patient_based_splits(
			dataset,
			train_ratio=self.config.TRAIN_SIZE,
			val_ratio=self.config.VALIDATION_SIZE,
			test_ratio=1.0 - self.config.TRAIN_SIZE - self.config.VALIDATION_SIZE,
			seed=42
		)

		train_dataset = AugmentedSubset(Subset(dataset, train_indices), augmentation)
		val_dataset = Subset(dataset, val_indices)
		test_dataset = Subset(dataset, test_indices)

		dataset_size = len(dataset)
		
		# Create dataloaders
		train_loader = DataLoader(
			train_dataset,
			batch_size=self.config.BATCH_SIZE,
			num_workers=self.config.NUM_WORKERS,
			shuffle=True,
			persistent_workers=self.config.NUM_WORKERS > 0,
		)
		
		val_loader = DataLoader(
			val_dataset,
			batch_size=1,
			shuffle=False,
			num_workers=self.config.NUM_WORKERS,
			persistent_workers=False,
		)
		
		test_loader = DataLoader(
			test_dataset,
			batch_size=1,
			shuffle=False,
			num_workers=self.config.NUM_WORKERS,
			persistent_workers=False,
		)
		
		# Log dataset statistics
		train_patients = set(dataset.samples[i]['patient_id'] for i in train_indices)
		val_patients = set(dataset.samples[i]['patient_id'] for i in val_indices)
		test_patients = set(dataset.samples[i]['patient_id'] for i in test_indices)

		logging.info(f"Dataset loaded: {dataset_size} samples")
		logging.info(f"Train: {len(train_dataset)} samples ({len(train_patients)} patients)")
		logging.info(f"Val: {len(val_dataset)} samples ({len(val_patients)} patients)")
		logging.info(f"Test: {len(test_dataset)} samples ({len(test_patients)} patients)")

		# Verify no patient overlap
		assert len(train_patients & val_patients) == 0, "Patient overlap between train and val!"
		assert len(train_patients & test_patients) == 0, "Patient overlap between train and test!"
		assert len(val_patients & test_patients) == 0, "Patient overlap between val and test!"
		logging.info("✓ No patient overlap between splits")
		
		return train_loader, val_loader, test_loader
	
	def _setup_scheduler(self):
		"""Setup learning rate scheduler.

		LR_SCHEDULE='cosine' -> CosineAnnealingLR over COSINE_T_MAX_EPOCHS epochs
		(stepped PER EPOCH). Cosine polished fine details in the sanity; the horizon
		is scaled to the training length (NOT the sanity's fast 900-step rate, which
		would hit eta_min in <1 epoch). 'plateau' -> the previous ReduceLROnPlateau.
		"""
		schedule = getattr(self.config, "LR_SCHEDULE", "plateau")
		if schedule == "cosine":
			t_max = getattr(self.config, "COSINE_T_MAX_EPOCHS", 20)
			return CosineAnnealingLR(
				self.optimizer,
				T_max=t_max,
				eta_min=getattr(self.config, "MIN_LR", 1e-5),
			)
		return ReduceLROnPlateau(
			self.optimizer,
			mode='min',
			factor=self.config.LR_REDUCTION_FACTOR,
			patience=self.config.LR_PATIENCE,
			min_lr=self.config.MIN_LOSS_THRESHOLD
		)

	def _save_visualization(self, outputs, targets, batch, epoch, loss_value):
		"""Save visualization of sinogram predictions.

		Input channels (batch['input']  [2, D, H, W]):
		  ch0 : CT berlingo (normalisé 0-1)
		  ch1 : dose berlingo (normalisé par max_dose)

		outputs / targets : sinogrammes  [1, N_CP, 64]
		"""
		vis_dir = self.checkpoint_dir / "visualizations"
		vis_dir.mkdir(exist_ok=True)

		patient_id   = batch.get('patient_id',   ['unknown'])
		pareto_index = batch.get('pareto_index', ['N/A'])

		# Sinogrammes  [N_CP, 64]  — squeeze channel and det_x dims from [1, N_CP, 64, 1]
		pred_sino   = outputs[0].squeeze().detach().float().cpu().numpy()
		target_sino = targets[0].squeeze().detach().float().cpu().numpy()

		# Canaux d'entrée  [2, D, H, W]
		inp = batch['input'][0]           # [2, D, H, W]
		mid_slice = inp.shape[1] // 2    # coupe axiale centrale

		ct_slice   = inp[0, mid_slice].detach().cpu().numpy()
		dose_slice = inp[1, mid_slice].detach().cpu().numpy() * self.config.MAX_DOSE

		# figure 2×3
		fig, axes = plt.subplots(2, 3, figsize=(18, 10))

		def show(ax, img, title, cmap, vmin=None, vmax=None, label=''):
			im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
			ax.set_title(title, fontsize=9, pad=4)
			ax.axis('off')
			cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
			if label:
				cb.set_label(label, fontsize=7)

		# Ligne 0 : sinogrammes prédit / cible / différence absolue
		show(axes[0, 0], pred_sino,
		     f'Sinogramme prédit  (epoch {epoch+1})', 'hot', 0, 1, 'leaf open frac.')
		show(axes[0, 1], target_sino,
		     f'Sinogramme cible  loss={loss_value:.5f}', 'hot', 0, 1, 'leaf open frac.')
		show(axes[0, 2], np.abs(pred_sino - target_sino),
		     'Différence absolue', 'RdYlGn_r', 0, 0.5)

		# Ligne 1 : CT berlingo | dose berlingo | vide
		show(axes[1, 0], ct_slice,   'CT berlingo  [ch0]', 'gray', 0, 1, 'HU norm.')
		show(axes[1, 1], dose_slice, 'Dose berlingo  [ch1]', 'hot', 0, self.config.MAX_DOSE, 'Gy')
		axes[1, 2].axis('off')

		plt.suptitle(
			f'Patient: {patient_id[0]}   Pareto: {pareto_index[0]}   '
			f'Slice: {mid_slice}/{inp.shape[1]}',
			fontsize=12, y=1.01
		)
		plt.tight_layout()

		p = pareto_index[0]
		safe_pareto = str(p.item() if hasattr(p, 'item') else p).replace('/', '_').replace('\\', '_')
		save_path = vis_dir / f'{self.session_prefix}_epoch_{epoch+1}_patient_{patient_id[0]}_pareto_{safe_pareto}.png'
		plt.savefig(save_path, dpi=130, bbox_inches='tight')
		plt.close()
		logging.info(f"Visualization saved: {save_path}")
	
	def save_config(self, description: str = ""):
		"""Save training configuration."""
		config_path = self.visualization_dir / "config.txt"
		
		with open(config_path, 'w') as f:
			f.write(f"{self.config.__str__()}")
			

	def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
		"""Save model checkpoint with scaler state."""
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		resume = "resume" if self.config.RESUME else "new"
		checkpoint_path = self.checkpoint_dir / f"best_model_{resume}_session_{self.session_prefix}.pth"
		
		checkpoint_data = {
			'epoch': epoch,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),
			'train_loss': train_loss,
			'val_loss': val_loss,
			'best_val_loss': self.best_val_loss, 
			'lr': self.optimizer.param_groups[0]['lr'],
			'config': self.config.__dict__,
		}

		if self.config.USE_MIXED_PRECISION and self.scaler:
			checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
		
		torch.save(checkpoint_data, checkpoint_path)
		
		
		logging.info(f"Best model saved: {checkpoint_path}")
		return checkpoint_path
	