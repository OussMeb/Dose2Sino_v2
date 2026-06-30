from utils.trainer_supervised import TrainerSupervised
from utils.config import Config
from utils.losses import CharbonnierLoss
from datetime import datetime
import torch

from models.unet import DosePrediction


import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
	import os
	os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

	config = Config()
	config.DATA_PATH = "/mnt/LeGrosDisque/oussama/tomo_data/"
	config.CACHE_DIR = "/mnt/LeGrosDisque/oussama/tomo_data/cache_sino"
	config.REDUCTION_RATIO = 8 # comme ca l'image fais 64*64 soit la taille du sino, a voir pour couper le patient en f de la taille du sino
	config.USE_CACHE = True
	config.NUM_WORKERS = 5
	config.BASE_FILTERS = 16 # Essayer de baisser ? TODO

	model = DosePrediction(base_filters=config.BASE_FILTERS, in_channel=2)
	loss = CharbonnierLoss()
 
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		torch.backends.cudnn.benchmark = True

	trainer = TrainerSupervised(config=config, model=model, device=device, loss_function=loss)
	start_time = datetime.now()
	trainer.train()
	end_time = datetime.now()
	logging.info(f"Training completed in {(end_time - start_time).total_seconds() / 3600:.2f} hours.")
 
 
	trainer.test()