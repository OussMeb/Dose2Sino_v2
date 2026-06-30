
class Config:
    """Configuration class to hold all training parameters."""

    # Paths
    DATA_PATH = "/mnt/data/tomo_data/"
    CHECKPOINT_DIR = "./checkpoints"
    CACHE_DIR = DATA_PATH + "cache"  # Directory for cached preprocessed data
    USE_CACHE = True  # Whether to use cached preprocessed data (speeds up loading but uses more disk space)
    
    # Channels
    ADD_OPTIMIZATION_PARAMETERS_CHANNEL = False

    # Dataset splits
    TRAIN_SIZE = 0.8
    VALIDATION_SIZE = 0.1
    TEST_SIZE = 0.1

    # Training parameters
    LEARNING_RATE = 1e-4
    EPOCHS = 200
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    AUTO_STOP = 5
    MAX_DOSE = 70.0  # Gy, used for normalization #keep in float for division in patient.py
    BASE_FILTERS = 16
    REDUCTION_RATIO = 2  # Ratio de réduction pour le dataset
    CO_JSON_REMOVAL = [
        "External - PTVs",
    ]  
    # remove external-ptvs ( de base evite les hot spot mais la loss le fera d'elle meme ( encore plus si ca tombe sur un oar)) \
    # same pour ring et ring10, en dehors des ptvs la loss ( mse ) force la descente 
    # Mixed precision
    USE_MIXED_PRECISION = True

    # Loss weights
    PTV_WEIGHT = 2.0
    OAR_WEIGHT = 1.5
    MIN_LOSS_THRESHOLD = 1e-6

    # Augmentation
    FLIP_PROB = 0.3
    ROTATE_PROB = 0.3
    ZOOM_PROB = 0.3
    NO_PTV_PROB = 0
    NO_PTV_AND_OAR_PROB = 0

    # Scheduler
    LR_REDUCTION_FACTOR = 0.9
    LR_PATIENCE = 3

    # Debug
    VIEWER_MODE = False

    # Resume training
    RESUME = False

    JOURNAL = ""  # str to set in journal.txt

    RESET_VALIDATION_LOSS = (
        False  # Reset validation loss on resume, useful when switching training modes
    )

    # ---- Hybrid-GAN hyperparameters ----------------------------------------
    # Generator warm-start LR (very low to preserve supervised pre-training)
    G_LR: float = 1e-5
    # Discriminator LR (higher — needs to catch up with the generator)
    D_LR: float = 1e-4

    # Generator total loss: λ_L1 * L1 + λ_adv * Adv + λ_clinical * Clinical
    LAMBDA_L1:       float = 10.0   # Strong supervision anchor for stability
    LAMBDA_ADV:      float = 1.0    # Adversarial term weight
    LAMBDA_CLINICAL: float = 5.0    # Clinical DVH constraint weight

    # DifferentialClinicalLoss sub-weights
    LAMBDA_PTV: float = 1.0    # PTV coverage + uniformity
    LAMBDA_OAR: float = 1.0    # OAR hinge (sparing)
    LAMBDA_TV:  float = 0.01   # Total Variation (smoothness / deliverability)

    # Discriminator architecture
    D_BASE_FILTERS: int = 32   # Keep low for 3D memory budget

    # Label smoothing — real labels = LABEL_SMOOTHING (< 1.0) to prevent D from
    # becoming overconfident and dominating G too early
    LABEL_SMOOTHING: float = 0.9

    # Run external dicompyler-core DVH validation every N epochs
    DVH_VALIDATION_INTERVAL: int = 5

    def __post_init__(self):
        assert abs(self.TRAIN_SIZE + self.VALIDATION_SIZE + self.TEST_SIZE - 1.0) < 1e-6

    def __str__(self):
        return "".join(
            [
                f"DATA_PATH: {self.DATA_PATH}\n",
                f"CHECKPOINT_DIR: {self.CHECKPOINT_DIR}\n",
                f"TRAIN_SIZE: {self.TRAIN_SIZE}\n",
                f"VALIDATION_SIZE: {self.VALIDATION_SIZE}\n",
                f"TEST_SIZE: {self.TEST_SIZE}\n",
                f"LEARNING_RATE: {self.LEARNING_RATE}\n",
                f"EPOCHS: {self.EPOCHS}\n",
                f"BATCH_SIZE: {self.BATCH_SIZE}\n",
                f"NUM_WORKERS: {self.NUM_WORKERS}\n",
                f"AUTO_STOP: {self.AUTO_STOP}\n",
                f"MAX_DOSE: {self.MAX_DOSE}\n",
                f"BASE_FILTERS: {self.BASE_FILTERS}\n",
                f"REDUCTION_RATIO: {self.REDUCTION_RATIO}\n",
                f"USE_MIXED_PRECISION: {self.USE_MIXED_PRECISION}\n",
                f"PTV_WEIGHT: {self.PTV_WEIGHT}\n",
                f"OAR_WEIGHT: {self.OAR_WEIGHT}\n",
                f"MIN_LOSS_THRESHOLD: {self.MIN_LOSS_THRESHOLD}\n",
                f"FLIP_PROB: {self.FLIP_PROB}\n",
                f"ROTATE_PROB: {self.ROTATE_PROB}\n",
                f"ZOOM_PROB: {self.ZOOM_PROB}\n",
                f"NO_PTV_PROB: {self.NO_PTV_PROB}\n",
                f"NO_PTV_AND_OAR_PROB: {self.NO_PTV_AND_OAR_PROB}\n",
                f"LR_REDUCTION_FACTOR: {self.LR_REDUCTION_FACTOR}\n",
                f"LR_PATIENCE: {self.LR_PATIENCE}\n",
                f"VIEWER_MODE: {self.VIEWER_MODE}\n",
                f"G_LR: {self.G_LR}\n",
                f"D_LR: {self.D_LR}\n",
                f"LAMBDA_L1: {self.LAMBDA_L1}\n",
                f"LAMBDA_ADV: {self.LAMBDA_ADV}\n",
                f"LAMBDA_CLINICAL: {self.LAMBDA_CLINICAL}\n",
                f"LAMBDA_PTV: {self.LAMBDA_PTV}\n",
                f"LAMBDA_OAR: {self.LAMBDA_OAR}\n",
                f"LAMBDA_TV: {self.LAMBDA_TV}\n",
                f"D_BASE_FILTERS: {self.D_BASE_FILTERS}\n",
                f"LABEL_SMOOTHING: {self.LABEL_SMOOTHING}\n",
                f"DVH_VALIDATION_INTERVAL: {self.DVH_VALIDATION_INTERVAL}\n",
            ]
        )
