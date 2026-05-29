        
        # src/config.py

import os
import random
import numpy as np
import tensorflow as tf


class Config:
    """
    Instance-based configuration (CRITICAL FIX).
    Ensures external overrides work correctly.
    """

    def __init__(self):

        # PATHS
        self.BASE_PATH = "/content/drive/MyDrive/mResearch"

        self.DATASET_PATH = os.path.join(self.BASE_PATH, "datasets")

        self.TRAIN_PATH = os.path.join(self.DATASET_PATH, "training")
        self.EXTERNAL_PATH = os.path.join(self.DATASET_PATH, "testing")

        self.OUTPUT_PATH = os.path.join(self.BASE_PATH, "outputs")

        # EXPERIMENT SETTINGS
        self.EXPERIMENT_NAME = "baseline_experiment"

        self.SAVE_BEST_MODEL = True
        self.SAVE_ALL_FOLDS = True
        self.SAVE_PREDICTIONS = True
        
        # TASK DEFINITION
        self.TASK_TYPE = "binary"  # "binary" or "multiclass"
        self.NUM_CLASSES = 2

        # MODEL SETTINGS
        self.FUSION_STRATEGY = "model_1"
        self.USE_FUSION = True

        self.BACKBONES = [
            "densenet201",
            "inceptionv3",
            "nasnetmobile",
            "resnet152v2"
        ]

        self.FREEZE_BACKBONE = True
        self.FINE_TUNE_AT = None

        # DATA SETTINGS
        self.MODALITIES = [
            "rnfl_dm",
            "rnfl_tm",
            "gcipl_dm",
            "gcipl_tm"
        ]

        self.IMAGE_SIZE = (224, 224)
        self.CHANNELS = 3

        # TRAINING
        self.BATCH_SIZE = 4
        self.EPOCHS = 30
        self.LEARNING_RATE = 1e-4

        # DATA AUGMENTATION
        self.USE_AUGMENTATION = True

        self.AUGMENTATION_PARAMS = {
            "flip": True,
            "rotation": 0.1,
            "zoom": 0.1,
            "contrast": 0.1
        }

        # K-FOLD
        self.N_SPLITS = 5
        self.SHUFFLE = True

        # EVALUATION
        self.BOOTSTRAP_SAMPLES = 1000
        self.CI_ALPHA = 0.95
        
        self.INTERPOLATION_POINTS = 100  # ROC

        # REPRODUCIBILITY
        self.SEED = 42
        
        # EXTERNAL VALIDATION SETTINGS
        self.RUN_EXTERNAL_VALIDATION = True

        # MODEL COMPARISON SETTINGS
        self.ENABLE_MODEL_COMPARISON = True
        self.SIGNIFICANCE_LEVEL = 0.05
        

        # CLASSIFICATION THRESHOLD
        self.THRESHOLD = 0.5  # used for sensitivity/specificity
    
    # SEED CONTRO
    def set_seed(self):
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        tf.random.set_seed(self.SEED)
    # CREATE DIR
    def create_dirs(self):
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)