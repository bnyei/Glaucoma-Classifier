# src/data/preprocessing.py

import tensorflow as tf
from src.config import Config


class Preprocessing:
    """
    Handles preprocessing and augmentation pipelines.

    Key Principles:
    - Augmentation ONLY for training
    - Validation & external data remain unchanged
    - Works with TensorFlow Dataset API
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # Build augmentation pipeline
        self.augmentation = self._build_augmentation() if cfg.USE_AUGMENTATION else None

    
    # AUGMENTATION PIPELINE
    
    def _build_augmentation(self):
        layers = []

        if self.cfg.AUGMENTATION_PARAMS.get("flip", True):
            layers.append(tf.keras.layers.RandomFlip("horizontal"))

        if self.cfg.AUGMENTATION_PARAMS.get("rotation", 0) > 0:
            layers.append(
                tf.keras.layers.RandomRotation(
                    self.cfg.AUGMENTATION_PARAMS["rotation"]
                )
            )

        if self.cfg.AUGMENTATION_PARAMS.get("zoom", 0) > 0:
            layers.append(
                tf.keras.layers.RandomZoom(
                    self.cfg.AUGMENTATION_PARAMS["zoom"]
                )
            )

        if self.cfg.AUGMENTATION_PARAMS.get("contrast", 0) > 0:
            layers.append(
                tf.keras.layers.RandomContrast(
                    self.cfg.AUGMENTATION_PARAMS["contrast"]
                )
            )

        return tf.keras.Sequential(layers, name="augmentation")

    
    # SINGLE IMAGE TRANSFORMS
    
    def apply_train(self, x):
        """
        Apply augmentation to training images.
        """
        if self.augmentation is not None:
            x = self.augmentation(x, training=True)
        return x

    def apply_validation(self, x):
        """
        No augmentation.
        """
        return x

    def apply_external(self, x):
        """
        External data must remain untouched.
        """
        return x

    
    # TF.DATA MAP FUNCTIONS (SINGLE MODALITY)
    
    def train_map_fn(self, x, y):
        x = self.apply_train(x)
        return x, y

    def val_map_fn(self, x, y):
        x = self.apply_validation(x)
        return x, y

    def external_map_fn(self, x, y):
        x = self.apply_external(x)
        return x, y

    
    # MULTIMODAL MAP FUNCTIONS
    
    def multimodal_train_map_fn(self, x_tuple, y):
        """
        x_tuple = (x1, x2, x3, x4)
        """
        x_aug = tuple(self.apply_train(x) for x in x_tuple)
        return x_aug, y

    def multimodal_val_map_fn(self, x_tuple, y):
        x_out = tuple(self.apply_validation(x) for x in x_tuple)
        return x_out, y

    def multimodal_external_map_fn(self, x_tuple, y):
        x_out = tuple(self.apply_external(x) for x in x_tuple)
        return x_out, y