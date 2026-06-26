# src/data/dataset.py

import tensorflow as tf
from src.config import Config
from src.data.preprocessing import Preprocessing


class DatasetBuilder:
    """
    Builds TensorFlow datasets for:
    ✔ Single modality
    ✔ Multimodal fusion
    ✔ External validation
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.preprocessing = Preprocessing(cfg)

    
    # MULTIMODAL DATASET (K-FOLD)
    
    def build_multimodal(
        self,
        dataset,
        indices,
        batch_size,
        training=True
    ):
        """
        dataset:
        {
            "data": {modality: X},
            "labels": y
        }
        """

        xs = [
            dataset["data"][modality][indices]
            for modality in self.cfg.MODALITIES
        ]

        y = dataset["labels"][indices]

        ds = tf.data.Dataset.from_tensor_slices((tuple(xs), y))

        if training:
            ds = ds.shuffle(
                buffer_size=len(y),
                seed=self.cfg.SEED,
                reshuffle_each_iteration=True
            )

        ds = ds.batch(batch_size)

        
        # PREPROCESSING (CLEAN & SAFE)
        
        if training:
            ds = ds.map(
                self.preprocessing.multimodal_train_map_fn,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            ds = ds.map(
                self.preprocessing.multimodal_val_map_fn,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    
    # SINGLE MODALITY
    
    def build_single_modality(
        self,
        dataset,
        indices,
        modality,
        batch_size,
        training=True
    ):
        x = dataset["data"][modality][indices]
        y = dataset["labels"][indices]

        ds = tf.data.Dataset.from_tensor_slices((x, y))

        if training:
            ds = ds.shuffle(
                buffer_size=len(y),
                seed=self.cfg.SEED,
                reshuffle_each_iteration=True
            )

        ds = ds.batch(batch_size)

        if training:
            ds = ds.map(
                self.preprocessing.train_map_fn,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            ds = ds.map(
                self.preprocessing.val_map_fn,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    
    # EXTERNAL MULTIMODAL DATASET
    
    def build_external_multimodal(
        self,
        dataset,
        batch_size
    ):
        xs = [
            dataset["data"][modality]
            for modality in self.cfg.MODALITIES
        ]

        y = dataset["labels"]

        ds = tf.data.Dataset.from_tensor_slices((tuple(xs), y))

        ds = ds.batch(batch_size)

        ds = ds.map(
            self.preprocessing.multimodal_external_map_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds