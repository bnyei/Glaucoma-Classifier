# src/models/backbone.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from src.config import Config


class BackboneFactory:

    def __init__(self, cfg):
        self.cfg = cfg

    
    # FREEZE / FINE-TUNE CONTROL
    
    def _configure_trainability(self, base_model):

        # Freeze everything first
        for layer in base_model.layers:
            layer.trainable = False

        # Optional fine-tuning
        if self.cfg.FINE_TUNE_AT is not None:
            for layer in base_model.layers[self.cfg.FINE_TUNE_AT:]:
                layer.trainable = True

    
    # WRAP MODEL
    
    def _wrap_model(self, base_model, preprocess_fn):
        """
        Adds preprocessing + pooling
        """

        self._configure_trainability(base_model)

        inputs = layers.Input(shape=(*self.cfg.IMAGE_SIZE, self.cfg.CHANNELS))

        # Correct preprocessing INSIDE model
        x = preprocess_fn(inputs)

        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)

        return Model(inputs=inputs, outputs=x)

    
    # DENSENET201
    
    def build_densenet201(self):
        base = tf.keras.applications.DenseNet201(
            include_top=False,
            weights="imagenet",
            input_shape=(*self.cfg.IMAGE_SIZE, self.cfg.CHANNELS)
        )

        return self._wrap_model(
            base,
            tf.keras.applications.densenet.preprocess_input
        )

    
    # INCEPTIONV3
    
    def build_inceptionv3(self):
        base = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            input_shape=(*self.cfg.IMAGE_SIZE, self.cfg.CHANNELS)
        )

        return self._wrap_model(
            base,
            tf.keras.applications.inception_v3.preprocess_input
        )

    
    # NASNET MOBILE
    
    def build_nasnetmobile(self):
        base = tf.keras.applications.NASNetMobile(
            include_top=False,
            weights="imagenet",
            input_shape=(*self.cfg.IMAGE_SIZE, self.cfg.CHANNELS)
        )

        return self._wrap_model(
            base,
            tf.keras.applications.nasnet.preprocess_input
        )

    
    # RESNET152V2
    
    def build_resnet152v2(self):
        base = tf.keras.applications.ResNet152V2(
            include_top=False,
            weights="imagenet",
            input_shape=(*self.cfg.IMAGE_SIZE, self.cfg.CHANNELS)
        )

        return self._wrap_model(
            base,
            tf.keras.applications.resnet_v2.preprocess_input
        )

    
    # GENERIC GETTER
    
    def get_backbone(self, name: str):

        name = name.lower()

        if name == "densenet201":
            return self.build_densenet201()

        elif name == "inceptionv3":
            return self.build_inceptionv3()

        elif name == "nasnetmobile":
            return self.build_nasnetmobile()

        elif name == "resnet152v2":
            return self.build_resnet152v2()

        else:
            raise ValueError(f"Unknown backbone: {name}")