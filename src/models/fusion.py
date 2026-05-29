# src/models/fusion.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from src.models.backbone import BackboneFactory
from src.config import Config


class FusionModels:
    """
    Multimodal fusion architectures:

        Model 1: Modality-specific backbones
        Model 2: Shared multimodal input → ALL backbones
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.factory = BackboneFactory(cfg)

    # CLASSIFIER HEAD
    def _classifier(self, x):
        x = layers.BatchNormalization()(x)

        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        output = layers.Dense(1, activation="sigmoid")(x)

        return output

    # MODEL 1: MODALITY-SPECIFIC
    def build_model_1(self):

        inputs = []
        features = []

        for i, modality in enumerate(self.cfg.MODALITIES):
            inp = layers.Input(
                shape=(*self.cfg.IMAGE_SIZE, self.cfg.CHANNELS),
                name=f"{modality}_input"
            )
            inputs.append(inp)

            backbone_name = self.cfg.BACKBONES[i % len(self.cfg.BACKBONES)]
            backbone = self.factory.get_backbone(backbone_name)

            feat = backbone(inp)
            feat = layers.BatchNormalization(name=f"{modality}_bn")(feat)

            features.append(feat)

        x = layers.Concatenate(name="fusion_concat")(features)
        x = layers.BatchNormalization(name="fusion_bn")(x)

        output = self._classifier(x)

        return Model(inputs=inputs, outputs=output, name="Fusion_Model_1")

    # MODEL 2: SHARED INPUT → ALL BACKBONES (NEW)
    def build_model_2(self):
        """
        All modalities are fused FIRST,
        then passed through ALL backbones independently.
        """

        inputs = []

        for modality in self.cfg.MODALITIES:
            inp = layers.Input(
                shape=(*self.cfg.IMAGE_SIZE, self.cfg.CHANNELS),
                name=f"{modality}_input"
            )
            inputs.append(inp)

        # Step 1: Concatenate modalities
        x = layers.Concatenate(axis=-1, name="channel_concat")(inputs)

        # Step 2: Reduce channels back to 3
        x = layers.Conv2D(
            filters=3,
            kernel_size=(1, 1),
            activation="relu",
            name="channel_reduction"
        )(x)

        # Step 3: Pass through ALL backbones
        features = []

        for backbone_name in self.cfg.BACKBONES:
            backbone = self.factory.get_backbone(backbone_name)

            feat = backbone(x)
            feat = layers.BatchNormalization(name=f"{backbone_name}_bn")(feat)

            features.append(feat)

        # Step 4: Fuse backbone outputs
        x = layers.Concatenate(name="fusion_concat")(features)
        x = layers.BatchNormalization(name="fusion_bn")(x)

        # Step 5: Classifier
        output = self._classifier(x)

        return Model(inputs=inputs, outputs=output, name="Fusion_Model_2")

    # GET MODEL
    def get_model(self):

        if not self.cfg.USE_FUSION:
            raise ValueError("Fusion disabled in config")

        if self.cfg.FUSION_STRATEGY == "model_1":
            return self.build_model_1()

        elif self.cfg.FUSION_STRATEGY == "model_2":
            return self.build_model_2()

        else:
            raise ValueError("Invalid fusion strategy")