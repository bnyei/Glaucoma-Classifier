# src/data/loader.py

import os
import cv2
import numpy as np
from typing import Dict
from src.config import Config


class DataLoader:
    """
    Multimodal loader with STRICT filename alignment.

    Assumes:
    datasets/
        training/
            rnfl_dm/glaucoma/*.png
            rnfl_tm/glaucoma/*.png
            gcipl_dm/glaucoma/*.png
            gcipl_tm/glaucoma/*.png
    """

    def __init__(self, cfg):
        self.cfg = cfg

    
    # LOAD IMAGE
    
    def _load_image(self, path):
        img = cv2.imread(path)

        if img is None:
            raise ValueError(f"Image not found: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.cfg.IMAGE_SIZE)
        img = img.astype("float32") / 255.0

        return img

    
    # GET COMMON FILENAMES
    
    def _get_common_filenames(self, base_path):
        """
        Find intersection of filenames across ALL modalities
        """

        modality_files = {}

        for modality in self.cfg.MODALITIES:
            modality_path = os.path.join(base_path, modality)

            class_files = {}

            for class_name in sorted(os.listdir(modality_path)):
                class_path = os.path.join(modality_path, class_name)

                if not os.path.isdir(class_path):
                    continue

                files = sorted(os.listdir(class_path))
                class_files[class_name] = set(files)

            modality_files[modality] = class_files

        # Find intersection across modalities per class
        common_files = {}

        classes = modality_files[self.cfg.MODALITIES[0]].keys()

        for cls in classes:
            sets = []

            for modality in self.cfg.MODALITIES:
                sets.append(modality_files[modality][cls])

            common = set.intersection(*sets)

            if len(common) == 0:
                raise ValueError(f"No common files found for class: {cls}")

            common_files[cls] = sorted(list(common))

        return common_files

    
    # LOAD DATASET
    
    def _load_dataset(self, base_path):

        print(f"\n📂 Loading dataset from: {base_path}")

        common_files = self._get_common_filenames(base_path)

        data = {m: [] for m in self.cfg.MODALITIES}
        labels = []
        filenames = []

        class_names = sorted(common_files.keys())

        for label_idx, cls in enumerate(class_names):

            file_list = common_files[cls]

            print(f"Class '{cls}': {len(file_list)} aligned samples")

            for fname in file_list:

                try:
                    for modality in self.cfg.MODALITIES:
                        path = os.path.join(base_path, modality, cls, fname)

                        if not os.path.exists(path):
                            raise ValueError(f"Missing file: {path}")

                        img = self._load_image(path)
                        data[modality].append(img)

                    labels.append(label_idx)
                    filenames.append(fname)

                except Exception as e:
                    print(f"Skipping {fname}: {e}")

        # Convert to numpy
        for m in self.cfg.MODALITIES:
            data[m] = np.array(data[m])

        labels = np.array(labels)

        print(f"\n✅ Total aligned samples: {len(labels)}")

        return {
            "data": data,
            "labels": labels,
            "filenames": filenames
        }

    
    # PUBLIC FUNCTIONS
    
    def load_training_data(self):
        print("\n📂 Loading TRAINING data...")
        return self._load_dataset(self.cfg.TRAIN_PATH)

    def load_external_data(self):
        print("\n🌍 Loading EXTERNAL data...")
        return self._load_dataset(self.cfg.EXTERNAL_PATH)