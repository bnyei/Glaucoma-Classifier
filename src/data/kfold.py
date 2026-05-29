# src/data/kfold.py

import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple, Dict
from src.config import Config


class KFoldManager:
    """
    Handles K-Fold splitting with:
    ✔ Stratification
    ✔ Alignment safety
    ✔ Traceability (filenames)
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.skf = StratifiedKFold(
            n_splits=cfg.N_SPLITS,
            shuffle=cfg.SHUFFLE,
            random_state=cfg.SEED
        )

    
    # GENERATE SPLITS
    
    def get_splits(self, y):
        """
        Generate K-fold splits based on labels.

        Parameters:
            y: np.ndarray (labels)

        Returns:
            List of (train_idx, val_idx)
        """

        splits = []

        for train_idx, val_idx in self.skf.split(np.zeros(len(y)), y):
            splits.append((train_idx, val_idx))

        return splits

    
    # VALIDATION CHECK (CRITICAL)
    
    def validate_splits(self, dataset: Dict, splits: List[Dict]):
        """
        Ensures:
        - No overlap between train/val
        - Full coverage of dataset
        """

        n_samples = len(dataset["labels"])
        all_val_indices = []

        for split in splits:
            train_idx = split["train_idx"]
            val_idx = split["val_idx"]

            # Check overlap
            if len(set(train_idx) & set(val_idx)) > 0:
                raise ValueError("❌ Overlap detected between train and validation!")

            all_val_indices.extend(val_idx)

        # Check full coverage
        if set(all_val_indices) != set(range(n_samples)):
            raise ValueError("❌ Some samples missing in validation folds!")

        print("✅ K-Fold validation checks passed")

    
    # SUMMARIZE SPLITS
    
    def summarize_splits(self, splits, y):
        """
        Print class distribution per fold.

        Parameters:
            splits: list of (train_idx, val_idx)
            y: labels array
        """

        print("\n📊 K-FOLD DISTRIBUTION SUMMARY\n")

        for i, (train_idx, val_idx) in enumerate(splits):
            y_train = y[train_idx]
            y_val = y[val_idx]

            print(f"Fold {i+1}")
            print(f" Train size: {len(train_idx)}, Class balance: {np.bincount(y_train)}")
            print(f" Val size:   {len(val_idx)}, Class balance: {np.bincount(y_val)}")
            print("-" * 40)

    
    # OPTIONAL: GET FILENAMES PER FOLD
    
    def get_fold_filenames(self, dataset: Dict, split: Dict):
        """
        Useful for:
        - debugging
        - audit trails (VERY important clinically)
        """

        filenames = dataset["filenames"]

        train_files = [filenames[i] for i in split["train_idx"]]
        val_files = [filenames[i] for i in split["val_idx"]]

        return train_files, val_files