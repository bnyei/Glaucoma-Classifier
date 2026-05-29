# src/utils/logger.py

import os
import json
import csv
import numpy as np
#from datetime import datetime


class Logger:
    """
    Handles:
    ✔ Experiment tracking
    ✔ Internal (K-fold) results
    ✔ External validation results
    ✔ Predictions storage
    ✔ Metrics storage
    ✔ Future statistical outputs
    """

    def __init__(self, cfg, mode="train"):
        self.cfg = cfg

        self.cfg.create_dirs()

        if mode == "train":
#            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{cfg.EXPERIMENT_NAME}"
            
        elif mode == "eval":
            # 🔥 Use existing experiment name EXACTLY
            self.experiment_name = f"{cfg.EXPERIMENT_NAME}"

        else:
            raise ValueError("mode must be 'train' or 'eval'")
            
        self.base_path = os.path.join(cfg.OUTPUT_PATH, self.experiment_name)

            # Structured directories
        self.paths = {
            "logs": os.path.join(self.base_path, "logs"),
            "models": os.path.join(self.base_path, "models"),

            "metrics_internal": os.path.join(self.base_path, "metrics", "internal"),
            "metrics_external": os.path.join(self.base_path, "metrics", "external"),
            "metrics_comparison": os.path.join(self.base_path, "metrics", "comparison"),

            "pred_internal": os.path.join(self.base_path, "predictions", "internal"),
            "pred_external": os.path.join(self.base_path, "predictions", "external"),

            "plots": os.path.join(self.base_path, "plots")
            }

        if mode == "train":
            self._create_directories()
            self._save_config()
            
    # DIRECTORY SETUP
    def _create_directories(self):
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    # SAVE CONFIG
    def _save_config(self):
        config_path = os.path.join(self.base_path, "config.json")

        config_dict = {
            k: str(v) if not isinstance(v, (int, float, str, bool, list, dict))
            else v
            for k, v in self.cfg.__dict__.items()
        }

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    # LOG MESSAGE
    def log(self, message):
        print(f"[{self.experiment_name}] {message}")

    # MODEL PATH
    def get_model_path(self, fold=None, best=False):

        if fold is not None:
            name = f"fold_{fold}"
            if best:
                name += "_best"
        else:
            name = "final_model"

        return os.path.join(self.paths["models"], f"{name}.keras")

    # INTERNAL: SAVE FOLD RESULTS
    def save_internal_fold(self, fold, history, metrics, y_true, y_pred, y_prob):

        # History
        hist_path = os.path.join(self.paths["logs"], f"history_fold_{fold}.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=4)

        # Metrics
        metrics_path = os.path.join(self.paths["metrics_internal"], f"fold_{fold}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Predictions
        pred_path = os.path.join(self.paths["pred_internal"], f"fold_{fold}.npz")
        np.savez(pred_path, y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    # INTERNAL: SAVE SUMMARY
    def save_internal_summary(self, summary_dict):
        path = os.path.join(self.paths["metrics_internal"], "summary.json")

        with open(path, "w") as f:
            json.dump(summary_dict, f, indent=4)

    # EXTERNAL: SAVE RESULTS
    def save_external_results(self, metrics, y_true, y_pred, y_prob):

        # Metrics
        path = os.path.join(self.paths["metrics_external"], "external_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Predictions
        pred_path = os.path.join(self.paths["pred_external"], "external_predictions.npz")
        np.savez(pred_path, y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    # SAVE GENERIC METRICS (FLEXIBLE)
    def save_metrics(self, metrics_dict, filename, category="internal"):
        if category == "internal":
            base = self.paths["metrics_internal"]
        elif category == "external":
            base = self.paths["metrics_external"]
        else:
            base = self.paths["metrics_comparison"]

        path = os.path.join(base, filename)

        with open(path, "w") as f:
            json.dump(metrics_dict, f, indent=4)

    # SAVE CSV
    def save_csv(self, data_list, filename, category="internal"):

        if len(data_list) == 0:
            return

        if category == "internal":
            base = self.paths["metrics_internal"]
        elif category == "external":
            base = self.paths["metrics_external"]
        else:
            base = self.paths["metrics_comparison"]

        path = os.path.join(base, filename)

        keys = data_list[0].keys()

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data_list)