# src/evaluation/comparison.py

import numpy as np
import pandas as pd
from src.evaluation.metrics import Metrics


class ModelComparison:

    @staticmethod
    def compare_models(
        y_true,
        model_outputs,
        cfg
    ):
        """
        model_outputs = {
            "Model 1": (y_pred, y_prob),
            "Model 2": (y_pred, y_prob)
        }
        """

        results = {}

        for name, (y_pred, y_prob) in model_outputs.items():

            metrics = Metrics.compute_all(
                y_true,
                y_pred,
                y_prob,
                n_bootstraps=cfg.BOOTSTRAP_SAMPLES,
                alpha=cfg.CI_ALPHA
            )

            results[name] = metrics

        return results

    
    # BUILD TABLE
    
    @staticmethod
    def build_table(results):

        rows = []

        for model_name, metrics in results.items():

            row = {"Model": model_name}

            for k, v in metrics.items():

                if isinstance(v, tuple):
                    val, low, high = v
                    row[k] = f"{val:.3f} ({low:.3f}-{high:.3f})"
                else:
                    row[k] = f"{v:.3f}"

            rows.append(row)

        df = pd.DataFrame(rows)

        return df

    @staticmethod
    def save_table(df, path):
        df.to_csv(path, index=False)