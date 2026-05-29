# src/evaluation/report.py

import pandas as pd
import numpy as np

class ReportGenerator:

    @staticmethod
    def compute_mean_sd(fold_metrics):
        """
        Compute mean ± SD across folds for each metric.
        """

        summary = {}

        # Get all metric keys (exclude CI keys)
        keys = [k for k in fold_metrics[0].keys() if not k.endswith("_ci")]

        for key in keys:
            values = [fm[key] for fm in fold_metrics if key in fm]
            values = np.array(values, dtype=float)

            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_sd"] = np.std(values)

        return summary

    @staticmethod
    def build_clinical_table(metrics_dict):

        table = {}

        for k, v in metrics_dict.items():

            
            # HANDLE CI (PRIMARY METRIC)
            
            if k.endswith("_ci") and isinstance(v, tuple) and len(v) == 2:
                low, high = v

                base_key = k.replace("_ci", "")
                val = metrics_dict.get(base_key, None)

                if val is not None:
                    row = f"{val:.3f} ({low:.3f} - {high:.3f})"

                    
                    # ADD MEAN ± SD
                    
                    mean_key = f"{base_key}_mean"
                    sd_key = f"{base_key}_sd"

                    if mean_key in metrics_dict and sd_key in metrics_dict:
                        mean = metrics_dict[mean_key]
                        sd = metrics_dict[sd_key]

                        row += f" | {mean:.3f} ± {sd:.3f}"

                    table[base_key] = row

        df = pd.DataFrame.from_dict(table, orient="index", columns=["Value"])

        return df

    @staticmethod
    def save_table(df, path):
        df.to_csv(path)