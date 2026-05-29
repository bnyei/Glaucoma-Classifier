# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)


class Metrics:

    
    # CORE METRICS (NO CI)
    
    @staticmethod
    def _compute_base_metrics(y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        auc = roc_auc_score(y_true, y_prob)

        return acc, sensitivity, specificity, auc

    
    # BOOTSTRAP CI ENGINE
    
    @staticmethod
    def _bootstrap_ci(y_true, y_pred, y_prob, n_bootstraps=1000, alpha=0.95):
        rng = np.random.RandomState(42)

        sens_list = []
        spec_list = []
        auc_list = []
        acc_list = []

        n = len(y_true)

        for _ in range(n_bootstraps):
            idx = rng.choice(n, n, replace=True)

            # Skip invalid samples
            acc, sens, spec, _ = Metrics._compute_base_metrics(
                y_true[idx], y_pred[idx], y_prob[idx]
            )

            acc_list.append(acc)
            sens_list.append(sens)
            spec_list.append(spec)

            if len(np.unique(y_true[idx])) >= 2:
                auc = roc_auc_score(y_true[idx], y_prob[idx])
                auc_list.append(auc)
            

        def compute_ci(arr):
            arr = np.array(arr)
            arr.sort()

            lower = np.percentile(arr, (1 - alpha) / 2 * 100)
            upper = np.percentile(arr, (alpha + (1 - alpha) / 2) * 100)

            return float(lower), float(upper)

        acc_ci = compute_ci(acc_list)
        sens_ci = compute_ci(sens_list)
        spec_ci = compute_ci(spec_list)
        auc_ci = compute_ci(auc_list)
        

        return acc_ci, sens_ci, spec_ci, auc_ci

    
    # MAIN FUNCTION (FINAL OUTPUT)
    
    @staticmethod
    def compute_all(y_true, y_pred, y_prob,
                    n_bootstraps=1000,
                    alpha=0.95):

        # Base metrics
        acc, sens, spec, auc = Metrics._compute_base_metrics(
            y_true, y_pred, y_prob
        )

        # Confidence intervals
        acc_ci, sens_ci, spec_ci, auc_ci = Metrics._bootstrap_ci(
            y_true, y_pred, y_prob,
            n_bootstraps=n_bootstraps,
            alpha=alpha
        )

        results = {
            "accuracy": float(acc),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "auc": float(auc),

            "accuracy_ci": acc_ci,          # (low, high)
            "sensitivity_ci": sens_ci,
            "specificity_ci": spec_ci,
            "auc_ci": auc_ci
        }

        return results