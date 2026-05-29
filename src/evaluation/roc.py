# src/evaluation/roc.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class ROCAnalysis:

    
    # BOOTSTRAP ROC WITH CI
    
    @staticmethod
    def compute_roc_with_ci(y_true, y_prob, n_bootstraps=1000, alpha=0.95):

        rng = np.random.RandomState(42)

        fpr_grid = np.linspace(0, 1, 100)

        tprs = []
        aucs = []

        for i in range(n_bootstraps):
            indices = rng.randint(0, len(y_true), len(y_true))

            if len(np.unique(y_true[indices])) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true[indices], y_prob[indices])
            interp_tpr = np.interp(fpr_grid, fpr, tpr)
            interp_tpr[0] = 0.0

            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))

        tprs = np.array(tprs)
        aucs = np.array(aucs)

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)

        lower_tpr = np.percentile(tprs, (1 - alpha) / 2 * 100, axis=0)
        upper_tpr = np.percentile(tprs, (1 + alpha) / 2 * 100, axis=0)

        mean_auc = np.mean(aucs)
        lower_auc = np.percentile(aucs, (1 - alpha) / 2 * 100)
        upper_auc = np.percentile(aucs, (1 + alpha) / 2 * 100)

        return {
            "fpr": fpr_grid,
            "mean_tpr": mean_tpr,
            "lower_tpr": lower_tpr,
            "upper_tpr": upper_tpr,
            "mean_auc": mean_auc,
            "lower_auc": lower_auc,
            "upper_auc": upper_auc
        }

    
    # PLOT ROC WITH CI BAND
    
    @staticmethod
    def plot_roc_ci(y_true, y_prob, save_path, title="ROC Curve"):

        roc_data = ROCAnalysis.compute_roc_with_ci(y_true, y_prob)

        fpr = roc_data["fpr"]
        mean_tpr = roc_data["mean_tpr"]
        lower_tpr = roc_data["lower_tpr"]
        upper_tpr = roc_data["upper_tpr"]

        mean_auc = roc_data["mean_auc"]
        lower_auc = roc_data["lower_auc"]
        upper_auc = roc_data["upper_auc"]

        plt.figure(figsize=(6, 6))

        plt.plot(
            fpr,
            mean_tpr,
            label=f"AUC = {mean_auc:.3f} ({lower_auc:.3f} - {upper_auc:.3f})"
        )

        plt.fill_between(
            fpr,
            lower_tpr,
            upper_tpr,
            alpha=0.3,
            label="95% CI"
        )

        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        
    
    # INTERNAL vs EXTERNAL ROC OVERLAY
    
    @staticmethod
    def plot_internal_vs_external_ci(
        y_true_int,
        y_prob_int,
        y_true_ext,
        y_prob_ext,
        save_path
    ):

        import matplotlib.pyplot as plt

        int_data = ROCAnalysis.compute_roc_with_ci(y_true_int, y_prob_int)
        ext_data = ROCAnalysis.compute_roc_with_ci(y_true_ext, y_prob_ext)

        plt.figure(figsize=(6, 6))

        # INTERNAL
        plt.plot(
            int_data["fpr"],
            int_data["mean_tpr"],
            label=f"Internal AUC = {int_data['mean_auc']:.3f} "
                    f"({int_data['lower_auc']:.3f}-{int_data['upper_auc']:.3f})"
        )

        plt.fill_between(
            int_data["fpr"],
            int_data["lower_tpr"],
            int_data["upper_tpr"],
            alpha=0.2
        )

        # EXTERNAL
        plt.plot(
            ext_data["fpr"],
            ext_data["mean_tpr"],
            linestyle="--",
            label=f"External AUC = {ext_data['mean_auc']:.3f} "
                    f"({ext_data['lower_auc']:.3f}-{ext_data['upper_auc']:.3f})"
        )

        plt.fill_between(
            ext_data["fpr"],
            ext_data["lower_tpr"],
            ext_data["upper_tpr"],
            alpha=0.2
        )

        plt.plot([0, 1], [0, 1], linestyle=":")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Internal vs External ROC")
        plt.legend(loc="lower right")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        
    @staticmethod
    def plot_model_comparison(
            y_true,
            model_probs_dict,
            save_path
        ):

        plt.figure(figsize=(6, 6))

        for name, y_prob in model_probs_dict.items():

            data = ROCAnalysis.compute_roc_with_ci(y_true, y_prob)

            plt.plot(
                data["fpr"],
                data["mean_tpr"],
                label=f"{name} AUC={data['mean_auc']:.3f}"
            )

        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Model Comparison ROC")
        plt.legend()

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()