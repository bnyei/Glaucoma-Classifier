import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.evaluation.statistics import Statistics
from src.evaluation.metrics import Metrics


class ROCComparison:

    @staticmethod
    def plot_with_pvalue(
        y_true,
        y_prob_1,
        y_prob_2,
        label_1="ES 1",
        label_2="ES 2",
        save_path=None
    ):
        """
        Plot ROC curves for two models with:
        ✔ AUC + CI
        ✔ DeLong p-value annotation
        """

        
        # ROC CURVES
        
        fpr1, tpr1, _ = roc_curve(y_true, y_prob_1)
        fpr2, tpr2, _ = roc_curve(y_true, y_prob_2)

        auc1 = auc(fpr1, tpr1)
        auc2 = auc(fpr2, tpr2)

        
        # CONFIDENCE INTERVALS
        
        m1 = Metrics.compute_all(y_true, (y_prob_1 > 0.5).astype(int), y_prob_1)
        m2 = Metrics.compute_all(y_true, (y_prob_2 > 0.5).astype(int), y_prob_2)

        auc1_ci = m1["auc"]
        auc2_ci = m2["auc"]

        
        # DELONG TEST
        
        p_value = Statistics.delong_roc_test(y_true, y_prob_1, y_prob_2)
        sig = Statistics.significance_label(p_value)

        
        # PLOT
        
        plt.figure(figsize=(7, 6))

        # Model 1
        plt.plot(
            fpr1,
            tpr1,
            label=f"{label_1} (AUC = {auc1_ci[0]:.3f} [{auc1_ci[1]:.3f}-{auc1_ci[2]:.3f}])",
            linewidth=2
        )

        # Model 2
        plt.plot(
            fpr2,
            tpr2,
            label=f"{label_2} (AUC = {auc2_ci[0]:.3f} [{auc2_ci[1]:.3f}-{auc2_ci[2]:.3f}])",
            linestyle="--",
            linewidth=2
        )

        # Diagonal
        plt.plot([0, 1], [0, 1], linestyle=":", linewidth=1)

        
        # ANNOTATION (KEY PART)
        
        plt.text(
            0.6,
            0.2,
            f"DeLong p = {p_value:.4f}\n({sig})",
            fontsize=11,
            bbox=dict(boxstyle="round", alpha=0.2)
        )

        # Labels
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend(loc="lower right")

        plt.grid(alpha=0.3)

        # Save
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()