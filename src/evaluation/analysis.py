# src/evaluation/analysis.py

import numpy as np

from src.evaluation.metrics import Metrics
from src.evaluation.statistics import Statistics
from src.evaluation.roc import ROCAnalysis


class Analysis:
    
    # INTERNAL ANALYSIS (K-FOLD)
    
    @staticmethod
    def analyze_internal(all_fold_predictions):

        from src.evaluation.metrics import Metrics

        fold_metrics = []

        for fold_data in all_fold_predictions:

            y_true = fold_data["y_true"]
            y_pred = fold_data["y_pred"]
            y_prob = fold_data["y_prob"]

            metrics = Metrics.compute_all(
                y_true,
                y_pred,
                y_prob
            )

            fold_metrics.append(metrics)

        return fold_metrics
    
    # EXTERNAL ANALYSIS
    
    @staticmethod
    def analyze_external(y_true_ext, y_prob_ext):

        y_pred_ext = (y_prob_ext >= 0.5).astype(int)

        metrics = Metrics.compute_all(
            y_true_ext,
            y_pred_ext,
            y_prob_ext
        )

        return metrics

    
    # INTERNAL vs EXTERNAL COMPARISON
    
    @staticmethod
    def compare_internal_external(all_fold_predictions, external_results):

        # ---- INTERNAL AGGREGATION ----
        y_true_int = np.concatenate([f[0] for f in all_fold_predictions])
        y_prob_int = np.concatenate([f[2] for f in all_fold_predictions])

        y_true_ext, y_prob_ext = external_results

        # ---- AUC COMPARISON ----
        auc_int = Metrics.compute_auc(y_true_int, y_prob_int)
        auc_ext = Metrics.compute_auc(y_true_ext, y_prob_ext)

        # ---- STATISTICAL TEST ----
        p_value = Statistics.delong_roc_test(
            y_true_int,
            y_prob_int,
            y_prob_ext
        )

        significance = Statistics.significance_label(p_value)

        return {
            "auc_internal": auc_int,
            "auc_external": auc_ext,
            "p_value": p_value,
            "significance": significance
        }

    
    # GENERATE PUBLICATION TABLE
    
    @staticmethod
    def generate_results_table(internal_summary, external_metrics):

        table = {
            "Internal AUC": internal_summary["auc"]["mean"],
            "Internal Sensitivity": internal_summary["sensitivity"]["mean"],
            "Internal Specificity": internal_summary["specificity"]["mean"],

            "External AUC": external_metrics["auc"],
            "External Sensitivity": external_metrics["sensitivity"],
            "External Specificity": external_metrics["specificity"]
        }

        return table

    
    # FULL PIPELINE WRAPPER
    
    @staticmethod
    def run(all_fold_predictions, external_data):

        y_true_ext, y_prob_ext = external_data

        # ---- INTERNAL ----
        fold_metrics, internal_summary = Analysis.analyze_internal(
            all_fold_predictions
        )

        # ---- EXTERNAL ----
        external_metrics = Analysis.analyze_external(
            y_true_ext,
            y_prob_ext
        )

        # ---- COMPARISON ----
        comparison = Analysis.compare_internal_external(
            all_fold_predictions,
            (y_true_ext, y_prob_ext)
        )

        # ---- TABLE ----
        table = Analysis.generate_results_table(
            internal_summary,
            external_metrics
        )

        return {
            "fold_metrics": fold_metrics,
            "internal_summary": internal_summary,
            "external_metrics": external_metrics,
            "comparison": comparison,
            "table": table
        }