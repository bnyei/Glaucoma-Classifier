# experiments/run_kfold.py

import sys
import os
import numpy as np

# Ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import DataLoader
from src.data.kfold import KFoldManager
from src.models.fusion import FusionModels
from src.training.trainer import Trainer
from src.utils.logger import Logger

from src.evaluation.metrics import Metrics
from src.evaluation.analysis import Analysis
from src.evaluation.roc import ROCAnalysis
from src.evaluation.report import ReportGenerator



def main(cfg):

    cfg.set_seed()
    logger = Logger(cfg, mode="eval")

    print("\n🚀 Running K-Fold Experiment")
    print(f"Fusion Strategy: {cfg.FUSION_STRATEGY}")

    
    # LOAD DATA
    loader = DataLoader(cfg)
    data_dict = loader.load_training_data()

    
    # K-FOLD SETUP
    kfold = KFoldManager(cfg)

    y = data_dict["labels"]

    splits = kfold.get_splits(y)
    kfold.summarize_splits(splits, y)

    
    # MODEL BUILDER
    fusion = FusionModels(cfg)
    model_builder = fusion.get_model

    
    # TRAINING
    trainer = Trainer(cfg)

    fold_metrics, fold_predictions, final_metrics = trainer.train_kfold(
        model_builder,
        data_dict,
        splits
    )

    
    # AGGREGATE OUT-OF-FOLD PREDICTIONS
    print("\n📊 Aggregating predictions across folds...")

    n_samples = len(data_dict["labels"])

    # Initialize containers
    y_true_all = np.zeros(n_samples)
    y_prob_all = np.zeros(n_samples)
    y_pred_all = np.zeros(n_samples)

    # Fill using fold indices
    for fold_data in fold_predictions:
        idx = fold_data["val_indices"]

        y_true_all[idx] = fold_data["y_true"]
        y_prob_all[idx] = fold_data["y_prob"]
        y_pred_all[idx] = fold_data["y_pred"]

    # Ensure correct types
    y_true_all = y_true_all.astype(int)
    y_pred_all = y_pred_all.astype(int)

    # Save raw predictions
    pred_path = os.path.join(
        logger.paths["pred_internal"],
        "aggregated_predictions.npz"
    )

    np.savez(
        pred_path,
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_prob=y_prob_all
    )
    print(f"✔ Saved aggregated predictions to: {pred_path}")
    
    # ✅ COMPUTE INTERNAL METRICS
    internal_metrics = Metrics.compute_all(
        y_true_all,
        y_pred_all,
        y_prob_all,
        n_bootstraps=cfg.BOOTSTRAP_SAMPLES,
        alpha=cfg.CI_ALPHA
    )

    logger.save_metrics(
        internal_metrics,
        filename="internal_metrics.json",
        category="internal"
    )
    
#    print("\n📊 Final Internal Metrics:")
#    for k, v in internal_metrics.items():
#        print(k, v)

    
    # ROC CURVE
    print("\n📈 Generating ROC Curve...")

    roc_path = os.path.join(logger.paths["plots"], "internal_roc_ci.png")

    ROCAnalysis.plot_roc_ci(
        y_true_all,
        y_prob_all,
        roc_path,
        title="Internal ROC Curve (with CI)"
    )

    
    # PUBLICATION TABLE (AUC, Sens, Spec, CI)
    table = ReportGenerator.build_clinical_table(internal_metrics)

    table_path = os.path.join(logger.paths["metrics_internal"], "clinical_table.csv")

    ReportGenerator.save_table(table, table_path)

    print("\n📊 Clinical Table:")
    print(table)

#    print("\n✅ FINAL RESULTS (READY FOR PAPER):")
#    for k, v in final_metrics.items():
#        if not k.endswith("_ci"):
#            ci_key = f"{k}_ci"
#            if ci_key in final_metrics:
#                low, high = final_metrics[ci_key]
#                print(f"{k}: {v:.4f} (CI {low:.4f} - {high:.4f})")


    
    # FINAL LOG
    
    trainer.logger.log("✅ K-Fold experiment completed successfully")


if __name__ == "__main__":
    main()