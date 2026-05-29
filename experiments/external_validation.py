# experiments/external_validation.py

import sys
import os
import numpy as np
import tensorflow as tf

# Ensure src import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.data.loader import DataLoader
from src.data.dataset import DatasetBuilder
from src.utils.logger import Logger

from src.evaluation.metrics import Metrics
from src.evaluation.statistics import Statistics
from src.evaluation.roc import ROCAnalysis
from src.evaluation.report import ReportGenerator



def main(cfg):

    cfg.set_seed()
    logger = Logger(cfg, mode="eval")

    print("\n🌍 Running External Validation")

    
    # LOAD EXTERNAL DATA
    loader = DataLoader(cfg)
    external_data = loader.load_external_data()

    builder = DatasetBuilder(cfg)

    ext_ds = builder.build_external_multimodal(
        external_data,
        cfg.BATCH_SIZE
    )

    # Extract labels once
    y_true = external_data["labels"]

    
    # LOAD ALL FOLD MODELS
    print("\n📦 Loading fold models...")

    models = []

    for fold in range(cfg.N_SPLITS):

        model_path = os.path.join(
            cfg.OUTPUT_PATH,
            cfg.EXPERIMENT_NAME,
            "models",
            f"fold_{fold}_best.keras"
        )

        if not os.path.exists(model_path):
            print(f"⚠️ Missing model: {model_path}")
            continue

        print(f"✔ Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        models.append(model)

    # 🚨 CRITICAL SAFETY CHECK
    if len(models) == 0:
        raise RuntimeError(
            "❌ No models loaded. Check EXPERIMENT_NAME or training output path."
        )
    
    # EXTRACT DATA ONCE
    print("\n🧠 Ensembling predictions...")

    
    # GET TRUE LABELS (SAFE EXTRACTION)
    y_true = np.concatenate([y.numpy() for _, y in ext_ds], axis=0)

    # MODEL ENSEMBLE (NO RETRACING, NO MEMORY OVERLOAD)
    all_probs = []

    for model in models:
        probs = model.predict(ext_ds, verbose=0)
        all_probs.append(probs.flatten())

    all_probs = np.array(all_probs)

    # Mean ensemble
    y_prob = np.mean(all_probs, axis=0)

    # Binary prediction
    y_pred = (y_prob > 0.5).astype(int)
    
    print("\n🔍 SANITY CHECK")
    print("y_true:", len(y_true))
    print("y_prob:", len(y_prob))

    assert len(y_true) == len(y_prob), "Mismatch between labels and predictions!"
    
    # METRICS WITH CI
    print("\n📊 Computing metrics...")

    y_true = external_data["labels"]

    external_metrics = Metrics.compute_all(
        y_true,
        y_pred,
        y_prob,
        n_bootstraps=cfg.BOOTSTRAP_SAMPLES,
        alpha=cfg.CI_ALPHA
    )

    
    # SAVE RESULTS
    logger.save_external_results(
        external_metrics,
        y_true,
        y_pred,
        y_prob
    )

    pred_path = os.path.join(
    logger.paths["pred_external"],
    "external_predictions.npz"
    )

    np.savez(
        pred_path,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob
    )

    print(f"✅ Saved external predictions → {pred_path}")

    
    # ROC CURVE
    print("\n📈 Generating ROC Curve...")

    roc_path = os.path.join(logger.paths["plots"], "external_roc_ci.png")

    ROCAnalysis.plot_roc_ci(
        y_true,
        y_prob,
        roc_path,
        title="External ROC Curve (with CI)"
    )

    # PUBLICATION TABLE (AUC, Sens, Spec, CI)
    table = ReportGenerator.build_clinical_table(external_metrics)

    table_path = os.path.join(logger.paths["metrics_external"], "clinical_table.csv")

    ReportGenerator.save_table(table, table_path)

    print("\n📊 Clinical Table:")
    print(table)
    
    
    # FINAL LOG
    print("\n🌍 External Results:")
    for k, v in external_metrics.items():
        print(f"{k}: {v}")

    logger.log("✅ External validation completed successfully")


if __name__ == "__main__":
    main()