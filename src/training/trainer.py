# src/training/trainer.py

import numpy as np
import tensorflow as tf
import os

from src.data.dataset import DatasetBuilder
from src.utils.logger import Logger
from src.evaluation.metrics import Metrics
from src.evaluation.learning_curve import LearningCurve
from src.evaluation.report import ReportGenerator



class Trainer:
    """
    Handles:
    ✔ Model compilation
    ✔ K-Fold training
    ✔ Prediction extraction
    ✔ Clinical metric computation 
    ✔ Logging
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_builder = DatasetBuilder(cfg)
        self.logger = Logger(cfg)

    # COMPILE MODEL
    def compile_model(self, model):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.cfg.LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )
        return model

    # TRAIN ONE FOLD
    def train_one_fold(self, model, train_ds, val_ds, fold):

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=3
            )
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.cfg.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        return history

    # GET PREDICTIONS
    def get_predictions(self, model, dataset):
        """
        Efficient prediction extraction
        """

        # Predict probabilities
        y_prob = model.predict(dataset, verbose=0).ravel()

        # Extract labels
        y_true = np.concatenate([y for _, y in dataset], axis=0)

        # Binary predictions
        y_pred = (y_prob >= 0.5).astype(int)

        # Compute metrics (publication-grade)
        metrics_dict = Metrics.compute_all(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            n_bootstraps=self.cfg.BOOTSTRAP_SAMPLES,
            alpha=self.cfg.CI_ALPHA
        )
        
        print(f"DEBUG: unique classes in y_true → {np.unique(y_true)}")

        return y_true, y_pred, y_prob, metrics_dict

    # PRINT METRICS (CLEAN)
    def print_metrics(self, metrics_dict, prefix=""):
        for k, v in metrics_dict.items():

            # Skip helper fields
            if k.endswith("_ci") or k.endswith("_mean") or k.endswith("_sd"):
                continue

            ci_key = f"{k}_ci"
            mean_key = f"{k}_mean"
            sd_key = f"{k}_sd"

            line = f"{prefix}{k}: {v:.4f}"

            # Add CI
            if ci_key in metrics_dict:
                low, high = metrics_dict[ci_key]
                line += f" (CI {low:.4f} - {high:.4f})"

            # Add Mean ± SD
            if mean_key in metrics_dict and sd_key in metrics_dict:
                mean = metrics_dict[mean_key]
                sd = metrics_dict[sd_key]
                line += f" | CV Mean±SD: {mean:.4f} ± {sd:.4f}"

            print(line)

    # MAIN K-FOLD TRAINING LOOP
    def train_kfold(self, model_builder, data_dict, splits):

        all_fold_metrics = []
        all_fold_predictions = []
        all_histories = []

        for fold, (train_idx, val_idx) in enumerate(splits):

            print(f"\n========== FOLD {fold+1}/{len(splits)} ==========")
            self.logger.log(f"Training Fold {fold+1}")

    
            # BUILD DATASETS
    
            if self.cfg.USE_FUSION:
                train_ds = self.dataset_builder.build_multimodal(
                    data_dict, train_idx, self.cfg.BATCH_SIZE, training=True
                )
                val_ds = self.dataset_builder.build_multimodal(
                    data_dict, val_idx, self.cfg.BATCH_SIZE, training=False
                )
            else:
                modality = self.cfg.MODALITIES[0]
                x, y = data_dict[modality]

                train_ds = self.dataset_builder.build_single_modality(
                    x[train_idx], y[train_idx], self.cfg.BATCH_SIZE, training=True
                )
                val_ds = self.dataset_builder.build_single_modality(
                    x[val_idx], y[val_idx], self.cfg.BATCH_SIZE, training=False
                )

    
            # BUILD + COMPILE MODEL
    
            tf.keras.backend.clear_session()

            model = model_builder()
            model = self.compile_model(model)

    
            # TRAIN
    
            history = self.train_one_fold(model, train_ds, val_ds, fold)
            all_histories.append(history.history)

    
            # PREDICTIONS + METRICS
    
            y_true, y_pred, y_prob, metrics_dict = self.get_predictions(model, val_ds)

            print(f"Fold {fold+1} Metrics:")
            self.print_metrics(metrics_dict)

    
            # LOGGING
    
            self.logger.save_internal_fold(
                fold=fold,
                history=history.history,
                metrics=metrics_dict,
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob
            )

            # Save best model
            if self.cfg.SAVE_BEST_MODEL:
                model_path = self.logger.get_model_path(fold=fold, best=True)
                model.save(model_path)

            # Store predictions for final aggregation
            all_fold_metrics.append(metrics_dict)
            all_fold_predictions.append({
                "val_indices": val_idx,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob
            })

            tf.keras.backend.clear_session()


        # FINAL (PUBLICATION-GRADE) EVALUATION

        print("\n========== FINAL AGGREGATED RESULTS ==========")

        all_y_true = []
        all_y_prob = []

        for fold_pred in all_fold_predictions:
            all_y_true.extend(fold_pred["y_true"])
            all_y_prob.extend(fold_pred["y_prob"])

        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)
        all_y_pred = (all_y_prob >= 0.5).astype(int)

        final_metrics = Metrics.compute_all(
            all_y_true,
            all_y_pred,
            all_y_prob,
            n_bootstraps=self.cfg.BOOTSTRAP_SAMPLES,
            alpha=self.cfg.CI_ALPHA
        )
        
        mean_sd_metrics = ReportGenerator.compute_mean_sd(all_fold_metrics)

        # Merge into final metrics
        final_metrics.update(mean_sd_metrics)

        print("Final Cross-Validated Performance:")
        self.print_metrics(final_metrics)

        # Save final summary
        self.logger.save_internal_summary(final_metrics)

        self.logger.log("K-Fold training completed successfully")


        # LEARNING CURVE

        curve_path = os.path.join(
            self.logger.paths["plots"],
            "learning_curve.png"
        )

        LearningCurve.plot(all_histories, curve_path)

        print(f"📈 Learning curve saved → {curve_path}")

        return all_fold_metrics, all_fold_predictions, final_metrics