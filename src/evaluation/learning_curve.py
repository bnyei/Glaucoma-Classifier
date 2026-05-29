import matplotlib.pyplot as plt
import numpy as np


class LearningCurve:

    @staticmethod
    def plot(history_list, save_path):

        plt.figure(figsize=(8, 6))

        
        # AVERAGE ACROSS FOLDS
        
        def aggregate(metric):
            values = [h[metric] for h in history_list if metric in h]
            min_len = min(len(v) for v in values)
            values = [v[:min_len] for v in values]
            return np.mean(values, axis=0)

        # Loss
        train_loss = aggregate("loss")
        val_loss = aggregate("val_loss")

        # AUC
        train_auc = aggregate("auc")
        val_auc = aggregate("val_auc")

        epochs = range(1, len(train_loss) + 1)

        
        # PLOTS
        
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.legend()
        plt.title("Loss Curve")

        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_auc, label="Train AUC")
        plt.plot(epochs, val_auc, label="Validation AUC")
        plt.legend()
        plt.title("AUC Curve")

        plt.xlabel("Epoch")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()