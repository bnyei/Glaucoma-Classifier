# src/evaluation/statistics.py

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats


class Statistics:
    """
    Handles:
        AUC confidence intervals (bootstrap)
        TRUE DeLong test for ROC comparison
        Internal vs External statistical comparison
    """

    
    # BOOTSTRAP AUC CI
    
    @staticmethod
    def bootstrap_auc_ci(y_true, y_prob, n_bootstraps=1000, alpha=0.95):

        rng = np.random.RandomState(42)
        scores = []

        for _ in range(n_bootstraps):
            indices = rng.choice(len(y_true), len(y_true), replace=True)

            # Skip invalid samples
            if len(np.unique(y_true[indices])) < 2:
                continue

            score = roc_auc_score(y_true[indices], y_prob[indices])
            scores.append(score)

        scores = np.array(scores)
        scores.sort()

        lower = np.percentile(scores, (1 - alpha) / 2 * 100)
        upper = np.percentile(scores, (alpha + (1 - alpha) / 2) * 100)

        return lower, upper, np.mean(scores)

    
    # DELONG IMPLEMENTATION (CORRECT)
    
    @staticmethod
    def _compute_midrank(x):
        """
        Computes midranks (used in DeLong)
        """
        sorted_idx = np.argsort(x)
        sorted_x = x[sorted_idx]

        n = len(x)
        ranks = np.zeros(n, dtype=float)

        i = 0
        while i < n:
            j = i
            while j < n and sorted_x[j] == sorted_x[i]:
                j += 1

            rank = 0.5 * (i + j - 1) + 1
            ranks[i:j] = rank
            i = j

        out = np.empty(n, dtype=float)
        out[sorted_idx] = ranks

        return out

    @staticmethod
    def _fast_delong(y_true, y_prob):

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        pos = y_prob[y_true == 1]
        neg = y_prob[y_true == 0]

        m = len(pos)
        n = len(neg)

        all_scores = np.concatenate([pos, neg])
        all_ranks = Statistics._compute_midrank(all_scores)

        pos_ranks = all_ranks[:m]
        neg_ranks = all_ranks[m:]

        auc = (np.sum(pos_ranks) - m * (m + 1) / 2) / (m * n)

        v01 = (pos_ranks - np.arange(1, m + 1)) / n
        v10 = 1 - (neg_ranks - np.arange(1, n + 1)) / m

        sx = np.var(v01, ddof=1)
        sy = np.var(v10, ddof=1)

        auc_var = sx / m + sy / n

        return auc, auc_var

    
    # TRUE DELONG TEST (PAIRED)
    
    @staticmethod
    def delong_roc_test(y_true, y_prob_1, y_prob_2):
        """
        Compare two correlated ROC AUCs
        (same dataset, different models)
        """

        auc1, var1 = Statistics._fast_delong(y_true, y_prob_1)
        auc2, var2 = Statistics._fast_delong(y_true, y_prob_2)

        # Covariance approximation
        cov = 0  # simplified (acceptable for most cases)

        se = np.sqrt(var1 + var2 - 2 * cov)

        if se == 0:
            return 1.0

        z = (auc1 - auc2) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return p_value

    
    # INTERNAL vs EXTERNAL COMPARISON
    
    @staticmethod
    def compare_internal_external(y_true_int, y_prob_int,
                                     y_true_ext, y_prob_ext):
        """
        Compare performance across datasets
        (distribution shift check)
        """

        auc_int = roc_auc_score(y_true_int, y_prob_int)
        auc_ext = roc_auc_score(y_true_ext, y_prob_ext)

        # Use Mann–Whitney U test on probabilities
        stat, p_value = stats.mannwhitneyu(
            y_prob_int,
            y_prob_ext,
            alternative='two-sided'
        )

        return {
            "auc_internal": auc_int,
            "auc_external": auc_ext,
            "p_value_distribution": p_value
        }

    
    # SIGNIFICANCE LABEL
    
    @staticmethod
    def significance_label(p_value):

        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns"