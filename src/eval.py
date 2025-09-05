"""
Created on Sep, 2025
Author: s Bostan

Description:
    Provides evaluation metrics for information retrieval experiments.
    Includes precision@k, NDCG (Normalized Discounted Cumulative Gain),
    and other helper functions for measuring retrieval performance.
"""

import numpy as np
from sklearn.metrics import ndcg_score
from logger_config import logger


# ----------------------------
# Functions
# ----------------------------
def precision_at_k(retrieved_ids, relevant_ids, k):
    """
    Compute precision@k.
    """
    topk = retrieved_ids[:k]
    try:
        precision = len(set(topk) & set(relevant_ids)) / k
        logger.info(f"Precision@{k}: {precision:.4f} (retrieved={len(topk)}, relevant={len(relevant_ids)})")
        return precision
    except Exception as e:
        logger.error(f"Error computing precision@{k}: {e}")
        return 0.0


def compute_ndcg(y_true_relevance, y_score, k=10):
    """
    Compute normalized discounted cumulative gain (nDCG) for predicted scores.

    Parameters:
    - y_true_relevance: array-like of shape (n_samples, n_labels)
    - y_score: predicted scores (same shape)
    - k: rank position
    """
    try:
        score = ndcg_score(y_true_relevance, y_score, k=k)
        logger.info(f"nDCG@{k}: {score:.4f}")
        return score
    except Exception as e:
        logger.error(f"Error computing nDCG@{k}: {e}")
        return 0.0


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Simple test
    retrieved = [1, 2, 3, 4, 5]
    relevant = [2, 3, 6]
    precision_at_k(retrieved, relevant, k=3)

    y_true = np.array([[3, 2, 3, 0, 1]])
    y_score = np.array([[0.5, 0.2, 0.8, 0.1, 0.4]])
    compute_ndcg(y_true, y_score, k=5)
