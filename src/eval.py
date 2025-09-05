from sklearn.metrics import ndcg_score
import numpy as np

def precision_at_k(retrieved_ids, relevant_ids, k):
    topk = retrieved_ids[:k]
    return len(set(topk) & set(relevant_ids)) / k

def compute_ndcg(y_true_relevance, y_score, k=10):
    """
    y_true_relevance: array-like of shape (n_samples, n_labels)
    y_score: predicted scores (same shape)
    """
    return ndcg_score(y_true_relevance, y_score, k=k)