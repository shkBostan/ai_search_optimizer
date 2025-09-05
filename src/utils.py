"""
Created on Sep, 2025
Author: s Bostan

Description:
    Utility functions for the AI Search Optimizer project.
    Includes I/O helpers, text normalization, cleaning, and other
    reusable functions to support data ingestion, preprocessing,
    and model indexing.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from logger_config import logger

# ----------------------------
# Load models and index once
# ----------------------------
try:
    logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Loading FAISS index from 'index.faiss'...")
    index = faiss.read_index("index.faiss")

    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    logger.info(f"Loaded metadata for {len(meta)} documents.")
except Exception as e:
    logger.error(f"Failed to initialize models or index: {e}")
    raise


# ----------------------------
# Functions
# ----------------------------
def semantic_search(query, top_k=50):
    """
    Return top_k candidate doc indices and similarity scores from FAISS for a query.
    """
    try:
        logger.info(f"Semantic search for query: '{query}' with top_k={top_k}")
        q_emb = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, top_k)
        logger.info(f"Returned {len(I[0])} candidate documents.")
        return I[0].tolist(), D[0].tolist()
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return [], []


def bm25_scores(query, bm25, tokenized):
    """
    Return BM25 scores for the query.
    """
    try:
        logger.info(f"Computing BM25 scores for query: '{query}'")
        q_tokens = query.split()
        scores = bm25.get_scores(q_tokens)
        logger.info(f"BM25 scores computed for {len(scores)} documents.")
        return scores
    except Exception as e:
        logger.error(f"BM25 scoring failed: {e}")
        return np.zeros(len(tokenized))
