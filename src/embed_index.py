"""
Created on Sep, 2025
Author: s Bostan

Description:
    This module builds dense embeddings for a collection of documents using
    SentenceTransformer models and creates a FAISS index for fast semantic search.
    Also persists metadata (id/title mapping) for later retrieval.

Licensed under the Apache License 2.0.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from tqdm import tqdm
from logger_config import logger

MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight and good for semantic search

# ----------------------------
# Functions
# ----------------------------
def build_embeddings(docs, model_name=MODEL_NAME, batch_size=64):
    """
    Encode documents to dense vectors using sentence-transformers.
    """
    try:
        logger.info(f"Loading SentenceTransformer model '{model_name}'...")
        model = SentenceTransformer(model_name)
        logger.info(f"Encoding {len(docs)} documents with batch_size={batch_size}...")
        embeddings = model.encode(
            docs, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        logger.info(f"Embeddings built: shape={embeddings.shape}")
        return embeddings, model
    except Exception as e:
        logger.error(f"Error building embeddings: {e}")
        raise

def build_faiss_index(embeddings, index_path="index.faiss"):
    """
    Build a FAISS index (normalized inner-product for cosine similarity) and save it.
    """
    try:
        logger.info("Normalizing embeddings for cosine similarity...")
        faiss.normalize_L2(embeddings)
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # exact index
        index.add(embeddings)
        faiss.write_index(index, index_path)
        logger.info(f"FAISS index built and saved to '{Path(index_path).resolve()}'")
        return index
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        raise

def save_metadata(meta, path="meta.pkl"):
    """
    Persist metadata (id/title mapping) to a pickle file.
    """
    try:
        with open(path, "wb") as f:
            pickle.dump(meta, f)
        logger.info(f"Metadata saved to '{Path(path).resolve()}'")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        raise

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    try:
        from ingest import load_data
        data_path = r"E:\pythonProject-github-public\ai_serach_optimizer\ai_search_optimizer\data\articles.csv"
        logger.info(f"Loading documents from '{data_path}'...")
        docs, meta = load_data(data_path)
        logger.info(f"{len(docs)} documents loaded.")

        embeddings, model = build_embeddings(docs)
        build_faiss_index(embeddings)
        save_metadata(meta)

        logger.info("FAISS index and metadata creation completed successfully.")
    except Exception as e:
        logger.error(f"embed_index.py execution failed: {e}")
