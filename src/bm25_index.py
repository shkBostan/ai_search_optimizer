"""
Created on Sep, 2025
Author: s Bostan

Description:
    Builds and persists a BM25 index for a set of documents.
    Functions include:
        - build_bm25: tokenizes documents and constructs BM25 index
        - save_bm25: saves BM25 object and tokenized documents to a pickle file
    This module is used to support hybrid search scoring along with semantic embeddings.
"""

from rank_bm25 import BM25Okapi
import pickle
import os
from ingest import load_data
from logger_config import logger


# ----------------------------
# Functions
# ----------------------------
def build_bm25(docs):
    """
    Build BM25 index from tokenized docs.
    """
    try:
        tokenized = [doc.split() for doc in docs]
        bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built for {len(tokenized)} documents.")
        return bm25, tokenized
    except Exception as e:
        logger.error(f"Error building BM25 index: {e}")
        raise


def save_bm25(bm25, tokenized, path="bm25.pkl"):
    """
    Save BM25 object + tokenized docs to pickle file.
    """
    try:
        with open(path, "wb") as f:
            pickle.dump({'bm25': bm25, 'tokenized': tokenized}, f)
        logger.info(f"BM25 index saved to '{os.path.abspath(path)}'")
    except Exception as e:
        logger.error(f"Failed to save BM25 index to '{path}': {e}")
        raise


# ----------------------------
# Execute when run directly
# ----------------------------
if __name__ == "__main__":
    data_path = r"E:\pythonProject-github-public\ai_search_optimizer\data\articles.csv"

    # Load documents
    docs, meta = load_data(data_path)
    logger.info(f"Documents loaded: {len(docs)}")

    # Build BM25 index
    bm25, tokenized = build_bm25(docs)

    # Save BM25 index
    save_bm25(bm25, tokenized)
    logger.info("BM25 index process completed successfully.")