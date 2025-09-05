from rank_bm25 import BM25Okapi
import numpy as np
import pickle
import os
from ingest import load_data

# ----------------------------
# Functions
# ----------------------------
def build_bm25(docs):
    """
    Build BM25 index from tokenized docs.
    """
    tokenized = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def save_bm25(bm25, tokenized, path="bm25.pkl"):
    """
    Save BM25 object + tokenized docs to pickle file.
    """
    with open(path, "wb") as f:
        pickle.dump({'bm25': bm25, 'tokenized': tokenized}, f)
    print(f"BM25 index built for {len(tokenized)} docs and saved to '{os.path.abspath(path)}'")

# ----------------------------
# Execute when run directly
# ----------------------------
if __name__ == "__main__":
    # Load documents
    docs, meta = load_data(r"E:\pythonProject-github-public\ai_search_optimizer\data\articles.csv")

    # Build BM25
    bm25, tokenized = build_bm25(docs)

    # Save BM25
    save_bm25(bm25, tokenized)