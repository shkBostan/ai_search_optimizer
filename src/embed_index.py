import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight and good for semantic search

def build_embeddings(docs, model_name=MODEL_NAME, batch_size=64):
    """
    Encode documents to dense vectors using sentence-transformers.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings, model

def build_faiss_index(embeddings, index_path="index.faiss", metadata_path="meta.pkl"):
    """
    Build a FAISS index (normalized inner-product for cosine similarity).
    """
    # Normalize vectors for cosine similarity with inner product
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # exact index; switch to IVFFlat for large corpora
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")
    return index

def save_metadata(meta, path="meta.pkl"):
    """
    Persist metadata (id/title mapping).
    """
    with open(path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved metadata to {path}")

if __name__ == "__main__":
    from ingest import load_data
    docs, meta = load_data(r"E:\pythonProject-github-public\ai_search_optimizer\data\articles.csv")
    emb, model = build_embeddings(docs)
    build_faiss_index(emb)
    save_metadata(meta)