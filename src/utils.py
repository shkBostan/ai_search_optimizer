import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# Load model, index, metadata once at startup in the API
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("index.faiss")
with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)

def semantic_search(query, top_k=50):
    """
    Return top_k candidate doc indices from FAISS for a query.
    """
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    return I[0].tolist(), D[0].tolist()

def bm25_scores(query, bm25, tokenized):
    """
    Return BM25 scores for the query.
    """
    q_tokens = query.split()
    scores = bm25.get_scores(q_tokens)
    return scores