from flask import Flask, request, jsonify
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from bm25_index import build_bm25
import numpy as np

# ----------------------------
# Load models and indexes once
# ----------------------------
app = Flask(__name__)

# Load semantic model and FAISS index
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("index.faiss")

with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)

# Load BM25
with open("bm25.pkl", "rb") as f:
    bm25_data = pickle.load(f)
bm25 = bm25_data['bm25']
tokenized = bm25_data['tokenized']

# Optional: Cross-Encoder for reranking (slow but accurate)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ----------------------------
# Helper functions
# ----------------------------
def semantic_search(query, top_k=50):
    """Return top_k candidate indices and scores from FAISS semantic search."""
    q_emb = semantic_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = faiss_index.search(q_emb, top_k)
    return I[0].tolist(), D[0].tolist()

def normalize_scores(scores):
    """Normalize numpy array scores to [0,1]."""
    arr = np.array(scores, dtype=float)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

# ----------------------------
# Search endpoint
# ----------------------------
@app.route("/search")
def search():
    """
    Query parameters:
      q: query string
      k: number of results (default=10)
      alpha: BM25 weight in hybrid scoring (default=0.3)
      rerank: if 'true', apply cross-encoder reranking
    """
    q = request.args.get("q", "")
    k = int(request.args.get("k", 10))
    alpha = float(request.args.get("alpha", 0.3))
    rerank_flag = request.args.get("rerank", "false").lower() == "true"

    if not q.strip():
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    # Step 1: Semantic search
    cand_ids, sem_scores = semantic_search(q, top_k=100)

    # Step 2: BM25 scores (all documents, but we use candidates only)
    bm25_scores_all = bm25.get_scores(q.split())

    # Step 3: Hybrid scoring
    sem_norm = normalize_scores(sem_scores)
    bm_norm = normalize_scores([bm25_scores_all[idx] for idx in cand_ids])

    results = []
    for idx_pos, doc_index in enumerate(cand_ids):
        final_score = alpha * bm_norm[idx_pos] + (1 - alpha) * sem_norm[idx_pos]
        results.append((doc_index, final_score))

    # Sort by hybrid score
    results = sorted(results, key=lambda x: x[1], reverse=True)[:k]

    # Step 4: Optional reranking with Cross-Encoder
    if rerank_flag:
        # Build pairs for cross-encoder: (query, document text)
        # Note: You need 'docs' list; easiest is to reload in memory
        from ingest import load_data
        docs, _ = load_data(r"E:\pythonProject-github-public\ai_search_optimizer\data\articles.csv")

        pairs = [(q, docs[doc_index]) for doc_index, _ in results]
        ce_scores = cross_encoder.predict(pairs)

        # Replace scores with reranked CE scores
        results = sorted(
            zip([doc_index for doc_index, _ in results], ce_scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]

    # Step 5: Build response
    out = []
    for doc_index, s in results:
        m = meta[doc_index]
        out.append({
            'id': m['id'],
            'title': m['title'],
            'score': float(s)
        })

    return jsonify(out)

# ----------------------------
# Main entry
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)