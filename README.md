# AI Search Optimizer

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![Flask](https://img.shields.io/badge/Flask-2.3.3-brightgreen)]()
[![FAISS](https://img.shields.io/badge/FAISS-1.7.4-orange)]()
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-brightgreen)]()
[![Cross-Encoder](https://img.shields.io/badge/Cross--Encoder-ms--marco--MiniLM--L--6--v2-orange)]()
[![BM25](https://img.shields.io/badge/BM25-rank__bm25-yellow)]()


Author: S. Bostan  
Created: Aug 2025  
Tech Stack: Python, Flask, FAISS, BM25, Sentence-Transformers, Cross-Encoder  

---

## Overview

AI Search Optimizer is a hybrid semantic search engine for text collections (e.g., articles or news). It combines dense embeddings (via FAISS and Sentence-Transformers) with BM25 ranking and optional Cross-Encoder reranking for high-precision search results.  

Features:
- Fast semantic search using FAISS
- Keyword-based baseline ranking using BM25
- Hybrid scoring (weighted combination of BM25 and semantic similarity)
- Optional reranking using a Cross-Encoder
- Logging and monitoring built-in


---

## Installation

1. Clone the repository:
git clone https://github.com/shkBostan/ai_search_optimizer.git
cd ai_search_optimizer

2. Create and activate a virtual environment:
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/macOS

3. Install dependencies:
pip install -r requirements.txt

---

## Usage

### 1. Prepare Data
Place your CSV dataset in `data/articles.csv` with columns:
- id (unique identifier)
- title
- body (text content)

### 2. Build Indexes
python src/ingest.py
python src/embed_index.py
python src/bm25_index.py

### 3. Run Flask API
python src/search_api.py

API endpoint: http://127.0.0.1:5000/search?q=query&k=10&alpha=0.3&rerank=true

Parameters:
- q: query string (required)
- k: number of results (default 10)
- alpha: BM25 weight in hybrid scoring (default 0.3)
- rerank: true or false to apply Cross-Encoder reranking

---

## Logging

- Integrated logging using logger_config
- Includes info, debug, and error levels for monitoring

---

## Evaluation

- eval.py provides precision@k and NDCG metrics
- Use to benchmark retrieval quality

---

## Docker Support

Build and run via Docker:
docker build -t ai-search-optimizer .
docker run -p 5000:5000 ai-search-optimizer

---

## License

MIT License

---

## Contact

Author: S. Bostan
GitHub: https://github.com/shkBostan
