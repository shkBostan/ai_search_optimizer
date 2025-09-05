"""
Created on Sep, 2025
Author: s Bostan

Description:
    Script to sequentially run the full pipeline for AI Search Optimizer:
        1. Load and preprocess documents
        2. Build BM25 index
        3. Encode documents with semantic embeddings and build FAISS index
        4. Persist metadata and indexes
    Intended as a single entry-point to initialize all necessary components.
"""

import os
import subprocess

def run_step(name, command):
    print("\n" + "="*50)
    print(f"STEP: {name}")
    print("="*50 + "\n")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"ERROR in step '{name}'! Exiting.")
        exit(1)

def main():
    # Step 1: Load data
    run_step("Load data (ingest.py)", "python src/ingest.py")

    # Step 2: Build embeddings + FAISS index
    run_step("Build embeddings + FAISS index (embed_index.py)", "python src/embed_index.py")

    # Step 3: Build BM25 index
    run_step("Build BM25 index (bm25_index.py)", "python src/bm25_index.py")

    # Step 4: Run search API (Flask)
    print("\n" + "="*50)
    print("STEP: Run search API (Flask)")
    print("="*50 + "\n")
    print("The API server will start now. Press CTRL+C to stop.")
    os.system("python src/search_api.py")  # blocking call

if __name__ == "__main__":
    main()