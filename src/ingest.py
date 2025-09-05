"""
Created on Sep, 2025
Author: s Bostan

Description:
    Load CSV files containing articles, clean text,
    and return documents and metadata.
"""

import pandas as pd
import re
from logger_config import logger
from tqdm import tqdm

# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(text: str) -> str:
    """
    Clean input text by removing HTML tags, extra spaces, and normalizing.
    """
    try:
        text = re.sub(r'<[^>]+>', ' ', text)        # Remove HTML tags
        text = re.sub(r'\s+', ' ', text).strip()   # Normalize whitespace
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text

# ----------------------------
# Data loading
# ----------------------------
def load_data(path: str):
    """
    Load CSV file and return list of documents and metadata.
    """
    try:
        df = pd.read_csv(path)
        docs = []
        meta = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading documents"):
            doc_text = f"{row.get('title','')} . {row.get('body','')}"
            doc_text = clean_text(str(doc_text))
            docs.append(doc_text)
            meta.append({'id': row['id'], 'title': row.get('title','')})
        logger.info(f"Loaded {len(docs)} documents from '{path}'")
        return docs, meta
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load data from '{path}': {e}")
        raise

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    data_path = r"E:\pythonProject-github-public\ai_search_optimizer\data\articles.csv"
    docs, meta = load_data(data_path)
    logger.info(f"Data loading complete. Total documents: {len(docs)}")