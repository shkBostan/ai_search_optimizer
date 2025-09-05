import pandas as pd
import re
from tqdm import tqdm

def clean_text(text):
    """
    Clean input text by removing HTML, extra spaces, and normalizing.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(path):
    """
    Load CSV and return list of documents and metadata.
    """
    df = pd.read_csv(path)
    docs = []
    meta = []
    for _, row in df.iterrows():
        doc_text = f"{row.get('title','')} . {row.get('body','')}"
        doc_text = clean_text(str(doc_text))
        docs.append(doc_text)
        meta.append({'id': row['id'], 'title': row.get('title','')})
    return docs, meta

if __name__ == "__main__":
    docs, meta = load_data(r"E:\pythonProject-github-public\ai_search_optimizer\data\articles.csv")
    print(f"Loaded {len(docs)} documents")