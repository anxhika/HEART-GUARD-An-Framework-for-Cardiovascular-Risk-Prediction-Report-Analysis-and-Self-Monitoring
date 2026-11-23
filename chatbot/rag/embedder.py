"""
BGE-M3 embedding singleton.
Loaded once into memory and reused for all embed calls.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def get_embedder():
    """Returns the singleton BGE-M3 SentenceTransformer instance."""
    global _model
    if _model is None:
        print("[Embedder] Loading BAAI/bge-m3 model (first time only)...")
        _model = SentenceTransformer("BAAI/bge-m3")
        print("[Embedder] Model loaded.")
    return _model

def embed_text(text: str) -> list:
    """Embed a single string and return a plain Python list."""
    model = get_embedder()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()

def embed_batch(texts: list) -> list:
    """Embed a list of strings and return a list of lists."""
    model = get_embedder()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    return vectors.tolist()
