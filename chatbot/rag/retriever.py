"""
Retrieves relevant guideline chunks from ChromaDB for a given query.
"""
from .vector_store import query, collection_count


def retrieve(user_query: str, n_results: int = 4) -> dict:
    """
    Returns:
        {
            "chunks": [str, ...],       # the actual text chunks
            "sources": [str, ...],      # unique source PDF names
            "has_results": bool,
        }
    """
    if collection_count() == 0:
        return {"chunks": [], "sources": [], "has_results": False}

    hits = query(user_query, n_results=n_results)

    # Filter out very low relevance hits (high cosine distance = low similarity)
    hits = [h for h in hits if h["distance"] < 0.75]

    if not hits:
        return {"chunks": [], "sources": [], "has_results": False}

    chunks = [h["text"] for h in hits]
    sources = list(dict.fromkeys(h["source"] for h in hits))  # deduplicated, ordered

    return {"chunks": chunks, "sources": sources, "has_results": True}
