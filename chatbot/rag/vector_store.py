"""
ChromaDB vector store wrapper.
Provides a persistent collection for the medical guideline embeddings.
"""
import chromadb
from django.conf import settings
from .embedder import embed_text, embed_batch

_client = None
_collection = None

def _get_collection():
    """Lazy-initialise the ChromaDB client and collection."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.CHATBOT_CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=settings.CHATBOT_CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def add_documents(chunks: list[dict]):
    """
    Store chunks in ChromaDB.
    Each chunk: {"id": str, "text": str, "source": str, "page": int}
    """
    collection = _get_collection()
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c.get("page", 0)} for c in chunks]
    embeddings = embed_batch(texts)
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"[VectorStore] Upserted {len(chunks)} chunks.")

def query(text: str, n_results: int = 5) -> list[dict]:
    """
    Query the collection and return top-N results.
    Returns list of {text, source, page, distance}.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []
    embedding = embed_text(text)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist})
    return hits

def collection_count() -> int:
    """Return the number of chunks stored in ChromaDB."""
    return _get_collection().count()
