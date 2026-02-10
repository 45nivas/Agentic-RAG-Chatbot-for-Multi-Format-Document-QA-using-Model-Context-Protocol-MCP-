from typing import List
import numpy as np
import chromadb
import time

# Lazy import to avoid circular dependency issues
def get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Warning: Could not load SentenceTransformer: {e}")
        return None


class EmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_path: str = "./chroma_db"):
        self.model = get_sentence_transformer()
        if self.model is None:
            raise RuntimeError("Could not initialize SentenceTransformer model")
        
        self.chunk_ids = []
        self.persist_path = persist_path

        # Use PersistentClient for durable storage (modern ChromaDB API)
        try:
            self.client = chromadb.PersistentClient(path=persist_path)
        except Exception:
            # Fallback to ephemeral client if persistent fails
            self.client = chromadb.Client()
        
        try:
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            self.collection = self.client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )

    def add_chunks(self, chunks: List[str]):
        if not chunks:
            return
        embeddings = self.model.encode(chunks)
        ids = [f"chunk_{len(self.chunk_ids)+i+1}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, embeddings=[emb.tolist() for emb in embeddings], ids=ids)
        self.chunk_ids.extend(ids)

    def query(self, query: str, top_k: int = 3):
        q_emb = self.model.encode([query])[0].tolist()
        
        # Ensure we don't request more results than exist
        count = self.collection.count()
        if count == 0:
            return {'documents': [], 'similarities': []}
        n = min(top_k, count)
        
        results = self.collection.query(query_embeddings=[q_emb], n_results=n)
        
        # Return both documents and distances (similarity scores)
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        # Convert cosine distance to cosine similarity: similarity = 1 - distance
        if distances:
            similarities = [max(0, 1 - dist) for dist in distances]
        else:
            similarities = []
        
        return {
            'documents': documents,
            'similarities': similarities
        }

    def clear(self):
        """Clear all data and reset the collection"""
        try:
            self.client.delete_collection("documents")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.chunk_ids = []
