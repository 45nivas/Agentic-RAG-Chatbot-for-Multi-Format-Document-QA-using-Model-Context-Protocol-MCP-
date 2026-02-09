from typing import List
import numpy as np
import chromadb
from chromadb.config import Settings
import uuid

# Lazy import to avoid circular dependency issues
def get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Warning: Could not load SentenceTransformer: {e}")
        return None

class EmbeddingStore:
    def __init__(self, persist_path: str = "./chroma_advanced"):
        self.model = get_sentence_transformer()
        if self.model is None:
            raise RuntimeError("Could not initialize SentenceTransformer model")
        
        # Modern ChromaDB API with persistence
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection("documents")
        except:
            self.collection = self.client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        
        self.chunk_ids = []

    def add_chunks(self, chunks: List[str]):
        if not chunks:
            return
        embeddings = self.model.encode(chunks)
        ids = [f"chunk_{uuid.uuid4().hex[:8]}" for _ in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            embeddings=[emb.tolist() for emb in embeddings],
            ids=ids
        )
        self.chunk_ids.extend(ids)

    def clear(self):
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection("documents")
            self.collection = self.client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            self.chunk_ids = []
        except Exception as e:
            print(f"Warning: Could not clear collection: {e}")

    def query(self, query: str, top_k: int = 3):
        q_emb = self.model.encode([query])[0].tolist()
        results = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        
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
