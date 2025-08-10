from typing import List
import numpy as np
import chromadb
from chromadb.config import Settings

# Lazy import to avoid circular dependency issues
def get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Warning: Could not load SentenceTransformer: {e}")
        return None
import time

class EmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = get_sentence_transformer()
        if self.model is None:
            raise RuntimeError("Could not initialize SentenceTransformer model")
        
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        try:
            self.collection = self.client.get_collection("documents")
        except:
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
        results = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        
        # Return both documents and distances (similarity scores)
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        # With cosine distance, the distance is already between 0 and 2
        # Convert cosine distance to cosine similarity: similarity = 1 - distance
        # Cosine similarity ranges from -1 to 1, but with positive embeddings it's usually 0 to 1
        if distances:
            similarities = [max(0, 1 - dist) for dist in distances]
        else:
            similarities = []
        
        return {
            'documents': documents,
            'similarities': similarities
        }
