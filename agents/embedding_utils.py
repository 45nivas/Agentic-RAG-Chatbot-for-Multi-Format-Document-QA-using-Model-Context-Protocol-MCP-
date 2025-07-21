from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
import time

class EmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.Client()
        
        collection_name = f"rag_docs_{int(time.time())}"
        try:
            self.collection = self.chroma_client.create_collection(name=collection_name)
        except:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        self.chunk_ids = []

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
        return results['documents'][0] if results['documents'] else []
