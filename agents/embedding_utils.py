from typing import List
import numpy as np
import chromadb
from chromadb.config import Settings
import uuid

# Try Sentence Transformers, fallback to TF-IDF
def get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"⚠️ SentenceTransformer not available, using TF-IDF: {e}")
        return None

class EmbeddingStore:
    def __init__(self, persist_path: str = "./chroma_advanced"):
        self.model = get_sentence_transformer()
        self.use_tfidf = self.model is None
        
        if self.use_tfidf:
            print("✅ Using TF-IDF embeddings (lightweight mode)")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=384,
                stop_words='english',
                ngram_range=(1, 2),
                sublinear_tf=True
            )
            self.all_chunks = []  # Store for TF-IDF
        else:
            print("✅ Using Sentence Transformers (384D embeddings)")
        
        # Modern ChromaDB API with persistence
        try:
            self.client = chromadb.PersistentClient(
                path=persist_path,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
        except:
            self.client = chromadb.Client(Settings(
                persist_directory=persist_path,
                anonymized_telemetry=False
            ))
        
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
        
        try:
            if self.use_tfidf:
                # TF-IDF mode
                self.all_chunks.extend(chunks)
                # Fit vectorizer on all chunks seen so far
                self.vectorizer.fit(self.all_chunks)
                embeddings = self.vectorizer.transform(chunks).toarray()
            else:
                # Sentence Transformers mode
                embeddings = self.model.encode(chunks)
            
            ids = [f"chunk_{uuid.uuid4().hex[:8]}" for _ in range(len(chunks))]
            self.collection.add(
                documents=chunks,
                embeddings=[emb.tolist() for emb in embeddings],
                ids=ids
            )
            self.chunk_ids.extend(ids)
        except Exception as e:
            print(f"Error adding chunks: {e}")

    def query(self, query_text: str, top_k: int = 3):
        try:
            if self.use_tfidf:
                # TF-IDF query
                query_embedding = self.vectorizer.transform([query_text]).toarray()[0]
            else:
                # Sentence Transformers query
                query_embedding = self.model.encode([query_text])[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results.get('distances') else [0.5] * len(documents)
            similarities = [1.0 - d for d in distances]
            
            return {
                'documents': documents,
                'similarities': similarities
            }
        except Exception as e:
            print(f"Query error: {e}")
            return {'documents': [], 'similarities': []}

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
