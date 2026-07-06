import os
import logging
from typing import List, Dict, Any
from functools import lru_cache
import chromadb
import numpy as np

from .locks import _local_pipeline_lock

logger = logging.getLogger(__name__)

# Global model registry to prevent reloading models in memory
_MODEL_REGISTRY = {}

def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Load a sentence transformer model with fallback logic if the requested model fails to download"""
    if model_name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_name]
        
    try:
        logger.info(f"⏳ Loading embedding model '{model_name}' (Downloading weights from Hugging Face on first boot)...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        logger.info(f"✅ Embedding model '{model_name}' successfully loaded!")
        _MODEL_REGISTRY[model_name] = model
        return model
    except Exception as e:
        logger.warning(f"⚠️ Warning: Could not load requested embedding model '{model_name}': {e}")
        
        # Fallback to standard lightweight model if specialized model fails
        fallback_model = "all-MiniLM-L6-v2"
        if fallback_model in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[fallback_model]
            
        try:
            logger.info(f"🔄 Falling back to standard lightweight embedding model '{fallback_model}'...")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(fallback_model)
            logger.info(f"✅ Fallback embedding model '{fallback_model}' loaded successfully!")
            _MODEL_REGISTRY[fallback_model] = model
            return model
        except Exception as ex:
            logger.critical(f"❌ Critical: Could not initialize fallback embedding model: {ex}")
            return None

@lru_cache(maxsize=1024)
def get_cached_embedding(text: str, model_name: str) -> List[float]:
    """Retrieve vector representations using an LRU cache to reduce CPU latency to 0ms for recurrent queries"""
    model = get_sentence_transformer(model_name)
    if model is None:
        raise RuntimeError("Embedding model initialization failed.")
    
    # Perform single-sentence encoding and serialize output list under the shared CPU execution lock
    with _local_pipeline_lock:
        embedding = model.encode([text])[0]
    return embedding.tolist()


class EmbeddingStore:
    def __init__(self, model_name: str = None, persist_path: str = "./chroma_db"):
        # Detect model from environment variable, default to PubMedBERT for high clinical precision
        self.model_name = model_name or os.environ.get('EMBEDDING_MODEL_NAME', 'NeuML/pubmedbert-base-embeddings')
        
        # Verify model loads successfully
        model = get_sentence_transformer(self.model_name)
        if model is None:
            raise RuntimeError("Could not initialize any SentenceTransformer model.")
            
        # Update model name if fallback was loaded under the hood
        if self.model_name in _MODEL_REGISTRY:
            pass
        elif "all-MiniLM-L6-v2" in _MODEL_REGISTRY:
            self.model_name = "all-MiniLM-L6-v2"
            
        self.chunk_ids = []
        self.persist_path = persist_path

        # Use PersistentClient for durable storage (modern ChromaDB API)
        try:
            self.client = chromadb.PersistentClient(path=persist_path)
        except Exception:
            # Fallback to ephemeral client if persistent client fails
            self.client = chromadb.Client()
        
        # Generate a unique, valid collection name based on the model name
        import re
        safe_model_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.model_name).lower()
        collection_name = f"docs_{safe_model_name}"[:63].strip('_')
        
        logger.info(f"Using ChromaDB collection: '{collection_name}' for model '{self.model_name}'")
        
        # Fetch or create collection thread-safely
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def add_chunks(self, chunks: List[str]):
        if not chunks:
            return
            
        embeddings = []
        for chunk in chunks:
            try:
                emb = get_cached_embedding(chunk, self.model_name)
                embeddings.append(emb)
            except Exception as e:
                # Direct encoding fallback
                model = get_sentence_transformer(self.model_name)
                emb = model.encode([chunk])[0].tolist()
                embeddings.append(emb)
                
        ids = [f"chunk_{len(self.chunk_ids)+i+1}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        self.chunk_ids.extend(ids)
        logger.info(f"💾 Added {len(chunks)} text chunks to ChromaDB (Total active IDs: {len(self.chunk_ids)})")

    def query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        try:
            q_emb = get_cached_embedding(query, self.model_name)
        except Exception:
            model = get_sentence_transformer(self.model_name)
            q_emb = model.encode([query])[0].tolist()
        
        # Ensure we don't request more results than exist in vector store
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
            similarities = [max(0.0, 1.0 - dist) for dist in distances]
        else:
            similarities = []
        
        return {
            'documents': documents,
            'similarities': similarities
        }

    def clear(self):
        """Clear all data and reset the collection safely"""
        import re
        safe_model_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.model_name).lower()
        collection_name = f"docs_{safe_model_name}"[:63].strip('_')
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
            
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.chunk_ids = []
        logger.info("🗑️ Collection cleared and re-initialized successfully!")
