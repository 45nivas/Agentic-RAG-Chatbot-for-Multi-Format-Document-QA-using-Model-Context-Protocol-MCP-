from dataclasses import dataclass
from typing import List, Dict, Any
from .mcp import MCPMessage
from .embedding_utils import EmbeddingStore

@dataclass
class RetrievalAgent:
    embedder: Any = None
    vector_store: Any = None

    def embed_and_retrieve(self, chunks: List[str], query: str, top_k: int = 3, similarity_threshold: float = 0.15) -> MCPMessage:
        try:
            if self.vector_store is None:
                self.vector_store = EmbeddingStore()
                self.vector_store.add_chunks(chunks)
            
            query_results = self.vector_store.query(query, top_k)
            documents = query_results['documents']
            similarities = query_results['similarities']
            
            # Check if highest similarity meets threshold
            max_similarity = max(similarities) if similarities else 0.0
            
            # Debug logging
            print(f"DEBUG - Query: '{query}'")
            print(f"DEBUG - Max similarity: {max_similarity:.3f}")
            print(f"DEBUG - All similarities: {[f'{s:.3f}' for s in similarities]}")
            print(f"DEBUG - Threshold: {similarity_threshold}")
            print(f"DEBUG - Threshold met: {max_similarity >= similarity_threshold}")
            if documents:
                print(f"DEBUG - First document chunk preview: {documents[0][:100]}...")
            
            return MCPMessage(
                sender="RetrievalAgent",
                receiver="LLMResponseAgent",
                type="RETRIEVAL_RESULT",
                payload={
                    "retrieved_context": documents, 
                    "query": query,
                    "similarities": similarities,
                    "max_similarity": max_similarity,
                    "threshold_met": max_similarity >= similarity_threshold
                }
            )
        except Exception as e:
            print(f"ERROR in RetrievalAgent: {e}")
            # Fallback to simple keyword matching
            query_words = query.lower().split()
            scored_chunks = []
            
            for chunk in chunks:
                chunk_lower = chunk.lower()
                score = sum(1 for word in query_words if word in chunk_lower)
                if score > 0:
                    scored_chunks.append((score, chunk))
            
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            documents = [chunk for score, chunk in scored_chunks[:top_k]]
            max_similarity = 0.5 if documents else 0.0  # Assume decent match for keyword
            
            return MCPMessage(
                sender="RetrievalAgent",
                receiver="LLMResponseAgent", 
                type="RETRIEVAL_RESULT",
                payload={
                    "retrieved_context": documents,
                    "query": query,
                    "similarities": [0.5] * len(documents),
                    "max_similarity": max_similarity,
                    "threshold_met": max_similarity >= similarity_threshold,
                    "fallback_mode": True
                }
            )
