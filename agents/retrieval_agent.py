from dataclasses import dataclass
from typing import List, Dict, Any
from .mcp import MCPMessage
from .embedding_utils import EmbeddingStore

@dataclass
class RetrievalAgent:
    embedder: Any = None
    vector_store: Any = None

    def embed_and_retrieve(self, chunks: List[str], query: str, top_k: int = 3) -> MCPMessage:
        if self.vector_store is None:
            self.vector_store = EmbeddingStore()
            self.vector_store.add_chunks(chunks)
        top_chunks = self.vector_store.query(query, top_k)
        return MCPMessage(
            sender="RetrievalAgent",
            receiver="LLMResponseAgent",
            type="RETRIEVAL_RESULT",
            payload={"retrieved_context": top_chunks, "query": query}
        )
