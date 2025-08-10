from dataclasses import dataclass
from typing import List, Any
from .ingestion_agent import IngestionAgent
from .retrieval_agent import RetrievalAgent
from .llm_response_agent import LLMResponseAgent
from .mcp import MCPMessage

@dataclass
class CoordinatorAgent:
    ingestion_agent: Any = None
    retrieval_agent: Any = None
    llm_agent: Any = None

    def __post_init__(self):
        self.ingestion_agent = IngestionAgent()
        self.retrieval_agent = RetrievalAgent()
        self.llm_agent = LLMResponseAgent()

    def process(self, file_paths: List[str], query: str) -> List[MCPMessage]:
        messages = []
        try:
            ingest_msg = self.ingestion_agent.parse_documents(file_paths)
            messages.append(ingest_msg)
            chunks = ingest_msg.payload["chunks"]
            
            if not chunks:
                return messages
            
            chunks = chunks[:20]
            
            retrieval_msg = self.retrieval_agent.embed_and_retrieve(chunks, query)
            messages.append(retrieval_msg)
            context = retrieval_msg.payload["retrieved_context"]
            threshold_met = retrieval_msg.payload.get("threshold_met", True)
            max_similarity = retrieval_msg.payload.get("max_similarity", 0.0)
            
            if not context:
                return messages
            
            llm_msg = self.llm_agent.generate_response(context, query, threshold_met, max_similarity)
            messages.append(llm_msg)
            
        except Exception as e:
            error_msg = MCPMessage(
                sender="CoordinatorAgent",
                receiver="UI",
                type="ERROR",
                payload={"error": str(e), "query": query}
            )
            messages.append(error_msg)
            
        return messages
