"""
LLMResponseAgent: Constructs prompt from retrieved context and queries LLM.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
from .mcp import MCPMessage
from .llm_utils import LLMClient

@dataclass
class LLMResponseAgent:
    llm: Any = None

    def generate_response(self, retrieved_context: List[str], query: str) -> MCPMessage:
        """
        Constructs prompt and queries LLM, returns MCPMessage with answer.
        """
        if self.llm is None:
            self.llm = LLMClient()
        answer = self.llm.ask(retrieved_context, query)
        return MCPMessage(
            sender="LLMResponseAgent",
            receiver="UI",
            type="LLM_ANSWER",
            payload={"answer": answer, "source_context": retrieved_context, "query": query}
        )

# Example MCP message
# {
#   "sender": "LLMResponseAgent",
#   "receiver": "UI",
#   "type": "LLM_ANSWER",
#   "trace_id": "...",
#   "payload": {"answer": "...", "source_context": ["..."], "query": "..."}
# }
