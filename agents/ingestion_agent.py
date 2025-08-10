from dataclasses import dataclass
from typing import List, Dict, Any
from .mcp import MCPMessage
from .document_utils import parse_document

@dataclass
class IngestionAgent:
    supported_formats: List[str] = None

    def __post_init__(self):
        self.supported_formats = ["pdf", "csv", "pptx", "docx", "txt", "md"]

    def parse_documents(self, file_paths: List[str]) -> MCPMessage:
        all_chunks = []
        for path in file_paths:
            chunks = parse_document(path)
            all_chunks.extend(chunks)
            print(f"DEBUG - Parsed {len(chunks)} chunks from {path}")
            if chunks:
                print(f"DEBUG - First chunk preview: {chunks[0][:200]}...")
        
        print(f"DEBUG - Total chunks: {len(all_chunks)}")
        return MCPMessage(
            sender="IngestionAgent",
            receiver="RetrievalAgent",
            type="CHUNKIFY_RESULT",
            payload={"chunks": all_chunks, "file_paths": file_paths}
        )
