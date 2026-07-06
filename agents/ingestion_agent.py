import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Any
from .mcp import MCPMessage
from .document_utils import parse_document

logger = logging.getLogger(__name__)

@dataclass
class IngestionAgent:
    supported_formats: List[str] = None

    def __post_init__(self):
        self.supported_formats = ["pdf", "csv", "pptx", "docx", "txt", "md"]

    def parse_documents(self, file_paths: List[str]) -> MCPMessage:
        all_chunks = []
        failed_files = []

        for path in file_paths:
            chunks, error = parse_document(path)
            if error:
                failed_files.append({"filename": os.path.basename(path), "reason": error})
            else:
                all_chunks.extend(chunks)
                logger.debug(f"Parsed {len(chunks)} chunks from {path}")
                if chunks:
                    logger.debug(f"First chunk preview: {chunks[0][:200]}...")
        
        logger.debug(f"Total chunks: {len(all_chunks)}")
        return MCPMessage(
            sender="IngestionAgent",
            receiver="RetrievalAgent",
            type="CHUNKIFY_RESULT",
            payload={"chunks": all_chunks, "file_paths": file_paths, "failed_files": failed_files}
        )
