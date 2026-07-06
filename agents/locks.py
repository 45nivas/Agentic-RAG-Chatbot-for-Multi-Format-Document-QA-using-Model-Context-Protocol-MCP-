import threading

# Shared lock to serialize CPU-bound PyTorch inference across agents (Qwen) and RAG embeddings (SentenceTransformers)
_local_pipeline_lock = threading.Lock()
