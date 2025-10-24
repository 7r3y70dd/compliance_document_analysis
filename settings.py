from pydantic import BaseModel
from typing import Optional
import os

class Settings(BaseModel):
    # Embedding model (small, fast CPU default). You can upgrade to 'sentence-transformers/all-mpnet-base-v2'.
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Optional instruction model for rationales (kept off by default for speed).
    rationale_model_name: str = os.getenv("RATIONALE_MODEL", "google/flan-t5-base")

    # Similarity thresholds
    satisfied_threshold: float = float(os.getenv("SATISFIED_T", 0.78))
    partial_threshold: float = float(os.getenv("PARTIAL_T", 0.60))

    # Max characters to accept per document
    max_chars: int = int(os.getenv("MAX_CHARS", 800_000))

    # Toggle HF local vs Inference API. Here we keep it local for simplicity.
    use_hf_inference_api: bool = bool(int(os.getenv("USE_HF_API", 0)))

settings = Settings()