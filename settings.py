from pydantic import BaseModel
from typing import Optional
import os

class Settings(BaseModel):
    # Embedding model (set via EMBEDDING_MODEL). Defaults to MiniLM; upgrade to all-mpnet-base-v2 or E5 variants.
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # NEW: E5 prefix handling (set USE_E5_PREFIXES=1 when using intfloat/e5-* models).
    use_e5_prefixes: bool = bool(int(os.getenv("USE_E5_PREFIXES", "0")))

    # NEW: Optional cross-encoder reranker for top-k candidates. Enable with USE_CROSS_ENCODER=1.
    use_cross_encoder: bool = bool(int(os.getenv("USE_CROSS_ENCODER", "0")))
    cross_encoder_model_name: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    cross_encoder_top_k: int = int(os.getenv("CROSS_ENCODER_TOP_K", "5"))

    # Optional instruction model for rationales.
    rationale_model_name: str = os.getenv("RATIONALE_MODEL", "google/flan-t5-base")

    # NEW: LLM judge to refine partial matches and produce a one-line rationale.
    use_llm_judge: bool = bool(int(os.getenv("USE_LLM_JUDGE", "0")))
    judge_model_name: str = os.getenv("JUDGE_MODEL", "google/flan-t5-base")

    # Similarity thresholds
    satisfied_threshold: float = float(os.getenv("SATISFIED_T", "0.78"))
    partial_threshold: float = float(os.getenv("PARTIAL_T", "0.60"))

    # Max characters to accept per document
    max_chars: int = int(os.getenv("MAX_CHARS", "800000"))

    # Toggle HF local vs Inference API. Here we keep it local for simplicity.
    use_hf_inference_api: bool = bool(int(os.getenv("USE_HF_API", "0")))

settings = Settings()