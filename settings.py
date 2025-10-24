# settings.py
from pydantic import BaseModel
import os

class Settings(BaseModel):
    # --- Embeddings / reranker ---
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    use_e5_prefixes: bool = bool(int(os.getenv("USE_E5_PREFIXES", "0")))

    use_cross_encoder: bool = bool(int(os.getenv("USE_CROSS_ENCODER", "0")))
    cross_encoder_model_name: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    cross_encoder_top_k: int = int(os.getenv("CROSS_ENCODER_TOP_K", "5"))

    # --- Optional rationale / LLM judge (existing) ---
    rationale_model_name: str = os.getenv("RATIONALE_MODEL", "google/flan-t5-base")
    use_llm_judge: bool = bool(int(os.getenv("USE_LLM_JUDGE", "0")))
    judge_model_name: str = os.getenv("JUDGE_MODEL", "google/flan-t5-base")

    # --- Semantic normalizer (HF-based) & numeric comparator ---
    # Back-compat: respect USE_SYNONYM_MAP if set; new flag is USE_SEMANTIC_NORMALIZER
    use_semantic_normalizer: bool = bool(int(os.getenv("USE_SEMANTIC_NORMALIZER", os.getenv("USE_SYNONYM_MAP", "1"))))
    use_numeric_check: bool = bool(int(os.getenv("USE_NUMERIC_CHECK", "1")))
    semantic_lexicon_path: str = os.getenv("SEMANTIC_LEXICON_PATH", "lexicon.yaml")
    semantic_threshold: float = float(os.getenv("SEMANTIC_THRESHOLD", "0.68"))
    semantic_max_tags: int = int(os.getenv("SEMANTIC_MAX_TAGS", "3"))

    # --- NEW: NLI judge (MNLI models) ---
    use_nli_judge: bool = bool(int(os.getenv("USE_NLI_JUDGE", "0")))                 # off by default
    nli_model_name: str = os.getenv("NLI_MODEL", "distilroberta-base-mnli")          # or roberta-large-mnli
    nli_satisfied_floor: float = float(os.getenv("NLI_SAT_FLOOR", "0.70"))           # entailment => satisfied
    nli_partial_floor: float = float(os.getenv("NLI_PARTIAL_FLOOR", "0.45"))         # mid => partial
    nli_annotate_only: bool = bool(int(os.getenv("NLI_ANNOTATE_ONLY", "1")))         # annotate (no relabel) by default

    # --- Thresholds / limits ---
    satisfied_threshold: float = float(os.getenv("SATISFIED_T", "0.78"))
    partial_threshold: float = float(os.getenv("PARTIAL_T", "0.60"))
    max_chars: int = int(os.getenv("MAX_CHARS", "800000"))

    # --- Misc ---
    use_hf_inference_api: bool = bool(int(os.getenv("USE_HF_API", "0")))

settings = Settings()
