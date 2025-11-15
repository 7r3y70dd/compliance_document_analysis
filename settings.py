# settings.py
from pydantic import BaseModel
import os

class Settings(BaseModel):
    # --- Embeddings ---
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: int = int(os.getenv("EMBEDDING_DEVICE", "-1"))  # -1 = CPU
    use_e5_prefixes: bool = bool(int(os.getenv("USE_E5_PREFIXES", "0")))

    # --- Optional cross-encoder reranker ---
    use_cross_encoder: bool = bool(int(os.getenv("USE_CROSS_ENCODER", "0")))
    cross_encoder_model_name: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    cross_encoder_top_k: int = int(os.getenv("CROSS_ENCODER_TOP_K", "5"))

    # --- Optional rationale / legacy judge (kept for back-compat) ---
    rationale_model_name: str = os.getenv("RATIONALE_MODEL", "google/flan-t5-base")
    use_llm_judge: bool = bool(int(os.getenv("USE_LLM_JUDGE", "0")))
    judge_model_name: str = os.getenv("JUDGE_MODEL", "google/flan-t5-base")

    # --- Semantic normalizer & numeric comparator ---
    use_semantic_normalizer: bool = bool(int(os.getenv("USE_SEMANTIC_NORMALIZER", os.getenv("USE_SYNONYM_MAP", "1"))))
    use_numeric_check: bool = bool(int(os.getenv("USE_NUMERIC_CHECK", "1")))
    semantic_lexicon_path: str = os.getenv("SEMANTIC_LEXICON_PATH", "lexicon.yaml")
    semantic_threshold: float = float(os.getenv("SEMANTIC_THRESHOLD", "0.68"))
    semantic_max_tags: int = int(os.getenv("SEMANTIC_MAX_TAGS", "3"))

    # --- NLI judge (MNLI models) ---
    use_nli_judge: bool = bool(int(os.getenv("USE_NLI_JUDGE", "0")))  # off by default
    nli_model_name: str = os.getenv("NLI_MODEL", "facebook/bart-large-mnli")  # robust default
    nli_device: int = int(os.getenv("NLI_DEVICE", "-1"))  # -1 = CPU
    # Original floors (kept):
    nli_satisfied_floor: float = float(os.getenv("NLI_SAT_FLOOR", "0.70"))
    nli_partial_floor: float = float(os.getenv("NLI_PARTIAL_FLOOR", "0.45"))
    nli_annotate_only: bool = bool(int(os.getenv("NLI_ANNOTATE_ONLY", "1")))  # annotate-only by default
    # Aliases used by new matcher:
    nli_entailment_min: float = float(os.getenv("NLI_ENTAILMENT_MIN", os.getenv("NLI_SAT_FLOOR", "0.80")))
    nli_contradiction_min: float = float(os.getenv("NLI_CONTRADICTION_MIN", "0.60"))

    # --- Thresholds / limits ---
    satisfied_threshold: float = float(os.getenv("SATISFIED_T", "0.78"))
    partial_threshold: float = float(os.getenv("PARTIAL_T", "0.60"))
    top_k_default: int = int(os.getenv("TOP_K_DEFAULT", "3"))
    round_similarity: int = int(os.getenv("ROUND_SIMILARITY", "3"))
    max_chars: int = int(os.getenv("MAX_CHARS", "800000"))

    # --- HF Inference API (optional) ---
    use_hf_inference_api: bool = bool(int(os.getenv("USE_HF_API", "0")))

    # --- LLM refine (new) ---
    use_llm_refine: bool = os.getenv("USE_LLM_REFINE", "false").lower() in ("1", "true", "yes")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "google/flan-t5-base")
    llm_device: int = int(os.getenv("LLM_DEVICE", "-1"))  # -1 = CPU
    llm_max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "192"))
    hf_hub_token: str = os.getenv("HF_HUB_TOKEN", "")

    # --- Rationale control (new) ---
    return_rationale_default: bool = os.getenv("RETURN_RATIONALE_DEFAULT", "true").lower() in ("1", "true", "yes")

    # LLM gating + performance
    llm_on_non_existent: bool = (os.getenv("LLM_ON_NON_EXISTENT", "1") in ("1", "true", "yes"))
    llm_on_partial: bool = (os.getenv("LLM_ON_PARTIAL", "0") in ("1", "true", "yes"))  # default OFF
    llm_min_sim_for_llm: float = float(os.getenv("LLM_MIN_SIM_FOR_LLM", "0.10"))  # don't waste LLM below this
    llm_min_sim_for_upgrade: float = float(os.getenv("LLM_MIN_SIM_FOR_UPGRADE", "0.20"))  # block upgrades below this
    llm_max_time: float = float(os.getenv("LLM_MAX_TIME", "2.0"))  # seconds cap per call (generation max_time)

    # NLI-aware upgrade guard (only if NLI enabled)
    llm_require_nli_for_upgrade: bool = (os.getenv("LLM_REQUIRE_NLI_FOR_UPGRADE", "0") in ("1", "true", "yes"))
    nli_entailment_min_for_upgrade: float = float(os.getenv("NLI_ENT_FOR_UPGRADE", "0.65"))

    # --- Compatibility aliases for matcher.py (no code changes needed) ---
    @property
    def sim_threshold_satisfied(self) -> float:
        return self.satisfied_threshold

    @property
    def sim_threshold_partial(self) -> float:
        return self.partial_threshold

    @property
    def rationale_default(self) -> bool:
        return self.return_rationale_default

    @property
    def use_numeric_comparator(self) -> bool:
        # matcher expects this name
        return self.use_numeric_check

settings = Settings()
