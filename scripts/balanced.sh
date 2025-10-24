#!/usr/bin/env bash
# Balanced defaults for the Policy Compliance Analyzer
# Usage: source ./profiles/balanced.sh

# Embeddings / retrieval
export SATISFIED_T=0.78
export PARTIAL_T=0.60
export USE_CROSS_ENCODER=1
export CROSS_ENCODER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
export CROSS_ENCODER_TOP_K=5

# Semantic normalizer
export USE_SEMANTIC_NORMALIZER=1
export SEMANTIC_THRESHOLD=0.68   # raise to be stricter (e.g., 0.72)

# Numeric checks
export USE_NUMERIC_CHECK=1

# NLI judge (allows upgrades/downgrades)
export USE_NLI_JUDGE=1
export NLI_MODEL="distilroberta-base-mnli"
export NLI_SAT_FLOOR=0.70
export NLI_PARTIAL_FLOOR=0.45
export NLI_ANNOTATE_ONLY=0       # 0 = can change labels; 1 = annotate only

# LLM refine (optional second opinion on partial/non-existent)
export USE_LLM_REFINE=true
export LLM_MODEL_NAME="google/flan-t5-base"
export LLM_DEVICE=-1             # -1=CPU, or 0 for GPU
export LLM_MAX_NEW_TOKENS=192

echo "[profiles/balanced] Environment loaded."
