#!/usr/bin/env bash
# set_env.sh — exports environment variables for compliance_document_analysis

export CROSS_ENCODER_MODEL="BAAI/bge-reranker-large"
export CROSS_ENCODER_TOP_K=5
export EMBEDDING_MODEL="intfloat/e5-base-v2"
export LLM_DEVICE=-1
export LLM_MAX_NEW_TOKENS=192
export LLM_MODEL_NAME="google/flan-t5-base"
export NLI_ANNOTATE_ONLY=0
export NLI_MODEL="facebook/bart-large-mnli"
export NLI_PARTIAL_FLOOR=0.45
export NLI_SAT_FLOOR=0.70
export PARTIAL_T=0.60
export SATISFIED_T=0.78
export SEMANTIC_THRESHOLD=0.68
export USE_CROSS_ENCODER=1
export USE_E5_PREFIXES=1
export USE_LLM_JUDGE=1
export USE_LLM_REFINE=true
export USE_NLI_JUDGE=1
export USE_NUMERIC_CHECK=1
export USE_SEMANTIC_NORMALIZER=1

echo "✅ Environment variables for compliance_document_analysis set successfully."

