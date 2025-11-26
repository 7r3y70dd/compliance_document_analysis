#!/usr/bin/env bash

# Pick the GPU you want to use
export CUDA_VISIBLE_DEVICES=0

# Embeddings (SentenceTransformer)
export EMBEDDING_DEVICE=-1
export EMBEDDING_MODEL="intfloat/e5-base-v2"
export USE_E5_PREFIXES=1

# Cross-encoder reranker (make sure your code passes device to CrossEncoder)
export USE_CROSS_ENCODER=1
export CROSS_ENCODER_MODEL="BAAI/bge-reranker-large"
export CROSS_ENCODER_TOP_K=5

# NLI judge
export USE_NLI_JUDGE=1
export NLI_MODEL="facebook/bart-large-mnli"
export NLI_DEVICE=0
# If you want it to actually affect labels (not just annotate):
export NLI_ANNOTATE_ONLY=0

# LLM judge / refine
export USE_LLM_JUDGE=1
export USE_LLM_REFINE=1
export LLM_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
export LLM_DEVICE=-1
export LLM_MAX_NEW_TOKENS=192

# Editor (rewriter) – only if your rewriter reads EDITOR_* envs
export EDITOR_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
export EDITOR_DEVICE=0
export EDITOR_MAX_NEW_TOKENS=128
# If you implemented 4-bit load in editor, enable it:
export EDITOR_LOAD_IN_4BIT=1

# Optional: reduce CUDA fragmentation for big models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Progress logging frequency (you already added)
export PROGRESS_EVERY=25
