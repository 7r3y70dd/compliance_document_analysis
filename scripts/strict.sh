#!/usr/bin/env bash
# Strict defaults for the Policy Compliance Analyzer
# Usage: source ./profiles/strict.sh

export SATISFIED_T=0.82
export PARTIAL_T=0.65
export USE_CROSS_ENCODER=1
export CROSS_ENCODER_TOP_K=8

export USE_SEMANTIC_NORMALIZER=1
export SEMANTIC_THRESHOLD=0.72

export USE_NUMERIC_CHECK=1

export USE_NLI_JUDGE=1
export NLI_SAT_FLOOR=0.80
export NLI_PARTIAL_FLOOR=0.55
export NLI_ANNOTATE_ONLY=0

export USE_LLM_REFINE=true

echo "[profiles/strict] Environment loaded."
