#!/usr/bin/env bash
# Lenient defaults for the Policy Compliance Analyzer
# Usage: source ./profiles/lenient.sh

export SATISFIED_T=0.74
export PARTIAL_T=0.55
export USE_CROSS_ENCODER=0

export USE_SEMANTIC_NORMALIZER=1
export SEMANTIC_THRESHOLD=0.62

export USE_NUMERIC_CHECK=1

export USE_NLI_JUDGE=1
export NLI_SAT_FLOOR=0.60
export NLI_PARTIAL_FLOOR=0.40
export NLI_ANNOTATE_ONLY=0

export USE_LLM_REFINE=true

echo "[profiles/lenient] Environment loaded."
