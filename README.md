# Policy–Compliance Analyzer (FastAPI)

Analyze a **company policy** against a **compliance document** (e.g., HIPAA) and label each compliance clause as **`satisfied`**, **`partially_satisfied`**, or **`non_existent`**. Uses Hugging Face embeddings, optional cross-encoder reranking, and an optional LLM judge for borderline cases.

---

## Overview

**Flow**
1. Parse both documents into **clauses** (headings/bullets/sentences).
2. Embed compliance clauses (queries) and policy clauses (passages).
3. **Match** each compliance clause to the best policy clause.
4. **Label** with thresholds; optionally:
   - **Rerank** with a cross-encoder for sharper matches.
   - **LLM judge** on partials to refine the label + produce a one-line rationale.

**Key files**
```
app.py               # FastAPI endpoints and request/response handling
settings.py          # Config & feature flags (env vars)
matcher.py           # Embeddings, (optional) reranker, LLM judge, rationale
clause_extractor.py  # Clause segmentation (rules based)
models.py            # Pydantic models
requirements.txt     # Python dependencies
run_local.sh         # (optional) local runner
Dockerfile           # Container image
```

---

## Setup

### 1) Python venv
```bash
cd /path/to/compliance_document_analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Start the server (baseline)
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3) Health check
```bash
curl -s http://localhost:8000/health | jq .
```
You’ll see the embedding model and which optional features are enabled.

---

## Feature Flags (safe toggles)

All optional quality boosts are **off by default**. Turn them on with env vars **before** launching uvicorn.

### A) Better embeddings (MPNet)
```bash
export EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### B) E5 embeddings (with prefixes)
```bash
export EMBEDDING_MODEL=intfloat/e5-base-v2
export USE_E5_PREFIXES=1
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### C) Cross-encoder reranking (sharper top-k)
```bash
export USE_CROSS_ENCODER=1
# optional:
export CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
export CROSS_ENCODER_TOP_K=5
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### D) LLM judge for partials + one-line rationale
```bash
export USE_LLM_JUDGE=1
# optional: export JUDGE_MODEL=google/flan-t5-base
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**Recommended quality preset**
```bash
export EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
export USE_CROSS_ENCODER=1
export USE_LLM_JUDGE=1
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Using the API

### 0) Health
```bash
curl -s http://localhost:8000/health | jq .
```

### 1) JSON body (`POST /analyze`)
```bash
curl -s -X POST http://localhost:8000/analyze   -H 'Content-Type: application/json'   -d '{
    "policy_text": "All data at rest is encrypted with AES-256. Access logs are retained for 365 days.",
    "compliance_text": "Encrypt PHI at rest using AES-256. Retain access logs for at least one year.",
    "top_k": 3,
    "use_rationale": false
  }' | jq .
```

### 2) Multipart file upload (`POST /analyze-multipart`)
```bash
curl --fail-with-body -X POST http://localhost:8000/analyze-multipart   -F "policy=@/absolute/path/policy.txt;type=text/plain"   -F "compliance=@/absolute/path/compliance.txt;type=text/plain"   -F "top_k=3"   -F "use_rationale=false" | jq .
```

**Response shape**
```json
{
  "overall": { "satisfied": 2, "partially_satisfied": 3, "non_existent": 5 },
  "clauses": [
    {
      "id": "C-0004",
      "text": "Data Encryption: PHI must be encrypted during storage and transmission.",
      "label": "partially_satisfied",
      "best_match": { "policy_text": "All electronic records are stored on encrypted servers.", "similarity": 0.62 },
      "alternatives": [ { "policy_text": "Transmission ... must use TLS 1.2+", "similarity": 0.59 } ],
      "rationale": "…present only if judge or rationale was enabled…"
    }
  ]
}
```
TRY NLI
```json
export USE_NLI_JUDGE=1
export NLI_ANNOTATE_ONLY=1
export NLI_MODEL=distilroberta-base-mnli   # or roberta-large-mnli
python -m uvicorn app:app --host 0.0.0.0 --port 8000

//let nli downgrade
export USE_NLI_JUDGE=1
export NLI_ANNOTATE_ONLY=0


//LLM JUDGE

export USE_LLM_JUDGE=1
export LLM_MODEL_NAME=google/flan-t5-base
export LLM_DEVICE=-1 




//Balanced defaults

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


//strict defaults
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


//lenient defaults
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


// best env so far

CROSS_ENCODER_MODEL=BAAI/bge-reranker-large
CROSS_ENCODER_TOP_K=5
EMBEDDING_MODEL=intfloat/e5-base-v2
LLM_DEVICE=-1
LLM_MAX_NEW_TOKENS=192
LLM_MODEL_NAME=google/flan-t5-base
NLI_ANNOTATE_ONLY=0
NLI_MODEL=roberta-large-mnli
NLI_PARTIAL_FLOOR=0.45
NLI_SAT_FLOOR=0.70
PARTIAL_T=0.60
SATISFIED_T=0.78
SEMANTIC_THRESHOLD=0.68
USE_CROSS_ENCODER=1
USE_E5_PREFIXES=1
USE_LLM_JUDGE=1
USE_LLM_REFINE=true
USE_NLI_JUDGE=1
USE_NUMERIC_CHECK=1
USE_SEMANTIC_NORMALIZER=1
```

---

## Docker (optional)

```bash
docker build -t policy-compl-svc .
docker run --rm -p 8000:8000   -e EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2   -e USE_CROSS_ENCODER=1   -e USE_LLM_JUDGE=1   policy-compl-svc
```

---

## Rollback (to original behavior)

- Simply **unset** the flags (or set to 0) and restart:
  ```bash
  export USE_E5_PREFIXES=0
  export USE_CROSS_ENCODER=0
  export USE_LLM_JUDGE=0
  python -m uvicorn app:app --host 0.0.0.0 --port 8000
  ```

---