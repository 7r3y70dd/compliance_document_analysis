set -euo pipefail
export EMBEDDING_MODEL=${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
export SATISFIED_T=${SATISFIED_T:-0.78}
export PARTIAL_T=${PARTIAL_T:-0.60}
uvicorn app:app --host 0.0.0.0 --port 8000