#!/usr/bin/env bash
# docker/setup.sh
#
# Deterministic, idempotent setup for the container.
# - Uses the container's Python (from the base image)
# - Upgrades pip tooling
# - Installs requirements.txt
#
# If something goes wrong (bad requirements, no network), this script
# will exit non-zero so the Docker build fails early & loudly.

set -euo pipefail

echo "==================="
echo "  Setup starting"
echo "==================="

# 1) Show Python version (for debugging)
python -V || python3 -V || {
  echo "ERROR: No python found on PATH."
  exit 1
}

PY_VERSION=$(
python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)

echo "Using Python ${PY_VERSION}"

# 2) Warn if we’re on a weird Python version (e.g. 3.14)
case "$PY_VERSION" in
  3.10|3.11|3.12)
    echo "Python version is supported (3.10–3.12)."
    ;;
  *)
    echo "WARNING: Python ${PY_VERSION} is not officially supported by this project."
    echo "         Things might still work, but binary wheels could be missing."
    ;;
esac

# 3) Upgrade packaging tools (safe & idempotent)
echo "Upgrading pip / setuptools / wheel..."
python -m pip install --upgrade pip setuptools wheel

# 4) Install project requirements
if [ -f requirements.txt ]; then
  echo "Installing Python dependencies from requirements.txt..."
  python -m pip install --no-cache-dir -r requirements.txt
else
  echo "WARNING: requirements.txt not found; skipping Python deps install."
fi

echo "==================="
echo "  Setup complete"
echo "==================="
