#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv-ml" ]; then
  python3.11 -m venv .venv-ml
fi

source .venv-ml/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r ml/requirements-stem-service.txt

cat <<'EOF'
Stem service environment is ready.
Start the local sidecar with:
  bash ml/run_stem_service.sh
EOF
