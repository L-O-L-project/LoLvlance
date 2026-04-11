#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-ml"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to a compatible interpreter, for example PYTHON_BIN=python3.11" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${ROOT_DIR}/ml/requirements-test.txt"

echo "ML test environment ready at ${VENV_DIR}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
