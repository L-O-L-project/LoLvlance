#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-ml"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  echo "Run ml/setup_test_env.sh first or set PYTHON_BIN explicitly." >&2
  exit 1
fi

cd "${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" -m unittest discover -s ml/tests -p 'test_*.py' -v
