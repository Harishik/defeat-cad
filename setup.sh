#!/usr/bin/env bash
# One-time setup on macOS / Linux.  Usage: ./setup.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3; do command -v $c >/dev/null && PY=$c && break; done
fi
echo "Using $($PY --version)"
$PY -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
if command -v ollama >/dev/null; then ollama pull qwen3.5:4b || true; else echo "Optional: install Ollama (https://ollama.com) for the AI assistant"; fi
echo "Done. Start with: ./run.sh"
