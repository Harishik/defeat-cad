#!/usr/bin/env bash
# Start DeFeat on macOS / Linux: http://localhost:8000
cd "$(dirname "$0")"
exec .venv/bin/python launch.py "$@"
