#!/usr/bin/env bash
# The quality gate every session runs. Ends with "CHECK RESULT: PASS" or "CHECK RESULT: FAIL".
# Usage: bash scripts/check.sh 2>&1 | tee logs/check_latest.log
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
fail=0
step(){ echo "== $1"; shift; "$@" || { echo "!! FAILED: $*"; fail=1; }; }
step "ruff"        uv run ruff check src tests scripts
step "ruff-format" uv run ruff format --check src tests scripts
step "pytest"      uv run pytest -q --maxfail=5
step "size-limits" uv run python scripts/check_sizes.py
if [ -n "$(ls notebooks/src/*.py 2>/dev/null)" ]; then step "notebooks" bash scripts/build_notebooks.sh --check; fi
[ $fail -eq 0 ] && echo "CHECK RESULT: PASS" || { echo "CHECK RESULT: FAIL"; exit 1; }
