#!/usr/bin/env bash
# Generate notebooks/*.ipynb from notebooks/src/*.py (jupytext percent format) and execute them.
# Never hand-edit the .ipynb files.
#   bash scripts/build_notebooks.sh          # write executed notebooks to notebooks/
#   bash scripts/build_notebooks.sh --check  # execute into a temp dir only (used by check.sh)
# Timeout per notebook (s): env FOODLENS_NB_TIMEOUT, default 900.
set -euo pipefail
cd "$(dirname "$0")/.."

mode="build"
[ "${1:-}" = "--check" ] && mode="check"
timeout="${FOODLENS_NB_TIMEOUT:-900}"

shopt -s nullglob
sources=(notebooks/src/*.py)
if [ ${#sources[@]} -eq 0 ]; then
  echo "no notebook sources in notebooks/src"
  exit 0
fi

if [ "$mode" = "check" ]; then
  out_dir="$(mktemp -d)"
  trap 'rm -rf "$out_dir"' EXIT
else
  out_dir="notebooks"
fi

for src in "${sources[@]}"; do
  name="$(basename "${src%.py}")"
  nb="$out_dir/$name.ipynb"
  echo "-- $src -> $nb"
  uv run jupytext --to ipynb --output "$nb" "$src"
  uv run python - "$nb" "$timeout" <<'PY'
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path, timeout = Path(sys.argv[1]), int(sys.argv[2])
nb = nbformat.read(path, as_version=4)
# Run with the repo root as cwd so relative paths (configs/, data/) resolve.
NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, path)
print(f"executed {path}")
PY
done
echo "notebooks: ${#sources[@]} built ($mode)"
