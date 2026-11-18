#!/usr/bin/env bash
# Lists (and with --yes, runs) removal commands for tools this project installed OUTSIDE .venv.
# Keep CMDS in sync whenever a session installs something globally (record it in the session log).
#   bash scripts/uninstall_global_tools.sh        # dry run: print commands
#   bash scripts/uninstall_global_tools.sh --yes  # execute them
set -euo pipefail

# P0 (2026-11-18): uv 0.11.6 was ALREADY installed on this machine, so it is not listed here.
CMDS=()

echo "# Project environment (always safe): rm -rf .venv"
if [ ${#CMDS[@]} -eq 0 ]; then
  echo "# No global tools were installed by this project."
  exit 0
fi
for c in "${CMDS[@]}"; do echo "$c"; done
if [ "${1:-}" = "--yes" ]; then
  for c in "${CMDS[@]}"; do echo "running: $c"; bash -c "$c"; done
fi
