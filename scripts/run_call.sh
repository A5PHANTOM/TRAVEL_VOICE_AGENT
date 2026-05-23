#!/usr/bin/env bash
set -euo pipefail

# Helper to source .env then run the call script with the given args
set -o allexport
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi
set +o allexport

python3 scripts/call.py "$@"
