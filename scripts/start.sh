#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  ./scripts/bootstrap.sh
fi
exec .venv/bin/python -m manga_localizer.cli ui "$@"
