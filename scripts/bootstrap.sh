#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="auto"
SKIP_MODELS=0
DEV=0
for arg in "$@"; do
  case "$arg" in
    --cpu) PROFILE="cpu" ;;
    --cuda129) PROFILE="cuda129" ;;
    --skip-models) SKIP_MODELS=1 ;;
    --dev) DEV=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv --python 3.12
  else
    python3.12 -m venv .venv
  fi
fi
PYTHON="$PROJECT_ROOT/.venv/bin/python"
"$PYTHON" -m pip install --upgrade pip wheel

if [[ "$PROFILE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then PROFILE="cuda129"; else PROFILE="cpu"; fi
fi
if [[ "$PROFILE" == "cuda129" ]]; then
  "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cu129
  "$PYTHON" -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
else
  "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  "$PYTHON" -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
fi

if [[ "$DEV" == 1 ]]; then EXTRA='.[ml,test]'; else EXTRA='.[ml]'; fi
"$PYTHON" -m pip install -e "$EXTRA"
if [[ "$SKIP_MODELS" == 0 ]]; then
  "$PYTHON" -m manga_localizer.cli models download all
fi
"$PYTHON" -m manga_localizer.cli doctor
echo "Ready. Start with: ./scripts/start.sh"
