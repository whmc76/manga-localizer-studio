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
if [[ "$PROFILE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then PROFILE="cuda129"; else PROFILE="cpu"; fi
fi
if command -v uv >/dev/null 2>&1; then
  SYNC_ARGS=(sync --locked --extra ml --python 3.12)
  if [[ "$DEV" == 1 ]]; then SYNC_ARGS+=(--extra test); fi
  uv "${SYNC_ARGS[@]}"
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
  if [[ "$PROFILE" == "cuda129" ]]; then
    uv pip install --python "$PYTHON" --reinstall 'torch==2.8.0+cu129' 'torchvision==0.23.0+cu129' --index-url https://download.pytorch.org/whl/cu129
    uv pip uninstall --python "$PYTHON" paddlepaddle
    uv pip install --python "$PYTHON" paddlepaddle-gpu --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
  else
    uv pip install --python "$PYTHON" --reinstall 'torch==2.8.0+cpu' 'torchvision==0.23.0+cpu' --index-url https://download.pytorch.org/whl/cpu
  fi
else
  echo "Warning: uv is not installed; using the compatible venv + pip path." >&2
  if [[ ! -x .venv/bin/python ]]; then python3.12 -m venv .venv; fi
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
  "$PYTHON" -m pip install --upgrade pip wheel
  if [[ "$DEV" == 1 ]]; then EXTRA='.[ml,test]'; else EXTRA='.[ml]'; fi
  "$PYTHON" -m pip install -e "$EXTRA"
  if [[ "$PROFILE" == "cuda129" ]]; then
    "$PYTHON" -m pip install 'torch==2.8.0+cu129' 'torchvision==0.23.0+cu129' --index-url https://download.pytorch.org/whl/cu129
    "$PYTHON" -m pip uninstall -y paddlepaddle
    "$PYTHON" -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
  else
    "$PYTHON" -m pip install 'torch==2.8.0+cpu' 'torchvision==0.23.0+cpu' --index-url https://download.pytorch.org/whl/cpu
  fi
fi

"$PYTHON" -m manga_localizer.cli assets download
if [[ "$SKIP_MODELS" == 0 ]]; then
  "$PYTHON" -m manga_localizer.cli models download all
fi
"$PYTHON" -m manga_localizer.cli doctor --require-ml
echo "Ready. Start with: ./scripts/start.sh"
