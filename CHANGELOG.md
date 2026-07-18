# Changelog

## 0.2.0 - 2026-07-18

- Add selectable built-in Hy-MT2, local Ollama, and online OpenAI-compatible translation backends.
- Keep coherent multi-page prompting and stable text-unit mapping across every backend.
- Make Hy-MT2 optional when Ollama or an online API is selected while keeping OCR local.
- Add backend connection checks and model discovery to the UI.
- Keep online API keys in process memory or `MLS_ONLINE_API_KEY`, never in settings or job files.
- Open the UI before downloading model weights so first-time users can choose a backend.
- Avoid Windows cuDNN DLL conflicts by pairing CUDA PyTorch with CPU Paddle detection.
- Select Manga OCR's CUDA device independently so CPU Paddle does not disable GPU recognition.
- Disable the unsupported Windows CPU oneDNN/PIR path in PaddleOCR.
- Preserve panel art between grouped text lines by erasing only padded, tight detector boxes and retaining edge-connected artwork.
- Skip redundant Paddle recognition because Manga OCR already owns Japanese recognition, substantially reducing CPU batch time.
- Reject context leakage, overlong text, and untranslated kana, then retry the affected unit with a strict bounded prompt.

## 0.1.2 - 2026-07-18

- Make uv the primary environment and dependency manager on Windows and Linux.
- Commit `uv.lock` for reproducible application, ML, and test dependencies.
- Keep CPU/CUDA-specific Torch and Paddle wheels as a hardware-aware uv overlay.
- Run the full Windows/Linux and Python 3.11/3.12 CI matrix through official setup-uv.

## 0.1.1 - 2026-07-18

- Automatically provision a pinned OFL CJK font on minimal Windows/Linux systems.
- Validate font downloads before atomic installation.
- Update GitHub Actions to Node 24-based action releases.
- Normalize CLI output to UTF-8 on Windows runners and legacy code pages.

## 0.1.0 - 2026-07-18

- Initial local-first OCR, context-aware translation, and bounded rendering pipeline.
- Responsive browser workspace and CLI.
- ModelScope-first model manager with transparent Manga OCR fallback.
- One-command virtual environment and weight bootstrap for Windows and Linux.
- MIT licensing, model notices, CI, tests, design contract, and UI parity ledger.
