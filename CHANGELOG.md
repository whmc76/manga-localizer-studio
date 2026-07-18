# Changelog

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
