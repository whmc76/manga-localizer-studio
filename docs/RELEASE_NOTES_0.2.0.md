# Manga Localizer Studio 0.2.0

This release adds selectable translation inference while keeping OCR and image processing local.

## Highlights

- Built-in Hy-MT2, local Ollama, and online OpenAI-compatible translation backends.
- uv-first locked setup with automatic Python, dependency, model, and font provisioning.
- ModelScope-first weight downloads, with an explicit upstream fallback only for Manga OCR.
- Connection checks, remote model discovery, conditional backend settings, and memory-only online API keys.
- Detection-only Paddle stage for substantially faster Windows batches, with CUDA Manga OCR and translation.
- Artwork-preserving, component-aware text cleanup; bounded typesetting cannot spill outside its region.
- Translation quality gates reject context leakage, overlong output, and untranslated kana.
- Incremental OCR, translation, and rendering progress in the UI and CLI.

Validated on Windows/Python 3.12 with a 125-page end-to-end book run and 30 automated tests.
