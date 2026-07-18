# Changelog

## 0.4.5 - 2026-07-19

- Reuse an already-running local workspace instead of starting a second server that immediately fails on port 8765.
- Keep the Windows launcher open with an actionable message when startup genuinely fails.

## 0.4.4 - 2026-07-19

- Suppress Paddle's optional ccache notice across the actual model-construction boundary while preserving all actionable inference warnings.
- Keep Hugging Face optional-file HTTP probes out of the user console by raising only the third-party HTTP client log threshold; download failures still surface as exceptions.

## 0.4.3 - 2026-07-19

- Pin the verified Torch 2.8.0 and Torchvision 0.23.0 pair for CPU and CUDA 12.9 instead of relying on ambiguous index metadata.
- Make bootstrap diagnostics import and validate the ML runtime, so an incompatible Torchvision installation can no longer be reported as ready.

## 0.4.2 - 2026-07-18

- Allow the Torchvision 0.23 build paired with Torch 2.8 on PyTorch's CUDA 12.9 Windows index, keeping the hardware-aware overlay consistent with project metadata.

## 0.4.1 - 2026-07-18

- Keep source previews stable during job polling and request translated previews only after the selected page has actually rendered, eliminating expected 404 request spam.
- Fit tall manga pages inside the preview pane without clipping and let the canvas consume the height already created by the settings inspector.
- Show phase-aware output placeholders while OCR, coherent translation, or rendering is still in progress.
- Install Torch and Torchvision together from the selected CPU/CUDA wheel source so MangaOCR has its intended image-processing backend.
- Suppress Paddle's optional ccache notice during ordinary inference while preserving actionable runtime warnings.

## 0.4.0 - 2026-07-18

- Replace destructive text-box cleanup in quality mode with pinned Big-LaMa inpainting; inference stays at native geometry and restores every unmasked pixel exactly.
- Detect display lettering, vertical multi-column text, font scale, weight, foreground, and contrasting stroke from source pixels without a multimodal LLM.
- Add managed Noto Sans CJK SC Bold and remove the former 92 px vertical-font ceiling, allowing large outlined Chinese titles to match the source composition.
- Expand cleanup around detected regions to include furigana and thick source outlines while keeping the declared edit boundary bounded and verifiable.
- Add a conservative light-on-dark title candidate pass so reversed manga lettering is not silently missed by the default detector; MangaOCR must still confirm Japanese text.
- Add Ollama as an optional local vision-OCR backend as well as a translation backend; OCR and translation model choices are independent.
- Default to lossless WebP to reduce PNG size inflation, while retaining lossless PNG as a compatibility option.
- Add checksum-validated, atomic LaMa weight download and Apache-2.0 model attribution.
- Add generic synthetic regressions for outlined display text and white speech bubbles plus a TorchScript boundary test proving that LaMa cannot alter unmasked pixels.

## 0.3.1 - 2026-07-18

- Separate explicit reviewer skips from unresolved text; legacy `skip: true` entries no longer pass as completed work.
- Refuse to render or verify a book while any detected text unit remains unresolved, including non-empty "translations" that still contain Japanese kana or leaked prompt context.
- Record translated, explicitly skipped, preserved sound-effect, and unresolved counts in manifests and QA reports.
- Decouple OCR quality from cleanup aggressiveness; quality OCR now uses edge-aware artwork cleanup instead of destructive full-box erasure.
- Re-audit the 125-page regression book: 948 valid Chinese replacements, zero unresolved or kana-bearing translation fields, and zero pixels changed outside declared text regions.

## 0.3.0 - 2026-07-18

- Restore full PaddleOCR line-level recognition as the default quality profile; retain detection-only OCR as an explicit fast preview.
- Stop treating rotated text regions as sound effects, which previously caused skipped dialogue.
- Relax the translation length gate so natural Chinese and omitted Japanese subjects are not fragmented.
- Translate sound effects by default while keeping preservation available as an option.
- Add reviewed-transcript import for reproducible human-in-the-loop correction and rerendering.
- Add complete text-region cleanup for quality mode while keeping conservative artwork cleanup in fast mode.
- Add a full-book verifier for exact dimensions and zero pixel changes outside declared text regions.

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
