# Changelog

## 0.5.2 - 2026-07-20

- Keep full-page 9B vision results audit-only unless their text agrees with the detector-owned region, preventing neighbouring numbered boxes from swapping dialogue.
- Add exact one-crop-per-unit 9B role auditing for dialogue, sound effects, artwork text, and noise; Japanese grammar can override a wrong VLM role label.
- Preserve clothing prints, logos, props, and implausibly oversized detector regions instead of allowing them into destructive cleanup.
- Reject nested recovery duplicates both within a missing-text recovery batch and across final effect units, while allowing strong exact-crop evidence to recover an automatically skipped effect.
- Add a general Japanese manga SFX layer for common heartbeat, rumble, impact, mechanical, vocal, movement, and wet-sound families; use the 9B model only for unknown effects and never as geometry authority.
- Translate sound effects by default in the CLI, API, UI defaults, and pipeline while retaining the explicit preserve option.
- Fix JMnedict female-name classification and expand natural simplified-Chinese candidate ranking so recurring names resolve automatically without title-specific name tables.
- Remove detached orthogonal scene/artwork boxes from legacy erase groups and constrain LaMa cleanup to the remaining detector-owned text geometry.
- Treat full-page VLM omissions as evidence rather than geometry, collapse substring/containment echoes against known OCR, and re-detect only the remaining local crops.
- Preserve page-spanning decorative sound effects instead of generating destructive cleanup masks; ordinary detector-confirmed dialogue and effects remain translatable.
- Refresh inferred recurring names through natural Chinese-name candidate filtering and migrate existing transcript references without any title-specific cast table.
- Prefer high-confidence Latin logo evidence over conflicting low-confidence Japanese crop hallucinations, protecting garment and prop lettering during legacy transcript audits.
- Remove the redundant second full-page VLM recovery pass after local crop verification to reduce quality-mode latency.
- Invalidate earlier OCR caches because the semantic role and recovery contracts changed.

## 0.5.1 - 2026-07-19

- Remove thick contrasting source outlines as part of the LaMa mask instead of selecting only the dark glyph core, eliminating white silhouettes behind translated text.
- Scale the bounded cleanup halo from the measured source font size so OCR boxes that omit ruby or the outer stroke cannot leave residual Japanese fragments between detector rectangles.
- Keep grouped detector regions independent and restore every pixel outside declared text regions; the 125-page lossless WebP benchmark again passes with zero unresolved units and zero out-of-boundary pixel changes.

## 0.5.0 - 2026-07-19

- Add a fully local quality ensemble: one unrestricted Qwen3.5 9B model performs staged drafting, candidate selection, and targeted semantic-risk retranslation, with Hy-MT2 providing the independent candidate.
- Bound the 9B reviewer to a 32K working context and deterministic structured output; the unrestricted model's hidden-thinking mode could consume 32K output tokens without emitting JSON, so semantic checks are expressed directly in the verifier prompt instead.
- Add ModelScope-first automatic name resolution with JMnedict-backed recurring-name consistency, without book-specific name tables.
- Combine Paddle exact geometry, Manga OCR recognition, and full-page 9B semantic correction; VLM-only boxes can never erase artwork.
- Recover VLM-reported omissions through enlarged Paddle re-detection and independent Manga OCR confirmation, including pages where the first detector pass finds no text.
- Prevent complete hiragana dialogue from being misclassified as sound effects while preserving explicit katakana effects, isolated glyphs, and short vocalizations.
- Make six previous pages the default story context and add source-density layout budgets for compact Chinese text in narrow manga regions.
- Require only Hy-MT2 plus the selected 9B-or-smaller Ollama model in quality-mode readiness checks, and retain the single-model fast profile for lighter hardware.
- Keep lossless WebP at native pixel dimensions and preserve LaMa's strict masked-edit boundary.
- Load Hy-MT2 only for its candidate stage and explicitly release it and the 9B model between stages so LaMa rendering does not inherit translation VRAM pressure.
- Enforce one shared translation acceptance gate across draft, repair, candidate selection, and semantic audit, including simplified-Chinese normalization, source-information density, glossary consistency, and negation preservation.
- Keep ordinary A/B selection deterministic and non-thinking, reserving bounded reasoning for semantic-risk and unresolved lines instead of applying it to hundreds of ordinary wording differences.
- Filter duplicate ruby, crop echoes, malformed tiny OCR fragments, and VLM-only hallucinations before translation; exact detector geometry remains the only authority allowed to erase pixels.
- Add targeted semantic checks for questions, negation, invitations, names, laughter, and game-related actions, then retranslate only the risky units with the same local 9B model.
- Preserve large multi-column outlined lettering as one composition while retaining the original per-column erase boxes, font weight, foreground color, and contrasting stroke.
- Complete a 125-page native-resolution benchmark with 1,285 text units, 687 Chinese replacements, 598 intentionally preserved sound-effect units, zero unresolved or invalid translations, and zero changed pixels outside declared edit regions.

## 0.4.6 - 2026-07-19

- Add a distinct final repair pass when a local translation remains empty, malformed, or contains Japanese kana, while keeping the incomplete-work quality gate strict.
- Infer a compact glossary for recurring katakana character names and reject per-line candidates that do not use the selected fixed name, keeping character references coherent without an LLM API.
- Re-read sparse full-page display lettering as one MangaOCR title region, preserving the original erase boxes while restoring a single large outlined Chinese composition.
- Reject duplicated cross-unit candidates, disproportionate short-text expansions, slash-joined context, unrelated glossary names, and source-less Latin characters before they reach rendering.
- Add a deterministic local fallback for short failed vocalizations so hearts and punctuation survive without retaining kana or requiring an API.
- Persist every candidate in `translation-draft.json` so a failed translation remains diagnosable and reviewable instead of discarding model output.
- Resume failed jobs from source- and backend-validated OCR caches, including safe compatibility with built-in quality caches created by earlier releases.

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
