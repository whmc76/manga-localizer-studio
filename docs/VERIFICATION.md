# Verification report

Date: 2026-07-18

## Automated checks

- 46 unit/API/contract tests passed on Windows with Python 3.12.
- `uv lock --check` passed; the 0.4.0 source distribution and wheel built successfully.
- Renderer regressions cover native-resolution LaMa boundaries, exact unmasked-pixel restoration, source-driven outlined display text, bold balloon text, furigana outside the main OCR box, and lossless output.
- OCR regressions cover ordinary Paddle regions, local Ollama vision calls, and conservative light-on-dark title discovery without page, coordinate, filename, or phrase rules.
- Model regressions cover checksum validation, atomic Big-LaMa installation, backend-dependent requirements, and the four-model UI contract.

## Full-book acceptance test

- Imported the reviewed transcript and rerendered a 125-page, 2126×3661 source book (final page 2800×3808) through the 0.4.0 CLI.
- Produced 125 lossless WebP files; every output dimension matched its source.
- Classified all 1,043 detected units: 949 valid Chinese replacements, 94 explicit duplicate/symbol preserves, zero unresolved units, and zero invalid translation fields.
- The default OCR was reproduced on a previously missed reversed title page: Paddle returned no box, the generic light-on-dark candidate pass found the title, and MangaOCR recognized the complete Japanese text.
- Full decoded-pixel comparison across all 125 pages found zero changed pixels outside declared cleanup/typesetting regions.
- Output size is 239.0 MiB, compared with 173.2 MiB for lossy source images and 467.1 MiB for the former PNG output (1.38× versus 2.70× source size).

## UI checks

- API and DOM contract tests cover all four navigation views, model readiness, independent OCR/translation backend selectors, conditional Ollama/online fields, quality-profile labels, and output-format selection.
- The previously captured desktop/mobile layout contract remains unchanged. A fresh live browser pass was unavailable in this Windows session because the in-app browser kernel could not install its runtime assets; this is an environment limitation, not counted as a visual pass.

## User-reported regressions

- Missed full pages: reversed light-on-dark title regions now enter the default OCR pipeline; the regression book has no untranslated text-bearing page.
- Poor artwork fill: quality mode uses bounded native-resolution Big-LaMa and copies all unmasked source pixels back exactly.
- Weak style matching: fill, outline, weight, orientation, and scale are inferred from source pixels; bold CJK fonts and thick contrasting strokes are supported.
- Oversized files: lossless WebP is the default, while PNG remains selectable.
- Accidental source overwrite: equal source/output paths are rejected by the API and pipeline, and output always goes to a distinct directory.
