# Manga Localizer Studio 0.3.0

This release fixes the quality regression introduced by the detection-only OCR optimization.

- Quality mode is now the default and uses PaddleOCR recognition boxes plus Manga OCR.
- Fast detection-only mode remains available for previews.
- Multi-page translation favors coherent, natural Chinese instead of a strict 1.5x character cap.
- Rotation no longer causes dialogue to be silently classified as a sound effect.
- Reviewed transcripts can be imported from the UI or CLI and rerendered without another model pass.
- Quality cleanup removes source glyphs completely inside bounded text regions.
- `scripts/verify_output.py` checks every page for dimensions and out-of-region pixel changes.

The 125-page regression book completed with 811 reviewed Chinese units, exact source dimensions, and zero changed pixels outside declared text regions.
