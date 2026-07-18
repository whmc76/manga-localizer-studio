# Manga Localizer Studio 0.3.1

This release fixes false-completion and destructive-cleanup regressions found by a full-book review.

- Legacy `skip: true` no longer means “reviewed complete”; it imports as unresolved unless an explicit reason is present.
- Reviewed translations that still contain Japanese kana, leaked page context, or abnormal expansion are sent through translation again.
- Valid reviewed Chinese is retained instead of being needlessly regenerated.
- Rendering and `scripts/verify_output.py` fail while any text unit remains unresolved.
- OCR quality no longer enables aggressive full-box cleanup. Production rendering uses edge-aware cleanup to protect artwork.
- Manifests and QA reports expose translated, explicitly skipped, preserved-SFX, invalid, and unresolved counts.

The 125-page regression book contains 1,042 reviewed OCR units: 948 valid Chinese replacements, 94 explicit duplicate/symbol preserves, and zero unresolved or kana-bearing translations. Exact dimensions and bounded-pixel verification are rerun for the release artifact.
