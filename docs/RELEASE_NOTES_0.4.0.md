# Manga Localizer Studio 0.4.0

This release moves the project to a local, style-aware production renderer.

- Quality mode uses Big-LaMa for bounded artwork reconstruction instead of filling OCR boxes with flat white or relying on OpenCV inpainting alone.
- Source lettering is analyzed for orientation, scale, weight, color, spacing, and outline. Large black titles with thick white strokes are rebuilt with managed bold Chinese fonts at comparable scale.
- A generic connected-component fallback proposes horizontal and vertical light-on-dark title regions that Paddle may miss; candidates only survive when MangaOCR recognizes Japanese.
- LaMa operates at the source crop's native resolution. Network output is accepted only inside the generated text mask; every unmasked source pixel is copied back exactly.
- Translation remains local by default with Hy-MT2. Ollama can now serve translation, vision OCR, or both, with independent model selection.
- The default output is lossless WebP to avoid the multi-fold disk inflation of PNG while preserving decoded RGB pixels. PNG remains selectable.
- The model manager automatically downloads the pinned Big-LaMa TorchScript asset with SHA-256 verification and atomic installation. Model weights are not bundled with the MIT-licensed source.

The release is tested against both generic synthetic layouts and a 125-page regression manga: 949 translated units, 94 explicit preserves, zero unresolved units, and zero changed pixels outside declared edit regions. The regression book is evidence, not a source of page-, title-, coordinate-, or phrase-specific renderer rules.
