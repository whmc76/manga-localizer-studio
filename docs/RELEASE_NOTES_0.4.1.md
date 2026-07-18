# Manga Localizer Studio 0.4.1

This patch release repairs the live preview and completes the MangaOCR runtime environment.

- Tall pages are shown in full with their native aspect ratio instead of being laid out at intrinsic width and clipped by the pane.
- The preview canvas grows with the taller settings inspector, so the existing vertical space is used for the page rather than left blank below a fixed-height canvas.
- Job polling no longer reloads the source image every second. The translated preview is requested only when that page has been rendered; earlier phases show a clear waiting state.
- Torchvision is installed alongside Torch from the same selected CPU or CUDA index, preventing a backend mismatch and removing MangaOCR's PIL-fallback warning on newly bootstrapped environments.
- Paddle's ccache message is hidden only during its import because ccache is an optional compilation cache, not a runtime requirement for inference.

The fix was exercised in a real browser against an actively running 125-page built-in OCR and local Hy-MT2 job. The browser showed the complete 2126×3661 source page, expanded the preview canvas to the available height, and issued no premature translated-preview requests during OCR.
