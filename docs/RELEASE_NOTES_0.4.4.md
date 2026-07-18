# Manga Localizer Studio 0.4.4

This console-quality patch keeps the verified 0.4.3 ML runtime and removes two misleading sources of noise.

- Paddle's ccache notice is now filtered while Paddle actually constructs its inference models. ccache is a compilation cache and is not needed for packaged model inference.
- HTTP INFO logs from Hugging Face optional-file capability probes no longer fill the terminal with expected HEAD/404 checks. Real download and model-load failures still raise normally.

Targeted tests prove that only the exact optional ccache message is filtered and an unrelated inference warning remains visible.
