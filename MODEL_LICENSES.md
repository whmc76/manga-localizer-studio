# Model and dependency notices

The source code in this repository is MIT licensed. Downloaded model weights are
not bundled and remain governed by their upstream licenses.

| Component | Purpose | Download source | License |
|---|---|---|---|
| PP-OCRv5 | Text-region detection | ModelScope through PaddleX | Apache-2.0 |
| Manga OCR | Japanese manga OCR | `kha-white/manga-ocr-base` (fallback) | Apache-2.0 |
| Big-LaMa | Masked artwork inpainting | pinned `simple-lama-inpainting` v0.1.0 asset (`big-lama.pt`, SHA-256 `7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c`) | Apache-2.0 |
| Hy-MT2 1.8B | Context-aware translation | `Tencent-Hunyuan/Hy-MT2-1.8B` on ModelScope | Apache-2.0 |
| Qwen3.5 Abliterated 9B | Local vision OCR, unrestricted translation, and staged semantic review | Ollama `huihui_ai/qwen3.5-abliterated:9b`, derived from Qwen3.5 | Apache-2.0 derivative; upstream usage warnings apply |
| JMnedict | Automatic Japanese name candidates | Downloaded on demand from EDRDG | CC BY-SA 4.0; attribution to EDRDG |
| Noto Sans CJK SC | Simplified Chinese typesetting | pinned `notofonts/noto-cjk` release | SIL OFL-1.1 |

Review the license files delivered with each downloaded model before commercial
deployment. The application records repository and provider metadata in each
model's `.ready.json` marker.

The abliterated 9B model intentionally reduces refusal/safety behavior so that
it can translate sensitive source material faithfully. It is local-only by
default, is not suitable as a public chatbot, and can produce sensitive or
inappropriate text. Users remain responsible for lawful input and output use.

ModelScope provides `damo/cv_fft_inpainting_lama`, but that legacy pipeline has
a larger runtime dependency graph and is not compatible with the project's
supported Python 3.12 environment. The model manager therefore uses the pinned
upstream TorchScript asset as an explicit fallback instead of silently changing
the supported runtime or downloading an unversioned file.
