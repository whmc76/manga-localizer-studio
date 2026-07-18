# Manga Localizer Studio 0.4.3

This runtime reliability patch replaces ambiguous PyTorch index resolution with the exact pair verified on the project's supported profiles.

- CPU bootstrap installs Torch 2.8.0+cpu with Torchvision 0.23.0+cpu.
- CUDA 12.9 bootstrap installs Torch 2.8.0+cu129 with Torchvision 0.23.0+cu129.
- `doctor --require-ml` now imports both packages, validates their public versions, reports CUDA availability, and fails bootstrap when the runtime is incomplete or mismatched.

The CUDA pair was installed into a real Windows Python 3.12 environment and verified with an NVIDIA GeForce RTX 5090. MangaOCR imported with warnings treated as errors, confirming that its intended Torchvision image processor is active rather than the PIL fallback.
