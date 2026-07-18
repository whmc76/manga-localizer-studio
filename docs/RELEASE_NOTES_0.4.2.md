# Manga Localizer Studio 0.4.2

This compatibility patch keeps the 0.4.1 live-preview fixes and corrects the Torchvision dependency boundary for hardware-aware bootstrap.

PyTorch's CUDA 12.9 Windows index currently pairs Torch 2.8 with Torchvision 0.23 for Python 3.12. The project now accepts that official matching pair, so the bootstrap overlay and installed package metadata agree. CPU and newer CUDA indexes can continue resolving newer compatible Torch/Torchvision pairs.
