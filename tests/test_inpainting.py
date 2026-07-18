import numpy as np
import pytest

from manga_localizer.inpainting import LaMaInpainter


def test_lama_wrapper_preserves_every_unmasked_pixel(tmp_path):
    torch = pytest.importorskip("torch")

    class WhiteFill(torch.nn.Module):
        def forward(self, image, mask):
            return torch.ones_like(image)

    model_path = tmp_path / "tiny-lama.pt"
    traced = torch.jit.trace(
        WhiteFill(),
        (torch.zeros((1, 3, 16, 16)), torch.zeros((1, 1, 16, 16))),
    )
    traced.save(str(model_path))
    source = np.arange(13 * 19 * 3, dtype=np.uint8).reshape((13, 19, 3))
    mask = np.zeros((13, 19), dtype=np.uint8)
    mask[3:9, 5:12] = 255
    output = LaMaInpainter(model_path, "cpu")(source, mask)
    assert output.shape == source.shape
    assert np.array_equal(output[mask == 0], source[mask == 0])
    assert np.all(output[mask > 0] == 255)
