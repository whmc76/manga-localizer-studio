import io

import numpy as np
from PIL import Image, ImageDraw

from manga_localizer.config import AppPaths
from manga_localizer.ocr import PageOCR, TextUnit
from manga_localizer.renderer import ArtworkPreservingRenderer, ensure_font


class BoundedTestRenderer(ArtworkPreservingRenderer):
    def __init__(self):
        self.font_path = None

    def _draw(self, image, text, box, mean, white_ratio):
        del text, mean, white_ratio
        x0, y0, x1, y1 = box
        ImageDraw.Draw(image).rectangle((x0 + 8, y0 + 8, x1 - 8, y1 - 8), fill="black")


def test_renderer_preserves_pixels_outside_text_box(tmp_path):
    source = tmp_path / "page.jpg"
    base = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(base)
    draw.rectangle((8, 8, 311, 231), outline="black", width=5)
    draw.text((120, 95), "TEST", fill="black")
    base.save(source, quality=100, subsampling=0)
    page = PageOCR(1, source.name, 320, 240, [
        TextUnit("p001u01", [90, 70, 235, 170], [90, 70, 235, 170], "テスト", 0.9, False, "测试")
    ])
    output = tmp_path / "page.png"
    BoundedTestRenderer().render_page(source, page, output)
    before = np.asarray(Image.open(source).convert("RGB"))
    after = np.asarray(Image.open(output).convert("RGB"))
    mask = np.ones(before.shape[:2], dtype=bool)
    mask[70:170, 90:235] = False
    assert output.exists()
    assert before.shape == after.shape
    assert np.array_equal(before[mask], after[mask])


def test_managed_font_download_is_validated_and_atomic(monkeypatch, tmp_path):
    root = tmp_path / "home"
    paths = AppPaths(root, root / "models", root / "cache", root / "jobs", root / "settings.json")
    payload = b"OTTO" + (b"0" * 1_000_000)
    monkeypatch.setattr("manga_localizer.renderer.urlopen", lambda *_args, **_kwargs: io.BytesIO(payload))
    font = ensure_font(paths, force_managed=True)
    assert font.exists()
    assert font.read_bytes()[:4] == b"OTTO"
    assert not font.with_suffix(".otf.part").exists()
