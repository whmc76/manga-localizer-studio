import numpy as np
from PIL import Image, ImageDraw

from manga_localizer.ocr import PageOCR, TextUnit
from manga_localizer.renderer import ArtworkPreservingRenderer, find_font


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
    ArtworkPreservingRenderer(find_font()).render_page(source, page, output)
    before = np.asarray(Image.open(source).convert("RGB"))
    after = np.asarray(Image.open(output).convert("RGB"))
    mask = np.ones(before.shape[:2], dtype=bool)
    mask[70:170, 90:235] = False
    assert output.exists()
    assert before.shape == after.shape
    assert np.array_equal(before[mask], after[mask])
