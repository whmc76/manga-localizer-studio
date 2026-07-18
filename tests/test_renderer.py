import io

import cv2
import numpy as np
from PIL import Image, ImageDraw

from manga_localizer.config import AppPaths
from manga_localizer.ocr import PageOCR, TextUnit
from manga_localizer.renderer import ArtworkPreservingRenderer, ensure_font


class BoundedTestRenderer(ArtworkPreservingRenderer):
    def __init__(self):
        self.font_path = None

    def _draw(self, image, text, box, style):
        del text, style
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


def test_grouped_text_erases_only_tight_detector_boxes(tmp_path):
    class EraseOnlyRenderer(ArtworkPreservingRenderer):
        def __init__(self):
            self.font_path = None

        def _draw(self, *_args, **_kwargs):
            return None

    source = tmp_path / "grouped.png"
    base = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(base)
    draw.rectangle((20, 75, 220, 105), fill="black")
    draw.text((45, 35), "TOP", fill="black")
    draw.text((145, 125), "BOTTOM", fill="black")
    base.save(source)
    page = PageOCR(1, source.name, 240, 180, [
        TextUnit(
            "p001u01",
            [35, 25, 215, 155],
            [35, 25, 215, 155],
            "テスト",
            0.9,
            False,
            "测试",
            erase_boxes=[[35, 25, 95, 65], [135, 115, 215, 155]],
        )
    ])
    output = tmp_path / "grouped-output.png"
    EraseOnlyRenderer().render_page(source, page, output)
    before = np.asarray(Image.open(source).convert("RGB"))
    after = np.asarray(Image.open(output).convert("RGB"))
    assert np.array_equal(before[75:106, 20:221], after[75:106, 20:221])


def test_cleanup_preserves_artwork_connected_to_detector_edge():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[48:53, :] = 0
    image[25:40, 40:60] = 0
    cleaned, _, _ = ArtworkPreservingRenderer._erase(image.copy(), [0, 0, 100, 100])
    assert np.array_equal(cleaned[48:53, :], image[48:53, :])
    assert np.all(cleaned[28:37, 43:57] == 255)


def test_quality_cleanup_removes_boundary_connected_source_glyphs():
    image = np.full((60, 60, 3), 255, dtype=np.uint8)
    image[0:45, 28:33] = 0
    cleaned, _, _ = ArtworkPreservingRenderer._erase_complete(
        image.copy(), [0, 0, 60, 60]
    )
    assert np.all(cleaned[0:45, 28:33] == 255)


def test_typesetting_is_clipped_to_declared_box(tmp_path):
    class OverdrawRenderer(ArtworkPreservingRenderer):
        def __init__(self):
            self.font_path = None

        def _draw(self, image, *_args):
            ImageDraw.Draw(image).rectangle((-10, -10, image.width + 10, image.height + 10), fill="black")

    source = tmp_path / "clip.png"
    Image.new("RGB", (100, 100), "white").save(source)
    page = PageOCR(1, source.name, 100, 100, [
        TextUnit("p001u01", [20, 20, 80, 80], [20, 20, 80, 80], "日本", 1.0, False, "中国")
    ])
    output = tmp_path / "clip-output.png"
    OverdrawRenderer().render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert np.all(rendered[:20] == 255)
    assert np.all(rendered[80:] == 255)
    assert np.all(rendered[:, :20] == 255)
    assert np.all(rendered[:, 80:] == 255)


def test_managed_font_download_is_validated_and_atomic(monkeypatch, tmp_path):
    root = tmp_path / "home"
    paths = AppPaths(root, root / "models", root / "cache", root / "jobs", root / "settings.json")
    payload = b"OTTO" + (b"0" * 1_000_000)
    monkeypatch.setattr("manga_localizer.renderer.urlopen", lambda *_args, **_kwargs: io.BytesIO(payload))
    font = ensure_font(paths, force_managed=True)
    assert font.exists()
    assert font.read_bytes()[:4] == b"OTTO"
    assert not font.with_suffix(".otf.part").exists()


def test_display_style_is_detected_without_title_specific_hint():
    height, width = 1400, 600
    gradient = np.linspace(170, 245, height, dtype=np.uint8)[:, None]
    image = np.repeat(gradient, width, axis=1)
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    # Synthetic thick black glyph cores with white outline on textured artwork.
    for y in (180, 420, 660, 900):
        cv2.rectangle(rgb, (205, y), (395, y + 150), (255, 255, 255), 28)
        cv2.rectangle(rgb, (225, y + 20), (375, y + 130), (0, 0, 0), -1)
    unit = TextUnit(
        "title",
        [50, 80, 550, 1280],
        [50, 80, 550, 1280],
        "日本語題名",
        1.0,
        False,
        "中文标题",
    )
    style = ArtworkPreservingRenderer._analyze_style(rgb, unit)
    assert style.display is True
    assert style.outlined is True
    assert style.bold is True
    assert style.stroke_fill == "white"
    assert style.font_size > 92


def test_uniform_white_dialogue_uses_bold_without_fake_outline():
    rgb = np.full((400, 300, 3), 255, dtype=np.uint8)
    rgb[80:320, 120:180] = 0
    unit = TextUnit(
        "dialogue", [40, 30, 260, 370], [40, 30, 260, 370], "日本語", 1.0, False, "对白"
    )
    style = ArtworkPreservingRenderer._analyze_style(rgb, unit)
    assert style.bold is True
    assert style.outlined is False
    assert style.stroke_ratio == 0


def test_quality_plain_balloon_cleans_furigana_outside_main_bbox(tmp_path):
    class EraseOnlyQuality(ArtworkPreservingRenderer):
        def __init__(self):
            self.font_path = None
            self.cleanup_profile = "quality"

        def _draw(self, *_args, **_kwargs):
            return None

    source = tmp_path / "furigana.png"
    image = Image.new("RGB", (220, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((75, 55, 125, 125), fill="black")
    # Small annotation lies outside bbox but inside the reviewed crop.
    draw.rectangle((148, 65, 158, 95), fill="black")
    image.save(source)
    page = PageOCR(
        1,
        source.name,
        220,
        180,
        [
            TextUnit(
                "dialogue",
                [60, 40, 140, 140],
                [40, 30, 175, 150],
                "日本語",
                1.0,
                False,
                "对白",
            )
        ],
    )
    output = tmp_path / "furigana-output.png"
    EraseOnlyQuality().render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert np.all(rendered[65:96, 148:159] == 255)
