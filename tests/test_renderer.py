import io

import cv2
import numpy as np
from PIL import Image, ImageDraw

from manga_localizer.config import AppPaths
from manga_localizer.ocr import PageOCR, TextUnit
from manga_localizer.renderer import ArtworkPreservingRenderer, TextStyle, ensure_font


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
    page = PageOCR(
        1,
        source.name,
        320,
        240,
        [
            TextUnit(
                "p001u01",
                [90, 70, 235, 170],
                [90, 70, 235, 170],
                "テスト",
                0.9,
                False,
                "测试",
            )
        ],
    )
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
    page = PageOCR(
        1,
        source.name,
        240,
        180,
        [
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
        ],
    )
    output = tmp_path / "grouped-output.png"
    EraseOnlyRenderer().render_page(source, page, output)
    before = np.asarray(Image.open(source).convert("RGB"))
    after = np.asarray(Image.open(output).convert("RGB"))
    assert np.array_equal(before[75:106, 20:221], after[75:106, 20:221])


def test_quality_lama_mask_never_uses_group_union_as_erase_geometry(tmp_path):
    class MaskPaintingInpainter:
        def __call__(self, crop, mask):
            result = crop.copy()
            result[mask > 0] = (255, 0, 255)
            return result

    source = tmp_path / "grouped-outlined.png"
    image = np.full((220, 320, 3), 180, dtype=np.uint8)
    # Two outlined text columns surround artwork that belongs to the panel.
    cv2.rectangle(image, (35, 35), (85, 185), (255, 255, 255), 12)
    cv2.rectangle(image, (45, 45), (75, 175), (0, 0, 0), -1)
    cv2.rectangle(image, (235, 35), (285, 185), (255, 255, 255), 12)
    cv2.rectangle(image, (245, 45), (275, 175), (0, 0, 0), -1)
    cv2.rectangle(image, (130, 60), (190, 160), (15, 15, 15), -1)
    Image.fromarray(image).save(source)
    page = PageOCR(
        1,
        source.name,
        320,
        220,
        [
            TextUnit(
                "p001u01",
                [20, 20, 300, 200],
                [20, 20, 300, 200],
                "日本語題名",
                1.0,
                False,
                "中文标题",
                erase_boxes=[[25, 25, 95, 195], [225, 25, 295, 195]],
            )
        ],
    )
    renderer = ArtworkPreservingRenderer(
        cleanup_profile="quality",
        inpainter=MaskPaintingInpainter(),
    )
    renderer._draw = lambda *_args, **_kwargs: None
    output = tmp_path / "grouped-outlined-output.png"
    renderer.render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert np.array_equal(rendered[60:161, 130:191], image[60:161, 130:191])


def test_cover_title_uses_a_wider_per_box_outline_mask():
    ordinary_style = TextStyle(
        220, 0.6, "black", "white", 0.06, True, 100, True, True, 15, False
    )
    display_style = TextStyle(
        220, 0.6, "black", "white", 0.06, True, 228, True, True, 15, True
    )
    ordinary = TextUnit("ordinary", [0, 0, 100, 200], [0, 0, 100, 200], "題", 1.0)
    cover = TextUnit(
        "cover", [0, 0, 100, 200], [0, 0, 100, 200], "題", 1.0, special="cover_title"
    )
    assert ArtworkPreservingRenderer._mask_dilation(ordinary, ordinary_style) == 15
    assert ArtworkPreservingRenderer._mask_dilation(ordinary, display_style) == 51
    assert ArtworkPreservingRenderer._mask_dilation(cover, display_style) == 51


def test_large_outlined_cleanup_fills_only_detector_owned_boxes(tmp_path):
    class MaskPaintingInpainter:
        def __call__(self, crop, mask):
            result = crop.copy()
            result[mask > 0] = (255, 0, 255)
            return result

    source = tmp_path / "large-outlined.png"
    image = np.full((300, 600, 3), 80, dtype=np.uint8)
    Image.fromarray(image).save(source)
    unit = TextUnit(
        "display",
        [20, 20, 580, 280],
        [20, 20, 580, 280],
        "大きい文字",
        1.0,
        False,
        "大字",
        erase_boxes=[[40, 40, 180, 260], [420, 40, 560, 260]],
    )
    page = PageOCR(1, source.name, 600, 300, [unit])
    renderer = ArtworkPreservingRenderer(
        cleanup_profile="quality", inpainter=MaskPaintingInpainter()
    )
    renderer._draw = lambda *_args, **_kwargs: None
    renderer._analyze_style = lambda *_args: TextStyle(
        80, 0.0, "white", "black", 0.066, True, 160, True, True, 40, False
    )
    output = tmp_path / "large-outlined-output.png"
    renderer.render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert np.all(rendered[40:260, 40:180] == (255, 0, 255))
    assert np.all(rendered[40:260, 420:560] == (255, 0, 255))
    assert np.array_equal(rendered[40:260, 260:340], image[40:260, 260:340])


def test_textured_outlined_cleanup_preserves_background_outside_glyphs(tmp_path):
    class MaskRecordingInpainter:
        def __init__(self):
            self.mask = None

        def __call__(self, crop, mask):
            self.mask = mask.copy()
            result = crop.copy()
            result[mask > 0] = (255, 0, 255)
            return result

    source = tmp_path / "outlined-halo.png"
    gradient = np.linspace(145, 205, 260, dtype=np.uint8)[None, :, None]
    image = np.repeat(np.repeat(gradient, 300, axis=0), 3, axis=2)
    # Three separated glyph cores with contrasting outlines.  The background
    # is deliberately textured so this follows the former solid-box branch.
    for y in (65, 135, 205):
        cv2.rectangle(image, (95, y - 18), (165, y + 18), (255, 255, 255), 8)
        cv2.rectangle(image, (108, y - 10), (152, y + 10), (0, 0, 0), -1)
    Image.fromarray(image).save(source)
    unit = TextUnit(
        "outlined",
        [50, 30, 210, 270],
        [40, 20, 220, 280],
        "日本語",
        1.0,
        False,
        "中文",
        erase_boxes=[[80, 35, 180, 245]],
    )
    page = PageOCR(1, source.name, 260, 300, [unit])
    inpainter = MaskRecordingInpainter()
    renderer = ArtworkPreservingRenderer(
        cleanup_profile="quality", inpainter=inpainter
    )
    renderer._draw = lambda *_args, **_kwargs: None
    renderer._analyze_style = lambda *_args: TextStyle(
        180, 0.2, "black", "white", 0.066, True, 80, True, True, 24, False
    )
    output = tmp_path / "outlined-halo-output.png"
    renderer.render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert inpainter.mask is not None
    # The mask follows glyph shapes rather than filling the detector rectangle
    # plus a type-size halo.  Most nearby screentone remains byte-identical.
    assert np.count_nonzero(inpainter.mask) < 14_000
    changed = np.any(rendered != image, axis=2)
    assert changed.sum() < 14_000
    assert np.array_equal(rendered[35:245, 45:75], image[35:245, 45:75])
    assert np.array_equal(rendered[35:245, 185:215], image[35:245, 185:215])


def test_renderer_preserves_detached_garment_logo_from_legacy_group(tmp_path):
    class MaskPaintingInpainter:
        def __call__(self, crop, mask):
            result = crop.copy()
            result[mask > 0] = (255, 0, 255)
            return result

    source = tmp_path / "garment-logo.png"
    image = np.full((800, 600, 3), 180, dtype=np.uint8)
    for x in (35, 85):
        for y in (390, 470, 550):
            cv2.rectangle(image, (x - 8, y - 18), (x + 28, y + 18), (255, 255, 255), 6)
            cv2.rectangle(image, (x, y - 10), (x + 20, y + 10), (0, 0, 0), -1)
    # Detached artwork in the legacy semantic group must not become erase
    # authority merely because it is text-like.
    cv2.rectangle(image, (220, 260), (500, 360), (25, 25, 25), -1)
    Image.fromarray(image).save(source)
    unit = TextUnit(
        "mixed",
        [10, 200, 580, 700],
        [0, 180, 590, 720],
        "おっぱいデカくね!?",
        0.94,
        False,
        "胸是不是变大了！？",
        erase_boxes=[
            [20, 330, 80, 680],
            [70, 340, 130, 620],
            [150, 200, 580, 430],
        ],
    )
    page = PageOCR(1, source.name, 600, 800, [unit])
    renderer = ArtworkPreservingRenderer(
        cleanup_profile="quality", inpainter=MaskPaintingInpainter()
    )
    renderer._draw = lambda *_args, **_kwargs: None
    renderer._analyze_style = lambda *_args: TextStyle(
        180, 0.2, "black", "white", 0.066, True, 80, True, True, 24, True
    )
    output = tmp_path / "garment-logo-output.png"
    renderer.render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert np.any(np.all(rendered[360:590, 20:80] == (255, 0, 255), axis=2))
    assert np.any(np.all(rendered[360:590, 70:130] == (255, 0, 255), axis=2))
    assert np.array_equal(rendered[200:430, 180:560], image[200:430, 180:560])


def test_grouped_column_width_is_an_upper_bound_for_font_size():
    rgb = np.full((700, 500, 3), 210, dtype=np.uint8)
    for x in (70, 300):
        cv2.rectangle(rgb, (x, 80), (x + 120, 620), (255, 255, 255), 18)
        cv2.rectangle(rgb, (x + 18, 100), (x + 102, 600), (0, 0, 0), -1)
    unit = TextUnit(
        "group",
        [40, 50, 450, 650],
        [40, 50, 450, 650],
        "日本語の縦書き",
        1.0,
        False,
        "中文竖排文字",
        erase_boxes=[[50, 60, 210, 640], [280, 60, 440, 640]],
    )
    style = ArtworkPreservingRenderer._analyze_style(rgb, unit)
    assert style.font_size <= 122


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
            ImageDraw.Draw(image).rectangle(
                (-10, -10, image.width + 10, image.height + 10), fill="black"
            )

    source = tmp_path / "clip.png"
    Image.new("RGB", (100, 100), "white").save(source)
    page = PageOCR(
        1,
        source.name,
        100,
        100,
        [
            TextUnit(
                "p001u01",
                [20, 20, 80, 80],
                [20, 20, 80, 80],
                "日本",
                1.0,
                False,
                "中国",
            )
        ],
    )
    output = tmp_path / "clip-output.png"
    OverdrawRenderer().render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert np.all(rendered[:20] == 255)
    assert np.all(rendered[80:] == 255)
    assert np.all(rendered[:, :20] == 255)
    assert np.all(rendered[:, 80:] == 255)


def test_managed_font_download_is_validated_and_atomic(monkeypatch, tmp_path):
    root = tmp_path / "home"
    paths = AppPaths(
        root, root / "models", root / "cache", root / "jobs", root / "settings.json"
    )
    payload = b"OTTO" + (b"0" * 1_000_000)
    monkeypatch.setattr(
        "manga_localizer.renderer.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )
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
