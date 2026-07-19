from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import AppPaths
from .inpainting import LaMaInpainter
from .model_manager import ModelManager
from .ocr import PageOCR, TextUnit


FONT_URL = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/"
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
)
BOLD_FONT_URL = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/"
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf"
)
SYSTEM_FONT_CANDIDATES = (
    "C:/Windows/Fonts/SourceHanSansCN-Medium.otf",
    "C:/Windows/Fonts/NotoSansSC-VF.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)
SYSTEM_BOLD_FONT_CANDIDATES = (
    "C:/Windows/Fonts/SourceHanSansCN-Bold.otf",
    "C:/Windows/Fonts/NotoSansSC-Bold.ttf",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


@dataclass(frozen=True)
class TextStyle:
    mean: float
    white_ratio: float
    fill: str
    stroke_fill: str
    stroke_ratio: float
    bold: bool
    font_size: int
    vertical: bool
    outlined: bool
    background_std: float
    display: bool


def managed_font_path(paths: AppPaths | None = None) -> Path:
    root = (paths or AppPaths.from_env()).root
    return root / "fonts" / "NotoSansCJKsc-Regular.otf"


def managed_bold_font_path(paths: AppPaths | None = None) -> Path:
    root = (paths or AppPaths.from_env()).root
    return root / "fonts" / "NotoSansCJKsc-Bold.otf"


def find_font(paths: AppPaths | None = None) -> Path:
    candidates = (
        os.environ.get("MLS_FONT"),
        managed_font_path(paths),
        *SYSTEM_FONT_CANDIDATES,
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "No CJK font found. Set MLS_FONT to a Simplified Chinese TTF/OTF/TTC file."
    )


def find_bold_font(paths: AppPaths | None = None) -> Path:
    candidates = (
        os.environ.get("MLS_BOLD_FONT"),
        managed_bold_font_path(paths),
        *SYSTEM_BOLD_FONT_CANDIDATES,
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return find_font(paths)


def _download_font(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".otf.part")
    request = Request(url, headers={"User-Agent": "Manga-Localizer-Studio/0.3"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if partial.stat().st_size < 1_000_000 or partial.read_bytes()[:4] != b"OTTO":
            raise RuntimeError("Downloaded CJK font failed size/header validation")
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target


def ensure_font(paths: AppPaths | None = None, force_managed: bool = False) -> Path:
    """Return a CJK font, downloading a pinned OFL font when none is available."""
    target = managed_font_path(paths)
    if target.exists():
        return target
    if not force_managed:
        try:
            return find_font(paths)
        except FileNotFoundError:
            pass
    return _download_font(FONT_URL, target)


def ensure_bold_font(
    paths: AppPaths | None = None, force_managed: bool = False
) -> Path:
    target = managed_bold_font_path(paths)
    if target.exists():
        return target
    if not force_managed:
        return find_bold_font(paths)
    return _download_font(BOLD_FONT_URL, target)


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _clean(text: str) -> str:
    return "".join(text.replace("．．．", "……").replace("．", "。").split())


def _fit_vertical(
    font_path: Path,
    text: str,
    width: int,
    height: int,
    preferred_size: int | None = None,
):
    start = min(height, max(20, preferred_size or min(160, width, height)))
    for size in range(start, 19, -2):
        capacity = max(1, height // round(size * 1.08))
        columns = math.ceil(len(text) / capacity)
        if columns * round(size * 1.12) <= width:
            return _font(font_path, size), size, capacity
    return _font(font_path, 20), 20, max(1, height // 22)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, current = [], ""
    for char in text:
        candidate = current + char
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _fit_horizontal(
    font_path: Path,
    draw,
    text: str,
    width: int,
    height: int,
    preferred_size: int | None = None,
):
    start = min(height, max(20, preferred_size or min(160, height)))
    for size in range(start, 19, -2):
        font = _font(font_path, size)
        lines = _wrap(draw, text, font, width)
        line_height = round(size * 1.24)
        if len(lines) * line_height <= height:
            return font, size, lines, line_height
    font = _font(font_path, 20)
    return font, 20, _wrap(draw, text, font, width), 25


class ArtworkPreservingRenderer:
    def __init__(
        self,
        font_path: Path | None = None,
        bold_font_path: Path | None = None,
        cleanup_profile: str = "artwork",
        paths: AppPaths | None = None,
        device: str = "auto",
        inpainter=None,
    ):
        self.paths = (paths or AppPaths.from_env()).ensure()
        self.font_path = font_path or ensure_font(self.paths)
        self.bold_font_path = bold_font_path or find_bold_font(self.paths)
        if cleanup_profile not in {"artwork", "quality"}:
            raise ValueError(f"Unknown cleanup profile: {cleanup_profile}")
        self.cleanup_profile = cleanup_profile
        self.device = device
        self._inpainter = inpainter

    @staticmethod
    def _appearance(rgb: np.ndarray, box: list[int]) -> tuple[float, float]:
        x0, y0, x1, y1 = box
        crop = rgb[y0:y1, x0:x1]
        if not crop.size:
            return 255.0, 1.0
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        return float(gray.mean()), float((gray > 225).mean())

    @staticmethod
    def _analyze_style(rgb: np.ndarray, unit: TextUnit) -> TextStyle:
        x0, y0, x1, y1 = unit.bbox
        crop = rgb[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        width, height = max(1, x1 - x0), max(1, y1 - y0)
        mean = float(gray.mean()) if gray.size else 255.0
        white_ratio = float((gray > 225).mean()) if gray.size else 1.0
        background = gray[gray > 135]
        background_std = float(background.std()) if background.size else 0.0
        # Japanese manga dialogue is conventionally vertical.  A union of
        # several vertical columns is often almost square, so aspect ratio
        # alone must only classify clearly wide strips as horizontal.
        vertical = unit.special == "cover_title" or height >= width * 0.62

        dark = (gray < 105).astype(np.uint8)
        if dark.any():
            near = cv2.dilate(dark, np.ones((21, 21), np.uint8), iterations=1) > 0
            core = cv2.dilate(dark, np.ones((3, 3), np.uint8), iterations=1) > 0
            ring = near & ~core
            halo_ratio = float((gray[ring] > 225).mean()) if ring.any() else 0.0
        else:
            halo_ratio = 0.0
        outlined = unit.special == "cover_title" or (
            background_std >= 12 and float(dark.mean()) >= 0.01 and halo_ratio >= 0.38
        )
        page_area = max(1, rgb.shape[0] * rgb.shape[1])
        display = unit.special == "cover_title" or (
            outlined
            and width * height / page_area >= 0.12
            and len(_clean(unit.zh)) <= 24
        )

        erase_boxes = unit.erase_boxes or [unit.bbox]
        if display:
            count = max(1, len(_clean(unit.zh)))
            estimated = min(width * 0.23, height / (count * 1.45))
        elif not unit.erase_boxes:
            source_count = max(1, len(_clean(unit.ja)), len(_clean(unit.zh)))
            estimated = math.sqrt(width * height / source_count) * 0.58
        elif vertical:
            line_widths = [max(1, box[2] - box[0]) for box in erase_boxes]
            # Detector rectangles include ruby, punctuation and sometimes two
            # touching glyph columns.  Their full width is therefore an upper
            # bound, not the source em-size.  Keeping roughly three quarters
            # of an outlined column (and two thirds of a plain one) matches
            # the visible main-glyph footprint without letting grouped OCR
            # regions inflate the translated typography.
            box_ratio = 0.76 if outlined else 0.68
            estimated = float(np.median(line_widths)) * box_ratio
            if len(erase_boxes) == 1 and line_widths[0] >= width * 0.8:
                estimated = min(
                    width * 0.72, height / max(1, len(_clean(unit.zh))) * 0.82
                )
        else:
            line_heights = [max(1, box[3] - box[1]) for box in erase_boxes]
            estimated = float(np.median(line_heights)) * (0.78 if outlined else 0.7)
        font_size = max(20, min(320, round(estimated)))

        if mean < 145 and white_ratio < 0.35:
            fill, stroke_fill = "white", "black"
        else:
            fill, stroke_fill = "black", "white"
        bold = outlined or (float(dark.mean()) >= 0.025 and font_size >= 32)
        stroke_ratio = 0.066 if outlined else (0.045 if fill == "white" else 0.0)
        return TextStyle(
            mean=mean,
            white_ratio=white_ratio,
            fill=fill,
            stroke_fill=stroke_fill,
            stroke_ratio=stroke_ratio,
            bold=bold,
            font_size=font_size,
            vertical=vertical,
            outlined=outlined,
            background_std=background_std,
            display=display,
        )

    def _get_inpainter(self):
        if self._inpainter is None:
            weights = ModelManager(self.paths).lama_weights_path()
            self._inpainter = LaMaInpainter(weights, self.device)
        return self._inpainter

    @staticmethod
    def _text_mask(crop: np.ndarray, style: TextStyle, dilation: int) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        if style.fill == "white":
            seed = (gray > 210).astype(np.uint8)
        else:
            seed = (gray < 135).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
        selected = np.zeros_like(seed, dtype=np.uint8)
        height, width = seed.shape
        # Furigana and punctuation are much smaller than the main glyphs but
        # belong to the same replacement region.  Keep their components while
        # still rejecting isolated screentone speckles.
        min_area = max(4, round(style.font_size * style.font_size * 0.0005))
        max_area = max(80, round(width * height * 0.16))
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            touches_edge = (
                x <= 1
                or y <= 1
                or x + component_width >= width - 1
                or y + component_height >= height - 1
            )
            aspect = max(component_width, component_height) / max(
                1, min(component_width, component_height)
            )
            if touches_edge or area < min_area or area > max_area or aspect > 12:
                continue
            selected[labels == label] = 255
        if dilation > 1 and selected.any():
            selected = cv2.dilate(
                selected,
                np.ones((dilation, dilation), np.uint8),
                iterations=1,
            )
        return selected

    @staticmethod
    def _mask_dilation(unit: TextUnit, style: TextStyle) -> int:
        if style.outlined and (
            unit.special == "cover_title" or style.display or style.font_size >= 120
        ):
            value = min(51, max(23, round(style.font_size * 0.22)))
        elif style.outlined:
            value = min(23, max(7, round(style.font_size * 0.14)))
        else:
            value = min(13, max(5, round(style.font_size * 0.08)))
        return value + 1 if value % 2 == 0 else value

    def _erase_with_lama(
        self, rgb: np.ndarray, unit: TextUnit, style: TextStyle
    ) -> np.ndarray:
        page_height, page_width = rgb.shape[:2]
        x0, y0, x1, y1 = unit.crop_bbox or unit.bbox
        cleanup_margin = (
            min(64, max(12, round(style.font_size * 0.4))) if style.outlined else 0
        )
        x0, y0 = max(0, x0 - cleanup_margin), max(0, y0 - cleanup_margin)
        x1, y1 = (
            min(page_width, x1 + cleanup_margin),
            min(page_height, y1 + cleanup_margin),
        )
        context = min(256, max(64, round(style.font_size * 1.5)))
        rx0, ry0 = max(0, x0 - context), max(0, y0 - context)
        rx1, ry1 = min(page_width, x1 + context), min(page_height, y1 + context)
        crop = rgb[ry0:ry1, rx0:rx1].copy()
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        # OCR may group several columns into one semantic unit.  The union is
        # a typesetting region only; it is never valid erase geometry because
        # it can contain faces, line art, furniture, or an entire panel.  LaMa
        # receives only detector-owned source rectangles and builds a pixel
        # mask independently inside each one.
        source_boxes = unit.erase_boxes or [[x0, y0, x1, y1]]
        # Display titles need enough reach to include their proportionally
        # thick white outline. Each detector-owned box remains independent, so
        # the wider title halo cannot turn the semantic group union into a mask.
        dilation = self._mask_dilation(unit, style)
        # A foreground-only component mask removes the dark glyph core but can
        # leave the contrasting source outline behind as a conspicuous white
        # silhouette.  On textured artwork, every confirmed outlined-text box
        # therefore becomes an independent LaMa region.  The detector boxes,
        # not their semantic union, remain the destructive authority.
        solid_source_boxes = style.outlined and (
            style.background_std >= 10 or style.display or unit.special == "cover_title"
        )
        for source_box in source_boxes:
            sx0 = max(x0, source_box[0]) - rx0
            sy0 = max(y0, source_box[1]) - ry0
            sx1 = min(x1, source_box[2]) - rx0
            sy1 = min(y1, source_box[3]) - ry0
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            pad_x = min(32, max(8, round((sx1 - sx0) * 0.08)))
            pad_y = min(32, max(8, round((sy1 - sy0) * 0.06)))
            if solid_source_boxes:
                # OCR rectangles frequently stop at the dark glyph core and
                # can omit small kana or the outer half of a thick contrasting
                # stroke.  Scale the cleanup halo from the measured type size
                # instead of the detector rectangle alone.  The cap keeps
                # separate columns separate, while the wider halo closes the
                # small inter-box gaps in one outlined text cluster.
                outline_pad = min(64, max(16, round(style.font_size * 0.72)))
                pad_x = max(pad_x, outline_pad)
                pad_y = max(pad_y, outline_pad)
            bx0, by0 = max(0, sx0 - pad_x), max(0, sy0 - pad_y)
            bx1, by1 = min(crop.shape[1], sx1 + pad_x), min(crop.shape[0], sy1 + pad_y)
            # On dark or highly textured artwork, opposite-polarity outlined
            # display text can be connected to the background in either the
            # black or white threshold.  Pixel-component selection then keeps
            # only pieces of the original glyph and LaMa faithfully recreates
            # the visible remainder.  For genuinely large outlined lettering,
            # the detector-owned rectangles are the safer source of truth.
            # Fill each rectangle independently; never fill their semantic
            # union, which may contain faces or other panel artwork.
            if solid_source_boxes:
                # Include the detector halo because the contrasting outline and
                # nearby kana can sit outside the reported dark glyph core.
                # Padding remains bounded per detector box rather than filling
                # the semantic union, which may contain unrelated artwork.
                mask[by0:by1, bx0:bx1] = 255
                continue
            local = self._text_mask(crop[by0:by1, bx0:bx1], style, dilation)
            mask[by0:by1, bx0:bx1] = np.maximum(mask[by0:by1, bx0:bx1], local)
        allowed = np.zeros_like(mask)
        allowed[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0] = 255
        mask = cv2.bitwise_and(mask, allowed)
        if not mask.any():
            return rgb
        crop = self._get_inpainter()(crop, mask)
        rgb[ry0:ry1, rx0:rx1] = crop
        return rgb

    @staticmethod
    def _interior_component_mask(binary: np.ndarray, dilation: int) -> np.ndarray:
        """Select text-like components without touching surrounding line art.

        Manga detector rectangles often cross panel art. Artwork normally enters
        through a rectangle edge, while glyph components sit inside it. Keeping
        edge-connected and very large components prevents the cleanup pass from
        turning characters, furniture, and panel borders into white blocks.
        """
        height, width = binary.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        mask = np.zeros_like(binary, dtype=np.uint8)
        max_area = max(64, round(width * height * 0.18))
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            touches_edge = (
                x <= 1
                or y <= 1
                or x + component_width >= width - 1
                or y + component_height >= height - 1
            )
            if touches_edge or area < 4 or area > max_area:
                continue
            mask[labels == label] = 255
        if dilation > 1 and mask.any():
            mask = cv2.dilate(
                mask, np.ones((dilation, dilation), np.uint8), iterations=1
            )
        return mask

    @staticmethod
    def _erase(rgb: np.ndarray, box: list[int]) -> tuple[np.ndarray, float, float]:
        x0, y0, x1, y1 = box
        crop = rgb[y0:y1, x0:x1].copy()
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        mean = float(gray.mean()) if gray.size else 255.0
        white_ratio = float((gray > 225).mean()) if gray.size else 1.0
        if white_ratio > 0.45 or (mean > 175 and white_ratio > 0.35):
            mask = ArtworkPreservingRenderer._interior_component_mask(
                (gray < 150).astype(np.uint8), 5
            )
            crop[mask > 0] = 255
        elif mean < 145:
            mask = ArtworkPreservingRenderer._interior_component_mask(
                (gray > 215).astype(np.uint8), 7
            )
            crop = cv2.inpaint(crop, mask, 6, cv2.INPAINT_TELEA)
        else:
            mask = ArtworkPreservingRenderer._interior_component_mask(
                (gray < 75).astype(np.uint8), 7
            )
            crop = cv2.inpaint(crop, mask, 6, cv2.INPAINT_TELEA)
        rgb[y0:y1, x0:x1] = crop
        return rgb, mean, white_ratio

    @staticmethod
    def _erase_complete(
        rgb: np.ndarray, box: list[int]
    ) -> tuple[np.ndarray, float, float]:
        """Remove all text-colored pixels inside an OCR-confirmed rectangle.

        This intentionally prefers complete source-text removal over conserving
        pixels *inside* the detected text region. Changes remain bounded to that
        region; surrounding artwork is never sampled, resized, or overwritten.
        """
        x0, y0, x1, y1 = box
        crop = rgb[y0:y1, x0:x1].copy()
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        mean = float(gray.mean()) if gray.size else 255.0
        white_ratio = float((gray > 225).mean()) if gray.size else 1.0
        if white_ratio > 0.58:
            mask = cv2.dilate(
                (gray < 150).astype(np.uint8) * 255,
                np.ones((7, 7), np.uint8),
                iterations=1,
            )
            crop[mask > 0] = 255
        elif mean < 145:
            mask = cv2.dilate(
                ((gray > 210) | (gray < 35)).astype(np.uint8) * 255,
                np.ones((13, 13), np.uint8),
                iterations=1,
            )
            crop = cv2.inpaint(crop, mask, 7, cv2.INPAINT_TELEA)
        else:
            mask = cv2.dilate(
                (gray < 75).astype(np.uint8) * 255,
                np.ones((31, 31), np.uint8),
                iterations=1,
            )
            crop = cv2.inpaint(crop, mask, 7, cv2.INPAINT_TELEA)
        rgb[y0:y1, x0:x1] = crop
        return rgb, mean, white_ratio

    def _draw(self, image: Image.Image, text: str, box: list[int], style: TextStyle):
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = box
        pad_x = max(4, round((x1 - x0) * 0.06))
        pad_y = max(4, round((y1 - y0) * 0.04))
        left, top, right, bottom = x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y
        width, height = max(12, right - left), max(12, bottom - top)
        text = _clean(text)
        if not text:
            return
        font_path = self.bold_font_path if style.bold else self.font_path
        if style.vertical:
            font, size, capacity = _fit_vertical(
                font_path, text, width, height, style.font_size
            )
            chunks = [
                text[index : index + capacity]
                for index in range(0, len(text), capacity)
            ]
            step_x = round(size * (1.12 if style.display else 1.1))
            step_y = round(size * (1.18 if style.display else 1.07))
            start_x = right - max(0, (width - len(chunks) * step_x) / 2) - size
            stroke = round(size * style.stroke_ratio)
            for column, chunk in enumerate(chunks):
                y = top + max(0, (height - len(chunk) * step_y) / 2)
                for row, char in enumerate(chunk):
                    draw.text(
                        (start_x - column * step_x, y + row * step_y),
                        char,
                        font=font,
                        fill=style.fill,
                        stroke_width=stroke,
                        stroke_fill=style.stroke_fill,
                    )
        else:
            font, size, lines, line_height = _fit_horizontal(
                font_path, draw, text, width, height, style.font_size
            )
            stroke = round(size * style.stroke_ratio)
            y = top + max(0, (height - len(lines) * line_height) / 2)
            for line in lines:
                line_width = draw.textbbox(
                    (0, 0), line, font=font, stroke_width=stroke
                )[2]
                draw.text(
                    (left + max(0, (width - line_width) / 2), y),
                    line,
                    font=font,
                    fill=style.fill,
                    stroke_width=stroke,
                    stroke_fill=style.stroke_fill,
                )
                y += line_height

    def render_page(
        self,
        source_path: Path,
        page: PageOCR,
        output_path: Path,
        preserve_sfx: bool = True,
    ) -> dict:
        original = np.array(Image.open(source_path).convert("RGB"))
        rgb = original.copy()
        styles: dict[str, TextStyle] = {}
        active = []
        for unit in page.units:
            if unit.skip or not unit.zh or (preserve_sfx and unit.is_sfx):
                continue
            x0, y0, x1, y1 = unit.bbox
            unit.bbox = [
                max(0, min(page.width, x0)),
                max(0, min(page.height, y0)),
                max(0, min(page.width, x1)),
                max(0, min(page.height, y1)),
            ]
            cx0, cy0, cx1, cy1 = unit.crop_bbox or unit.bbox
            unit.crop_bbox = [
                max(0, min(page.width, cx0)),
                max(0, min(page.height, cy0)),
                max(0, min(page.width, cx1)),
                max(0, min(page.height, cy1)),
            ]
            if unit.bbox[2] > unit.bbox[0] and unit.bbox[3] > unit.bbox[1]:
                active.append(unit)
        for unit in active:
            style = self._analyze_style(original, unit)
            styles[unit.id] = style
            profile = getattr(self, "cleanup_profile", "artwork")
            if profile == "quality" and (
                style.outlined or style.background_std >= 10 or style.fill == "white"
            ):
                rgb = self._erase_with_lama(rgb, unit, style)
                continue
            if profile == "quality":
                # Plain black dialogue on a white balloon does not need neural
                # inpainting, but its furigana often sits just outside the main
                # OCR box.  Clean one expanded reviewed crop with the
                # edge-connected component guard so balloon borders survive.
                cx0, cy0, cx1, cy1 = unit.crop_bbox or unit.bbox
                # OCR boxes frequently cover the main kanji column but omit a
                # neighbouring kana column.  A 96 px high-resolution search
                # margin captures that column; component filtering, not a flat
                # fill, decides which pixels are actually changed.
                margin = min(128, max(96, round(style.font_size * 0.8)))
                bounded = [
                    max(0, cx0 - margin),
                    max(0, cy0 - margin),
                    min(page.width, cx1 + margin),
                    min(page.height, cy1 + margin),
                ]
                rgb, _, _ = self._erase(rgb, bounded)
                continue
            # Grouping several OCR lines is useful for coherent recognition and
            # typesetting, but erasing their union also destroys every drawing
            # between those lines. Only clean the original tight detector boxes.
            erase_boxes = unit.erase_boxes or [unit.bbox]
            for erase_box in erase_boxes:
                ex0, ey0, ex1, ey1 = erase_box
                erase_width, erase_height = ex1 - ex0, ey1 - ey0
                pad_x = min(24, max(8, round(erase_width * 0.06)))
                pad_y = min(24, max(8, round(erase_height * 0.04)))
                bounded = [
                    max(0, min(page.width, ex0 - pad_x)),
                    max(0, min(page.height, ey0 - pad_y)),
                    max(0, min(page.width, ex1 + pad_x)),
                    max(0, min(page.height, ey1 + pad_y)),
                ]
                if bounded[2] > bounded[0] and bounded[3] > bounded[1]:
                    erase = (
                        self._erase_complete if profile == "quality" else self._erase
                    )
                    rgb, _, _ = erase(rgb, bounded)
        rendered = Image.fromarray(rgb)
        for unit in active:
            x0, y0, x1, y1 = unit.bbox
            # Draw into a bounded crop so font antialiasing and stroke pixels
            # cannot spill one pixel beyond the declared typesetting region.
            region = rendered.crop((x0, y0, x1, y1))
            self._draw(region, unit.zh, [0, 0, x1 - x0, y1 - y0], styles[unit.id])
            rendered.paste(region, (x0, y0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".webp":
            # libwebp's exhaustive method 6 is several times slower on full
            # manga pages without improving decoded pixels.  Method 2 remains
            # lossless and keeps first-run batch rendering practical.
            rendered.save(output_path, "WEBP", lossless=True, quality=100, method=2)
        else:
            rendered.save(output_path, "PNG", compress_level=7)
        return {
            "source": source_path.name,
            "output": output_path.name,
            "width": page.width,
            "height": page.height,
            "translated_units": len(active),
        }

    def render_book(
        self,
        source_dir: Path,
        pages: list[PageOCR],
        output_dir: Path,
        preserve_sfx: bool = True,
        output_format: str = "webp",
    ) -> list[dict]:
        manifest = []
        for page in pages:
            source = source_dir / page.file
            suffix = ".webp" if output_format == "webp" else ".png"
            output = output_dir / f"{source.stem}{suffix}"
            manifest.append(self.render_page(source, page, output, preserve_sfx))
        (output_dir / "translation_manifest.json").write_text(
            json.dumps({"pages": manifest}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest
