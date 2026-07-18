from __future__ import annotations

import json
import math
import os
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import AppPaths
from .ocr import PageOCR


FONT_URL = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/"
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
)
SYSTEM_FONT_CANDIDATES = (
    "C:/Windows/Fonts/SourceHanSansCN-Medium.otf",
    "C:/Windows/Fonts/NotoSansSC-VF.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


def managed_font_path(paths: AppPaths | None = None) -> Path:
    root = (paths or AppPaths.from_env()).root
    return root / "fonts" / "NotoSansCJKsc-Regular.otf"


def find_font(paths: AppPaths | None = None) -> Path:
    candidates = (os.environ.get("MLS_FONT"), managed_font_path(paths), *SYSTEM_FONT_CANDIDATES)
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "No CJK font found. Set MLS_FONT to a Simplified Chinese TTF/OTF/TTC file."
    )


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
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".otf.part")
    request = Request(FONT_URL, headers={"User-Agent": "Manga-Localizer-Studio/0.1"})
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


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _clean(text: str) -> str:
    return "".join(text.replace("．．．", "……").replace("．", "。").split())


def _fit_vertical(font_path: Path, text: str, width: int, height: int):
    for size in range(min(92, height), 19, -2):
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


def _fit_horizontal(font_path: Path, draw, text: str, width: int, height: int):
    for size in range(82, 19, -2):
        font = _font(font_path, size)
        lines = _wrap(draw, text, font, width)
        line_height = round(size * 1.24)
        if len(lines) * line_height <= height:
            return font, size, lines, line_height
    font = _font(font_path, 20)
    return font, 20, _wrap(draw, text, font, width), 25


class ArtworkPreservingRenderer:
    def __init__(self, font_path: Path | None = None):
        self.font_path = font_path or ensure_font()

    @staticmethod
    def _erase(rgb: np.ndarray, box: list[int]) -> tuple[np.ndarray, float, float]:
        x0, y0, x1, y1 = box
        crop = rgb[y0:y1, x0:x1].copy()
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        mean = float(gray.mean()) if gray.size else 255.0
        white_ratio = float((gray > 225).mean()) if gray.size else 1.0
        if white_ratio > 0.6:
            mask = (gray < 150).astype(np.uint8) * 255
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
            crop[mask > 0] = 255
        elif mean < 145:
            mask = ((gray > 215) | (gray < 35)).astype(np.uint8) * 255
            mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
            crop = cv2.inpaint(crop, mask, 6, cv2.INPAINT_TELEA)
        else:
            mask = (gray < 75).astype(np.uint8) * 255
            mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=1)
            crop = cv2.inpaint(crop, mask, 6, cv2.INPAINT_TELEA)
        rgb[y0:y1, x0:x1] = crop
        return rgb, mean, white_ratio

    def _draw(self, image: Image.Image, text: str, box: list[int], mean: float, white_ratio: float):
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = box
        pad_x = max(4, round((x1 - x0) * 0.06))
        pad_y = max(4, round((y1 - y0) * 0.04))
        left, top, right, bottom = x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y
        width, height = max(12, right - left), max(12, bottom - top)
        text = _clean(text)
        if not text:
            return
        fill = "white" if mean < 145 and white_ratio < 0.35 else "black"
        stroke_fill = "black" if fill == "white" else "white"
        vertical = height >= width * 1.2
        if vertical:
            font, size, capacity = _fit_vertical(self.font_path, text, width, height)
            chunks = [text[index : index + capacity] for index in range(0, len(text), capacity)]
            step_x, step_y = round(size * 1.1), round(size * 1.07)
            start_x = right - max(0, (width - len(chunks) * step_x) / 2) - size
            stroke = max(2, size // 12) if white_ratio < 0.55 else 0
            for column, chunk in enumerate(chunks):
                y = top + max(0, (height - len(chunk) * step_y) / 2)
                for row, char in enumerate(chunk):
                    draw.text(
                        (start_x - column * step_x, y + row * step_y),
                        char,
                        font=font,
                        fill=fill,
                        stroke_width=stroke,
                        stroke_fill=stroke_fill,
                    )
        else:
            font, size, lines, line_height = _fit_horizontal(
                self.font_path, draw, text, width, height
            )
            stroke = max(2, size // 12) if white_ratio < 0.55 else 0
            y = top + max(0, (height - len(lines) * line_height) / 2)
            for line in lines:
                line_width = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)[2]
                draw.text(
                    (left + max(0, (width - line_width) / 2), y),
                    line,
                    font=font,
                    fill=fill,
                    stroke_width=stroke,
                    stroke_fill=stroke_fill,
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
        stats: dict[str, tuple[float, float]] = {}
        active = []
        for unit in page.units:
            if not unit.zh or (preserve_sfx and unit.is_sfx):
                continue
            x0, y0, x1, y1 = unit.bbox
            unit.bbox = [
                max(0, min(page.width, x0)),
                max(0, min(page.height, y0)),
                max(0, min(page.width, x1)),
                max(0, min(page.height, y1)),
            ]
            if unit.bbox[2] > unit.bbox[0] and unit.bbox[3] > unit.bbox[1]:
                active.append(unit)
        for unit in active:
            rgb, mean, white_ratio = self._erase(rgb, unit.bbox)
            stats[unit.id] = (mean, white_ratio)
        rendered = Image.fromarray(rgb)
        for unit in active:
            self._draw(rendered, unit.zh, unit.bbox, *stats[unit.id])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(output_path, "PNG", compress_level=5)
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
    ) -> list[dict]:
        manifest = []
        for page in pages:
            source = source_dir / page.file
            output = output_dir / f"{source.stem}.png"
            manifest.append(self.render_page(source, page, output, preserve_sfx))
        (output_dir / "translation_manifest.json").write_text(
            json.dumps({"pages": manifest}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest
