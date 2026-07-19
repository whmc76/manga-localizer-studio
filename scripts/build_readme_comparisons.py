from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build compact README source/output comparison images."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--pages", type=int, default=3)
    args = parser.parse_args()

    sources = list_images(args.source)[: args.pages]
    outputs = list_images(args.output)[: args.pages]
    if len(sources) != args.pages or len(outputs) != args.pages:
        raise ValueError("Source and output folders must contain every requested page")

    args.destination.mkdir(parents=True, exist_ok=True)
    width, column_width, margin, image_height = 1280, 600, 24, 1033
    header_height, bottom_margin = 76, 24
    title_font, badge_font = load_font(30), load_font(19)
    background, charcoal, coral = (246, 244, 239), (38, 35, 32), (230, 100, 80)

    for page_number, (source, output) in enumerate(zip(sources, outputs), start=1):
        canvas = Image.new(
            "RGB", (width, header_height + image_height + bottom_margin), background
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 18), "原图", font=title_font, fill=charcoal)
        draw.rounded_rectangle(
            (width // 2 + margin, 14, width // 2 + margin + 158, 58),
            radius=14,
            fill=coral,
        )
        draw.text(
            (width // 2 + margin + 17, 20),
            "中文输出",
            font=title_font,
            fill="white",
        )
        draw.text((width - 136, 24), "v0.5 实测", font=badge_font, fill=charcoal)

        for image_path, x in ((source, margin), (output, width // 2 + margin)):
            with Image.open(image_path) as image:
                preview = ImageOps.contain(
                    image.convert("RGB"),
                    (column_width, image_height),
                    Image.Resampling.LANCZOS,
                )
            canvas.paste(preview, (x, header_height))

        destination = args.destination / f"comparison-page-{page_number:02d}.webp"
        canvas.save(destination, "WEBP", lossless=True, method=6)
        print(destination)


if __name__ == "__main__":
    main()
