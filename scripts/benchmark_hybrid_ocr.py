from __future__ import annotations

import argparse
import json
from pathlib import Path

from manga_localizer.config import AppPaths
from manga_localizer.ocr import DEFAULT_OLLAMA_VISION_MODEL, HybridMangaOCR, list_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the production hybrid OCR on selected source pages."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--pages", required=True, help="Comma-separated 1-based pages")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_VISION_MODEL)
    args = parser.parse_args()

    selected = {int(value) for value in args.pages.split(",")}
    paths = AppPaths.from_env().ensure()
    service = HybridMangaOCR(paths, args.base_url, args.model, profile="quality")
    pages = []
    for index, image in enumerate(list_images(args.source), start=1):
        if index not in selected:
            continue
        print(f"ocr {index}: {image.name}", flush=True)
        page = service.analyze(image, index)
        pages.append(page.to_dict())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
