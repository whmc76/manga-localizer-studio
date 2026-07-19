from __future__ import annotations

import argparse
import json
from pathlib import Path

from manga_localizer.pipeline import page_from_dict
from manga_localizer.renderer import ArtworkPreservingRenderer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a benchmark transcript losslessly."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--pages",
        type=int,
        nargs="+",
        help="Render only these 1-based page numbers for a visual spot check.",
    )
    args = parser.parse_args()

    payload = json.loads(args.transcript.read_text(encoding="utf-8"))
    pages = [page_from_dict(item) for item in payload["pages"]]
    if args.pages:
        selected = set(args.pages)
        pages = [page for page in pages if page.page in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    renderer = ArtworkPreservingRenderer(cleanup_profile="quality")
    manifest = []
    for index, page in enumerate(pages, start=1):
        source = args.source / page.file
        output = args.output / f"{source.stem}.webp"
        item = renderer.render_page(source, page, output, preserve_sfx=True)
        manifest.append(item)
        print(f"render {index}/{len(pages)} {item['output']}", flush=True)
    (args.output / "benchmark_manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "transcript": str(args.transcript.resolve()),
                "images": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
