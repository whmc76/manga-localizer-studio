"""Verify page geometry and bounded pixel edits for a completed localization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from manga_localizer.pipeline import page_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--preserve-sfx", action="store_true")
    return parser.parse_args()


def allowed_mask(page, preserve_sfx: bool) -> np.ndarray:
    mask = np.zeros((page.height, page.width), dtype=bool)
    for unit in page.units:
        if unit.skip or not unit.zh or (preserve_sfx and unit.is_sfx):
            continue
        x0, y0, x1, y1 = unit.bbox
        mask[max(0, y0) : min(page.height, y1), max(0, x0) : min(page.width, x1)] = True
        for ex0, ey0, ex1, ey1 in unit.erase_boxes or [unit.bbox]:
            pad_x = min(24, max(8, round((ex1 - ex0) * 0.06)))
            pad_y = min(24, max(8, round((ey1 - ey0) * 0.04)))
            mask[
                max(0, ey0 - pad_y) : min(page.height, ey1 + pad_y),
                max(0, ex0 - pad_x) : min(page.width, ex1 + pad_x),
            ] = True
    return mask


def main() -> None:
    args = parse_args()
    payload = json.loads(args.transcript.read_text(encoding="utf-8"))
    pages = [page_from_dict(item) for item in payload["pages"]]
    failures = []
    translated = skipped = outside_changes = 0
    for page in pages:
        source_path = args.source / page.file
        output_path = args.output / f"{source_path.stem}.png"
        if not output_path.exists():
            failures.append(f"page {page.page}: missing {output_path.name}")
            continue
        source = np.asarray(Image.open(source_path).convert("RGB"))
        output = np.asarray(Image.open(output_path).convert("RGB"))
        if source.shape != output.shape:
            failures.append(f"page {page.page}: dimensions {source.shape} != {output.shape}")
            continue
        mask = allowed_mask(page, args.preserve_sfx)
        changed = np.any(source != output, axis=2)
        outside = int(np.count_nonzero(changed & ~mask))
        outside_changes += outside
        if outside:
            failures.append(f"page {page.page}: {outside} pixels changed outside text regions")
        translated += sum(
            not unit.skip and bool(unit.zh) and not (args.preserve_sfx and unit.is_sfx)
            for unit in page.units
        )
        skipped += sum(unit.skip or (args.preserve_sfx and unit.is_sfx) for unit in page.units)
        print(f"[{page.page:03d}/{len(pages):03d}] outside={outside}", flush=True)
    report = {
        "pages": len(pages),
        "translated_units": translated,
        "preserved_or_skipped_units": skipped,
        "outside_changed_pixels": outside_changes,
        "failures": failures,
    }
    report_path = args.output / "quality_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
