"""Tight-crop OCR/role audit plus incremental local quality translation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manga_localizer.config import AppPaths
from manga_localizer.name_dictionary import JapaneseNameDictionary
from manga_localizer.ocr import DEFAULT_OLLAMA_VISION_MODEL, OllamaVisionOCR
from manga_localizer.pipeline import page_from_dict
from manga_localizer.translator import (
    DEFAULT_LOCAL_TRANSLATION_MODEL,
    HyMTTranslator,
    LocalQualityTranslator,
    OllamaTranslator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--vision-model", default=DEFAULT_OLLAMA_VISION_MODEL)
    parser.add_argument("--editor-model", default=DEFAULT_LOCAL_TRANSLATION_MODEL)
    parser.add_argument("--target", default="简体中文")
    parser.add_argument("--pages", type=int, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = AppPaths.from_env().ensure()
    payload = json.loads(args.transcript.read_text(encoding="utf-8"))
    pages = [page_from_dict(item) for item in payload["pages"]]
    glossary = dict(payload.get("glossary", {}))
    selected = set(args.pages or [page.page for page in pages])
    vision = OllamaVisionOCR(args.base_url, args.vision_model, timeout=900)
    changed_ids: set[str] = set()

    def save() -> None:
        result = {
            "source": payload.get("source", str(args.source.resolve())),
            "pages": [page.to_dict() for page in pages],
            "glossary": glossary,
        }
        temporary = args.output.with_suffix(args.output.suffix + ".part")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(args.output)

    for current, page in enumerate(pages, 1):
        if page.page not in selected:
            continue
        before = {unit.id: (unit.ja, unit.is_sfx) for unit in page.units}
        vision.refine_local_crops(args.source / page.file, page)
        page_changes = 0
        for unit in page.units:
            if before[unit.id] == (unit.ja, unit.is_sfx):
                continue
            changed_ids.add(unit.id)
            page_changes += 1
            unit.zh = ""
            unit.translation_attempts = []
        save()
        print(
            f"crop audit {current}/{len(pages)} page={page.page} changed={page_changes}",
            flush=True,
        )

    unresolved = {
        unit.id
        for page in pages
        for unit in page.units
        if page.page in selected and not unit.is_sfx and not unit.skip and not unit.zh
    }
    print(
        f"crop audit changed={len(changed_ids)} needs_translation={len(unresolved)}",
        flush=True,
    )
    if unresolved:
        names = JapaneseNameDictionary(paths.cache)
        translator = LocalQualityTranslator(
            OllamaTranslator(args.base_url, args.editor_model, args.target, names),
            HyMTTranslator(paths.models / "hy-mt2", args.target, "auto", names),
        )
        try:
            translator.translate_pages(
                pages,
                context_pages=6,
                story_context=True,
                preserve_sfx=True,
                glossary=glossary,
                progress=lambda current, total: print(
                    f"incremental translation {current}/{total}", flush=True
                ),
            )
            glossary = dict(translator.resolved_glossary or glossary)
            selected_ids = {
                unit.id
                for page in pages
                for unit in page.units
                if page.page in selected and not unit.is_sfx and not unit.skip
            }
            remaining = (
                translator.auditor.residual_quality_ids(pages, glossary) & selected_ids
            )
            if remaining:
                sample = ", ".join(sorted(remaining)[:12])
                raise RuntimeError(
                    f"Residual quality gate rejects {len(remaining)} unit(s): {sample}"
                )
        finally:
            translator.editor.unload()
            translator.candidate.unload()
            translator.auditor.unload()
    save()
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
