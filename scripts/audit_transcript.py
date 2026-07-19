"""Run the project's residual local quality gate on an existing transcript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manga_localizer.config import AppPaths
from manga_localizer.name_dictionary import JapaneseNameDictionary
from manga_localizer.ocr import (
    duplicate_tiny_fragment,
    malformed_tiny_ocr,
    tiny_low_confidence_nontext,
)
from manga_localizer.pipeline import page_from_dict
from manga_localizer.translator import DEFAULT_LOCAL_TRANSLATION_MODEL, OllamaTranslator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=DEFAULT_LOCAL_TRANSLATION_MODEL)
    parser.add_argument("--target", default="简体中文")
    parser.add_argument("--context-review", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.transcript.read_text(encoding="utf-8"))
    pages = [page_from_dict(item) for item in payload["pages"]]
    for page in pages:
        for unit in page.units:
            if (
                tiny_low_confidence_nontext(page, unit)
                or malformed_tiny_ocr(page, unit)
                or duplicate_tiny_fragment(page, unit)
            ):
                unit.is_sfx = True
                unit.special = unit.special or "ocr_duplicate"
    paths = AppPaths.from_env().ensure()
    glossary = dict(payload.get("glossary", {}))
    auditor = OllamaTranslator(
        args.base_url,
        args.model,
        args.target,
        JapaneseNameDictionary(paths.cache),
    )
    auditor.translation_register = auditor._detect_translation_register(pages)
    glossary = auditor._resolve_glossary(pages, glossary)
    auditor._restore_named_dialogue_roles(pages)
    auditor.resolved_glossary = dict(glossary)

    def save() -> None:
        result = {
            "source": payload.get("source", ""),
            "pages": [page.to_dict() for page in pages],
            "glossary": glossary,
        }
        temporary = args.output.with_suffix(args.output.suffix + ".part")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(args.output)

    try:
        if args.context_review:
            auditor.review_pages(
                pages,
                glossary,
                context_pages=6,
                preserve_sfx=True,
                progress=lambda current, total: print(
                    f"context review {current}/{total}", flush=True
                ),
            )
            save()
        for attempt in range(1, 4):
            repaired = auditor.apply_deterministic_fallbacks(pages, glossary)
            if repaired:
                print(f"deterministic fallback: {len(repaired)} unit(s)", flush=True)
            pending = auditor.residual_quality_ids(pages, glossary)
            print(f"residual audit pass {attempt}: {len(pending)} unit(s)", flush=True)
            if not pending:
                break
            auditor.retranslate_risk_units(
                pages,
                pending,
                glossary,
                preserve_sfx=True,
                progress=lambda current, total: print(
                    f"audit {current}/{total}", flush=True
                ),
                forced_ids=pending,
            )
            save()
        remaining = auditor.residual_quality_ids(pages, glossary)
        if remaining:
            save()
            sample = ", ".join(sorted(remaining)[:12])
            raise RuntimeError(
                f"Residual quality gate still rejects {len(remaining)} unit(s): {sample}"
            )
    finally:
        auditor.unload()
    save()
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
