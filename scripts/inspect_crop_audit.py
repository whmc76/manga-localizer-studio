from __future__ import annotations

import argparse
import json
from pathlib import Path

from manga_localizer.ocr import DEFAULT_OLLAMA_VISION_MODEL, OllamaVisionOCR
from manga_localizer.pipeline import page_from_dict


class TracedVisionOCR(OllamaVisionOCR):
    def _chat(self, images: list[bytes], prompt: str, schema: dict) -> dict:
        result = super()._chat(images, prompt, schema)
        print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect exact-crop VLM audit output.")
    parser.add_argument("image", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_VISION_MODEL)
    args = parser.parse_args()

    payload = json.loads(args.transcript.read_text(encoding="utf-8"))
    raw_page = next(item for item in payload["pages"] if item["page"] == args.page)
    page = page_from_dict(raw_page)
    result = TracedVisionOCR(args.base_url, args.model, timeout=900).refine_local_crops(
        args.image, page
    )
    print(
        json.dumps(
            {
                "final": [
                    {
                        "id": unit.id,
                        "text": unit.ja,
                        "is_sfx": unit.is_sfx,
                        "skip_reason": unit.skip_reason,
                        "special": unit.special,
                    }
                    for unit in result.units
                ]
            },
            ensure_ascii=True,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
