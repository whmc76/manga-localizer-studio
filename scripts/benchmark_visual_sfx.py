from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from manga_localizer.ocr import DEFAULT_OLLAMA_VISION_MODEL, OllamaVisionOCR


KINDS = [
    "heartbeat",
    "impact",
    "engine",
    "movement",
    "friction",
    "liquid",
    "breath",
    "vocalization",
    "ambience",
    "other",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate visual-context translation of known manga SFX."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--effects",
        required=True,
        help=(
            "JSON object mapping stable ids either to Japanese text or to "
            '{"text":"...","bbox":[x0,y0,x1,y1]}.'
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_VISION_MODEL)
    args = parser.parse_args()

    raw_effects = json.loads(args.effects)
    effects = {
        key: value if isinstance(value, dict) else {"text": str(value)}
        for key, value in raw_effects.items()
    }
    ids = list(effects)
    listed = "\n".join(
        f"- {key}: {value.get('text', '')}" for key, value in effects.items()
    )
    item_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": ids},
            "kind": {"type": "string", "enum": KINDS},
            "zh": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["id", "kind", "zh", "confidence"],
    }
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": item_schema}},
        "required": ["items"],
    }
    prompt = f"""你会收到同一张漫画的原图、一张红框编号图，以及按清单顺序排列的局部上下文放大图。红框编号与清单顺序一致，标记不属于漫画。
结合每个红框所在分镜的动作、人物表情和相邻对白，判断下面每个已确认日语效果字的实际声音类别，并翻译成自然、简短的简体中文效果字。
只处理清单内的 id；必须逐项返回。zh 应像漫画印刷效果字，不得写人物、解释或剧情，不得机械音译；保留重复、强弱和标点。

待判断效果字：
{listed}"""
    with Image.open(args.image) as source:
        annotated = source.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    line_width = max(5, round(min(annotated.size) * 0.004))
    for index, value in enumerate(effects.values(), start=1):
        box = value.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        draw.rectangle(tuple(box), outline="#ff2d2d", width=line_width)
        draw.text((box[0] + line_width, box[1] + line_width), str(index), fill="red")
    overlay = BytesIO()
    annotated.save(overlay, "JPEG", quality=92, subsampling=0)
    context_crops: list[bytes] = []
    for value in effects.values():
        box = value.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x0, y0, x1, y1 = box
        region_width = max(1, x1 - x0)
        region_height = max(1, y1 - y0)
        pad_x = max(500, round(region_width * 1.5))
        pad_y = max(500, round(region_height * 1.5))
        crop = annotated.crop(
            (
                max(0, x0 - pad_x),
                max(0, y0 - pad_y),
                min(annotated.width, x1 + pad_x),
                min(annotated.height, y1 + pad_y),
            )
        )
        output = BytesIO()
        crop.save(output, "JPEG", quality=94, subsampling=0)
        context_crops.append(output.getvalue())
    client = OllamaVisionOCR(args.base_url, args.model, timeout=900)
    result = client._chat(
        [args.image.read_bytes(), overlay.getvalue(), *context_crops], prompt, schema
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
