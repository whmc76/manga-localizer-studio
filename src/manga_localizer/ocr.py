from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .config import AppPaths, configure_model_caches
from .model_manager import ModelDependencyError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
JP_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
KATAKANA_RE = re.compile(r"^[\u30a0-\u30ffー・…ッっ♡♥！？!?.．]+$")


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Source folder does not exist: {folder}")
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )


@dataclass
class TextUnit:
    id: str
    bbox: list[int]
    crop_bbox: list[int]
    ja: str
    score: float
    is_sfx: bool = False
    zh: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageOCR:
    page: int
    file: str
    width: int
    height: int
    units: list[TextUnit]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["units"] = [unit.to_dict() for unit in self.units]
        return payload


def _area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _axis_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _axis_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _connects(a: dict, b: dict) -> bool:
    abox, bbox = a["box"], b["box"]
    aw, ah = abox[2] - abox[0], abox[3] - abox[1]
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x_gap = _axis_gap(abox[0], abox[2], bbox[0], bbox[2])
    y_gap = _axis_gap(abox[1], abox[3], bbox[1], bbox[3])
    x_overlap = _axis_overlap(abox[0], abox[2], bbox[0], bbox[2])
    y_overlap = _axis_overlap(abox[1], abox[3], bbox[1], bbox[3])
    if ah >= aw * 1.25 and bh >= bw * 1.25:
        return x_gap <= 72 and y_overlap / max(1, min(ah, bh)) >= 0.22
    if aw >= ah * 1.25 and bw >= bh * 1.25:
        return y_gap <= 58 and x_overlap / max(1, min(aw, bw)) >= 0.22
    return x_gap <= 36 and y_gap <= 36


def _groups(regions: list[dict]) -> list[list[dict]]:
    parent = list(range(len(regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if _connects(regions[i], regions[j]):
                union(i, j)
    buckets: dict[int, list[dict]] = {}
    for index, region in enumerate(regions):
        buckets.setdefault(find(index), []).append(region)
    return list(buckets.values())


def _union_box(items: list[dict]) -> list[int]:
    return [
        min(item["box"][0] for item in items),
        min(item["box"][1] for item in items),
        max(item["box"][2] for item in items),
        max(item["box"][3] for item in items),
    ]


class PaddleMangaOCR:
    """Paddle text detection followed by manga-specific Japanese recognition."""

    def __init__(self, paths: AppPaths, device: str = "auto"):
        configure_model_caches(paths)
        try:
            from manga_ocr import MangaOcr
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ModelDependencyError(
                "OCR dependencies are missing. Run scripts/bootstrap with an ML profile."
            ) from exc

        resolved_device = self._resolve_device(device)
        self.detector = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            text_recognition_batch_size=32,
            device=resolved_device,
        )
        self.reader = MangaOcr(force_cpu=resolved_device == "cpu")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import paddle

            if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                return "gpu:0"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _payload(result) -> dict:
        raw = result.json if hasattr(result, "json") else result
        if callable(raw):
            raw = raw()
        return raw.get("res", raw)

    @staticmethod
    def _regions(payload: dict) -> list[dict]:
        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])
        boxes = payload.get("rec_boxes", [])
        selected = []
        for index, raw_box in enumerate(boxes):
            box = [int(value) for value in raw_box]
            if _area(box) < 650:
                continue
            text = str(texts[index]).strip() if index < len(texts) else ""
            score = float(scores[index]) if index < len(scores) else 0.0
            # Keep large detector boxes even when Paddle recognition fails. MangaOCR
            # is the source of truth for Japanese text inside the detected crop.
            if not text and _area(box) < 4_500:
                continue
            selected.append({"box": box, "text": text, "score": score})
        return selected

    def analyze(self, image_path: Path, page_number: int) -> PageOCR:
        image = Image.open(image_path).convert("RGB")
        results = list(self.detector.predict(str(image_path)))
        payload = self._payload(results[0]) if results else {}
        groups = _groups(self._regions(payload))
        groups.sort(key=lambda group: (_union_box(group)[1] // 180, -_union_box(group)[0]))
        units: list[TextUnit] = []
        for index, group in enumerate(groups, start=1):
            box = _union_box(group)
            pad_x = max(10, round((box[2] - box[0]) * 0.04))
            pad_y = max(10, round((box[3] - box[1]) * 0.03))
            crop_box = [
                max(0, box[0] - pad_x),
                max(0, box[1] - pad_y),
                min(image.width, box[2] + pad_x),
                min(image.height, box[3] + pad_y),
            ]
            refined = self.reader(image.crop(tuple(crop_box))).strip()
            if not JP_RE.search(refined):
                continue
            score = max((item["score"] for item in group), default=0.0)
            clean = refined.strip()
            units.append(
                TextUnit(
                    id=f"p{page_number:03d}u{index:02d}",
                    bbox=box,
                    crop_bbox=crop_box,
                    ja=clean,
                    score=round(score, 4),
                    is_sfx=bool(KATAKANA_RE.fullmatch(clean)) and len(clean) <= 14,
                )
            )
        return PageOCR(page_number, image_path.name, image.width, image.height, units)
