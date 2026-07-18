from __future__ import annotations

import base64
import json
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import AppPaths, configure_model_caches
from .model_manager import ModelDependencyError


def manga_force_cpu(device: str, cuda_available: bool) -> bool:
    """Keep Manga OCR on CUDA independently from Paddle's device backend."""
    return device == "cpu" or not cuda_available


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
JP_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
KATAKANA_RE = re.compile(r"^[\u30a0-\u30ffー・…ッっ♡♥！？!?.．]+$")
EXPLICIT_SKIP_REASONS = frozenset({"duplicate", "noise", "decorative", "preserve"})
UNRESOLVED_SKIP_REASON = "unresolved"


@contextmanager
def _suppress_optional_ccache_warning():
    """Hide Paddle's compile-cache notice without hiding inference warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"No ccache found\..*",
            category=UserWarning,
        )
        yield


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
    skip: bool = False
    # A skipped unit is only complete when a reviewer records an explicit,
    # machine-checkable reason. Legacy ``skip: true`` values are imported as
    # ``unresolved`` so a missed translation can never silently pass QA.
    skip_reason: str = ""
    # Keep the detector's tight line boxes. ``bbox`` is the union used for
    # translation layout, while these boxes constrain destructive cleanup to
    # the pixels that actually contained source text.
    erase_boxes: list[list[int]] = field(default_factory=list)
    # Optional semantic layout hint recorded by a reviewed transcript.  It is
    # never required for ordinary OCR output, but lets cover titles and other
    # display lettering keep their intended composition without page-specific
    # hard-coding in the renderer.
    special: str = ""

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


def _light_on_dark_regions(image: Image.Image) -> list[dict]:
    """Find title-like light lettering that general-purpose OCR often misses.

    Paddle's detector is strongest on dark text over light balloons.  Manga
    title cards and captions frequently reverse that polarity, so this adds a
    conservative, model-independent candidate pass.  It only proposes
    horizontal or vertical runs made from several bright components surrounded
    by a mostly dark local background; MangaOCR still has to recognize Japanese
    before a candidate becomes a TextUnit.
    """
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    page_area = height * width
    bright = (gray >= 200).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    accepted = np.zeros_like(bright)
    min_component_area = max(80, round(page_area * 0.00001))
    for label in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[label])
        if area < min_component_area or box_width < 7 or box_height < 7:
            continue
        if box_width > width * 0.35 or box_height > height * 0.35:
            continue
        pad = max(12, round(max(box_width, box_height) * 0.18))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + box_width + pad), min(height, y + box_height + pad)
        local = gray[y0:y1, x0:x1]
        if local.size == 0 or float((local < 105).mean()) < 0.45:
            continue
        accepted[labels == label] = 1

    if not accepted.any():
        return []
    kernels = (
        np.ones((max(5, round(height * 0.006)), max(15, round(width * 0.055))), np.uint8),
        np.ones((max(15, round(height * 0.035)), max(5, round(width * 0.008))), np.uint8),
    )
    candidates: list[dict] = []
    for kernel in kernels:
        joined = cv2.morphologyEx(accepted, cv2.MORPH_CLOSE, kernel)
        joined_count, joined_labels, joined_stats, _ = cv2.connectedComponentsWithStats(joined, 8)
        for label in range(1, joined_count):
            x, y, box_width, box_height, _ = (int(value) for value in joined_stats[label])
            box_area = box_width * box_height
            if box_area < page_area * 0.0015 or box_area > page_area * 0.25:
                continue
            aspect = max(box_width / max(1, box_height), box_height / max(1, box_width))
            if aspect < 1.8:
                continue
            member_count = len(np.unique(labels[joined_labels == label])) - 1
            if member_count < 3:
                continue
            local = gray[y : y + box_height, x : x + box_width]
            if float((local < 105).mean()) < 0.4:
                continue
            box = [x, y, x + box_width, y + box_height]
            if any(
                _axis_overlap(box[0], box[2], item["box"][0], item["box"][2])
                * _axis_overlap(box[1], box[3], item["box"][1], item["box"][3])
                >= min(_area(box), _area(item["box"])) * 0.7
                for item in candidates
            ):
                continue
            candidates.append({"box": box, "text": "", "score": 0.5, "reverse": True})
    return candidates


class PaddleMangaOCR:
    """Paddle text localization followed by manga-specific Japanese recognition.

    ``quality`` uses Paddle's recognition boxes as a second signal.  MangaOCR
    remains the Japanese source of truth, but the recognizer filters detector
    noise and produces line boxes that group substantially better on manga.
    ``fast`` keeps the detection-only path for previews and low-end machines.
    """

    def __init__(self, paths: AppPaths, device: str = "auto", profile: str = "quality"):
        configure_model_caches(paths)
        try:
            from manga_ocr import MangaOcr
            # Paddle imports its C++ extension helper even for ordinary inference.
            # ccache is only useful when compiling custom extensions, so do not
            # present its absence as a broken runtime component to desktop users.
            with _suppress_optional_ccache_warning():
                from paddleocr import PaddleOCR, TextDetection
        except ImportError as exc:
            raise ModelDependencyError(
                "OCR dependencies are missing. Run scripts/bootstrap with an ML profile."
            ) from exc

        resolved_device = self._resolve_device(device)
        if profile not in {"quality", "fast"}:
            raise ValueError(f"Unknown OCR profile: {profile}")
        self.profile = profile
        with _suppress_optional_ccache_warning():
            if profile == "quality":
                self.detector = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_server_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    device=resolved_device,
                    text_recognition_batch_size=32,
                    enable_mkldnn=False,
                )
            else:
                self.detector = TextDetection(
                    model_name="PP-OCRv5_mobile_det",
                    device=resolved_device,
                    enable_mkldnn=False,
                )
        try:
            import torch

            torch_cuda = torch.cuda.is_available()
        except (ImportError, RuntimeError):
            torch_cuda = False
        self.reader = MangaOcr(force_cpu=manga_force_cpu(device, torch_cuda))

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
        if "dt_polys" in payload:
            selected = []
            scores = payload.get("dt_scores", [])
            for index, polygon in enumerate(payload.get("dt_polys", [])):
                xs = [int(point[0]) for point in polygon]
                ys = [int(point[1]) for point in polygon]
                box = [min(xs), min(ys), max(xs), max(ys)]
                if _area(box) < 650:
                    continue
                score = float(scores[index]) if index < len(scores) else 0.0
                selected.append(
                    {
                        "box": box,
                        "text": "",
                        "score": score,
                    }
                )
            return selected
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
            has_japanese = bool(JP_RE.search(text))
            kana_count = len(re.findall(r"[\u3040-\u30ff]", text))
            large_detector_box = _area(box) >= 5_000
            confident_effect = score >= 0.72 and _area(box) >= 1_200 and bool(text)
            if not (has_japanese or large_detector_box or confident_effect):
                continue
            if (
                has_japanese
                and score < 0.58
                and not (score >= 0.42 and kana_count >= 2)
                and not large_detector_box
            ):
                continue
            selected.append({"box": box, "text": text, "score": score})
        return selected

    def analyze(self, image_path: Path, page_number: int) -> PageOCR:
        image = Image.open(image_path).convert("RGB")
        results = list(self.detector.predict(str(image_path)))
        payload = self._payload(results[0]) if results else {}
        regions = self._regions(payload)
        for candidate in _light_on_dark_regions(image):
            candidate_box = candidate["box"]
            overlaps_detector = any(
                _axis_overlap(candidate_box[0], candidate_box[2], item["box"][0], item["box"][2])
                * _axis_overlap(candidate_box[1], candidate_box[3], item["box"][1], item["box"][3])
                >= min(_area(candidate_box), _area(item["box"])) * 0.65
                for item in regions
            )
            if not overlaps_detector:
                regions.append(candidate)
        groups = _groups(regions)
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
            crop = image.crop(tuple(crop_box))
            if any(item.get("reverse", False) for item in group):
                crop = ImageOps.invert(crop)
            refined = self.reader(crop).strip()
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
                    is_sfx=(
                        (bool(KATAKANA_RE.fullmatch(clean)) and len(clean) <= 14)
                    ),
                    erase_boxes=[item["box"] for item in group],
                )
            )
        return PageOCR(page_number, image_path.name, image.width, image.height, units)


class OllamaVisionOCR:
    """Optional local vision-OCR adapter using Ollama's native chat endpoint.

    The specialized Paddle/Manga OCR path remains the accuracy-first default.
    This adapter exists so users with an Ollama vision model can use the same
    local service boundary for OCR and translation without any cloud API.
    """

    def __init__(self, base_url: str, model: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout = timeout
        if not self.model:
            raise ValueError("An Ollama vision model is required for Ollama OCR")

    @staticmethod
    def _json_payload(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Ollama OCR did not return JSON")
            return json.loads(content[start : end + 1])

    def analyze(self, image_path: Path, page_number: int) -> PageOCR:
        with Image.open(image_path) as image:
            width, height = image.size
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        prompt = (
            "Detect every Japanese text region in this manga page, including dialogue, "
            "narration, furigana and sound effects. Return JSON only as "
            '{"regions":[{"bbox":[x0,y0,x1,y1],"text":"...",'
            '"score":0.0,"is_sfx":false}]}. Coordinates must be integer source-image '
            f"pixels within width={width}, height={height}. Do not translate or omit text."
        )
        body = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
                "messages": [
                    {"role": "user", "content": prompt, "images": [encoded]}
                ],
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Ollama OCR request failed: {exc}") from exc
        content = str(payload.get("message", {}).get("content", ""))
        regions = self._json_payload(content).get("regions", [])
        units: list[TextUnit] = []
        for region in regions[:300]:
            try:
                raw = [float(value) for value in region["bbox"]]
                if len(raw) != 4:
                    continue
                if all(0 <= value <= 1 for value in raw):
                    raw = [raw[0] * width, raw[1] * height, raw[2] * width, raw[3] * height]
                box = [
                    max(0, min(width, round(raw[0]))),
                    max(0, min(height, round(raw[1]))),
                    max(0, min(width, round(raw[2]))),
                    max(0, min(height, round(raw[3]))),
                ]
                text = str(region.get("text", "")).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if box[2] <= box[0] or box[3] <= box[1] or not JP_RE.search(text):
                continue
            pad_x = max(8, round((box[2] - box[0]) * 0.04))
            pad_y = max(8, round((box[3] - box[1]) * 0.03))
            units.append(
                TextUnit(
                    id="",
                    bbox=box,
                    crop_bbox=[
                        max(0, box[0] - pad_x),
                        max(0, box[1] - pad_y),
                        min(width, box[2] + pad_x),
                        min(height, box[3] + pad_y),
                    ],
                    ja=text,
                    score=float(region.get("score", 0.75)),
                    is_sfx=bool(region.get("is_sfx", False))
                    or (bool(KATAKANA_RE.fullmatch(text)) and len(text) <= 14),
                    erase_boxes=[box],
                )
            )
        units.sort(key=lambda unit: (unit.bbox[1] // 180, -unit.bbox[0]))
        for index, unit in enumerate(units, start=1):
            unit.id = f"p{page_number:03d}u{index:02d}"
        return PageOCR(page_number, image_path.name, width, height, units)
