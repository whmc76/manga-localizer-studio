from __future__ import annotations

import base64
import json
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

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
DEFAULT_OLLAMA_VISION_MODEL = "huihui_ai/qwen3.5-abliterated:9b"


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
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def likely_sfx_text(text: str) -> bool:
    """Conservatively recognize short sound effects from OCR text alone.

    MangaOCR occasionally prefixes a repeated katakana sound with one or two
    hallucinated hiragana characters.  Treating those strings as dialogue
    creates conspicuous nonsense translations such as phonetic gibberish.
    Mixed-kana text is accepted only when a katakana bigram actually repeats,
    so ordinary phrases containing a loanword are left as dialogue.
    """
    clean = text.strip()
    if not clean or len(clean) > 14:
        return False
    if KATAKANA_RE.fullmatch(clean):
        return True
    if re.search(r"[\u3400-\u9fff]", clean):
        return False
    katakana = "".join(re.findall(r"[ァ-ヺー]", clean))
    hiragana_count = len(re.findall(r"[ぁ-ゖ]", clean))
    if len(katakana) < 4 or hiragana_count > 3:
        return False
    return any(
        katakana.count(katakana[index : index + 2]) >= 2
        for index in range(len(katakana) - 1)
    )


def semantic_sfx_classification(text: str, score: float, vision_is_sfx: bool) -> bool:
    """Resolve unsafe short-fragment and VLM role classifications conservatively.

    A vision model can read an isolated brush-stroke fragment as a grammatical
    word and then invite the translator to invent dialogue around it.  Very
    short, low-confidence OCR is therefore preserved as display lettering/noise
    instead of translated.  Conversely, a confident hiragana phrase ending in
    a Japanese comma is ordinary narration/dialogue even when the VLM labels it
    as an effect (for example ``ついに、``).
    """
    clean = re.sub(r"\s+", "", text.strip())
    semantic_chars = re.sub(r"[．。…、，,!?！？:：♡♥〰〜～]", "", clean)
    if likely_sfx_text(clean):
        return True
    hiragana_count = len(re.findall(r"[ぁ-ゖ]", semantic_chars))
    katakana_count = len(re.findall(r"[ァ-ヺ]", semantic_chars))
    # Punctuation is stronger linguistic evidence than an absent detector
    # score. Paddle's detection-only payload commonly reports ``0`` even for
    # perfectly readable text, so short dialogue such as ``いや、`` and
    # ``ん？`` must not be silently preserved as sound effects.
    has_kanji = bool(re.search(r"[\u3400-\u9fff]", semantic_chars))
    has_dialogue_punctuation = bool(re.search(r"[．。…、，,?？]", clean))
    if has_kanji:
        return False
    if hiragana_count and (
        hiragana_count >= 2
        or hiragana_count + katakana_count >= 3
        or has_dialogue_punctuation
    ):
        return False
    if katakana_count and has_dialogue_punctuation:
        return False
    if score < 0.70 and len(semantic_chars) <= 3:
        return True
    return vision_is_sfx


def tiny_low_confidence_nontext(page: PageOCR, unit: TextUnit) -> bool:
    """Identify tiny detector fragments that cannot support their OCR string."""
    width = max(1, unit.bbox[2] - unit.bbox[0])
    height = max(1, unit.bbox[3] - unit.bbox[1])
    semantic = re.sub(r"[\s.．。…、，,!?！？:：♡♥〰〜～『』「」]", "", unit.ja)
    if not semantic or unit.score > 0.05:
        return False
    unbalanced_delimiter = unit.ja.count("『") != unit.ja.count("』") or unit.ja.count(
        "「"
    ) != unit.ja.count("」")
    return (
        unbalanced_delimiter
        and height <= max(32, round(page.height * 0.012))
        and width * height / max(1, len(semantic)) < 420
    )


def malformed_tiny_ocr(page: PageOCR, unit: TextUnit) -> bool:
    """Reject implausible repeated-kanji OCR found in tiny artwork patches."""
    width = max(1, unit.bbox[2] - unit.bbox[0])
    height = max(1, unit.bbox[3] - unit.bbox[1])
    compact = re.sub(r"\s+", "", unit.ja)
    return bool(
        unit.score < 0.8
        and width <= page.width * 0.02
        and height <= page.height * 0.04
        and re.match(r"^([\u3400-\u9fff])\1", compact)
    )


def duplicate_tiny_fragment(page: PageOCR, unit: TextUnit) -> bool:
    """Detect ruby fragments and VLM crop echoes of a stronger detector unit."""
    width = max(1, unit.bbox[2] - unit.bbox[0])
    height = max(1, unit.bbox[3] - unit.bbox[1])
    is_tiny = width <= page.width * 0.04 and height <= page.height * 0.03
    area = width * height

    def semantic(text: str) -> str:
        normalized = text.translate(
            str.maketrans(
                "ぁぃぅぇぉゃゅょっァィゥェォャュョッ",
                "あいうえおやゆよつアイウエオヤユヨツ",
            )
        )
        return re.sub(r"[^ぁ-ゖァ-ヺー\u3400-\u9fff]", "", normalized)

    compact = semantic(unit.ja)
    kanji = "".join(re.findall(r"[\u3400-\u9fff]", compact))
    if len(compact) < 2:
        return False
    for other in page.units:
        if other.id == unit.id:
            continue
        other_area = max(1, other.bbox[2] - other.bbox[0]) * max(
            1, other.bbox[3] - other.bbox[1]
        )
        other_compact = semantic(other.ja)
        repeated = compact == other_compact or (
            len(compact) >= 3 and compact in other_compact
        )
        repeated = repeated or (len(kanji) >= 2 and kanji in other_compact)
        if not repeated:
            continue
        if is_tiny and other_area >= area * 4:
            return True
        if len(compact) >= 5 and other_area >= area * 4:
            return True
        other_width = max(1, other.bbox[2] - other.bbox[0])
        other_height = max(1, other.bbox[3] - other.bbox[1])
        weak_union_box = len(unit.erase_boxes) <= 1 and width >= height * 0.8
        stronger_vertical_boxes = (
            len(other.erase_boxes) >= 2 and other_height >= other_width * 1.4
        )
        if len(compact) >= 5 and weak_union_box and stronger_vertical_boxes:
            return True
        if (
            len(compact) >= 5
            and other_area >= area * 1.6
            and len(other.erase_boxes) > len(unit.erase_boxes)
            and other_height >= other_width * 1.2
        ):
            return True
    return False


def prefer_semantic_ocr(current: str, candidate: str, score: float) -> bool:
    """Accept contextual OCR corrections without replacing complete text by a prefix."""
    if score < 0.72 or not JP_RE.search(candidate):
        return False

    def normalize(value: str) -> str:
        return re.sub(r"[\s．。…、，,!?！？:：]", "", value)

    current_clean = normalize(current)
    candidate_clean = normalize(candidate)
    if not current_clean:
        return True
    # Vision models sometimes stop after the first clause while assigning high
    # confidence. If the proposed text is merely a short prefix/subsequence of
    # a longer detector crop, MangaOCR contains strictly more source evidence.
    if (
        len(candidate_clean) < len(current_clean) * 0.72
        and candidate_clean
        and candidate_clean in current_clean
    ):
        return False
    return True


def merge_region_candidates(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Union detector candidates without letting recognition delete text boxes."""
    merged = list(primary)
    for candidate in secondary:
        candidate_box = candidate["box"]
        duplicate = any(
            _axis_overlap(
                candidate_box[0], candidate_box[2], item["box"][0], item["box"][2]
            )
            * _axis_overlap(
                candidate_box[1], candidate_box[3], item["box"][1], item["box"][3]
            )
            >= min(_area(candidate_box), _area(item["box"])) * 0.65
            for item in merged
        )
        if not duplicate:
            merged.append(candidate)
    return merged


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Source folder does not exist: {folder}")
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
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
    # Local-only audit trail for failed/repair translation attempts. Persisted
    # in draft transcripts so a quality-gate failure remains diagnosable.
    translation_attempts: list[str] = field(default_factory=list)
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
    # Full-page vision may see text outside detector-owned geometry. These
    # findings are audit-only: without exact detector boxes they must never be
    # erased or rendered automatically.
    semantic_missing: list[dict] = field(default_factory=list)

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


def _cover_title_units(
    units: list[TextUnit], page_width: int, page_height: int
) -> list[TextUnit]:
    """Return a conservative full-page display-title group, if one exists."""
    if not 2 <= len(units) <= 6:
        return []
    page_area = max(1, page_width * page_height)
    box = [
        min(unit.bbox[0] for unit in units),
        min(unit.bbox[1] for unit in units),
        max(unit.bbox[2] for unit in units),
        max(unit.bbox[3] for unit in units),
    ]
    width, height = box[2] - box[0], box[3] - box[1]
    total_area = sum(_area(unit.bbox) for unit in units)
    centers = [(unit.bbox[0] + unit.bbox[2]) / 2 for unit in units]
    if not (
        height >= page_height * 0.55
        and page_width * 0.18 <= width <= page_width * 0.58
        and width * height >= page_area * 0.15
        and total_area >= page_area * 0.05
        and min(centers) >= page_width * 0.24
        and max(centers) <= page_width * 0.76
        and sum(len(unit.ja) for unit in units) >= 6
    ):
        return []
    return units


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
        np.ones(
            (max(5, round(height * 0.006)), max(15, round(width * 0.055))), np.uint8
        ),
        np.ones(
            (max(15, round(height * 0.035)), max(5, round(width * 0.008))), np.uint8
        ),
    )
    candidates: list[dict] = []
    for kernel in kernels:
        joined = cv2.morphologyEx(accepted, cv2.MORPH_CLOSE, kernel)
        joined_count, joined_labels, joined_stats, _ = cv2.connectedComponentsWithStats(
            joined, 8
        )
        for label in range(1, joined_count):
            x, y, box_width, box_height, _ = (
                int(value) for value in joined_stats[label]
            )
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
            # Recognition-independent detection is the source of truth.  A
            # Chinese recognizer can score valid Japanese too low, so its
            # richer output is additive and can never remove detector boxes.
            self.detector = TextDetection(
                model_name="PP-OCRv5_mobile_det",
                device=resolved_device,
                enable_mkldnn=False,
            )
            self.recognition_detector = None
            if profile == "quality":
                self.recognition_detector = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_server_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    device=resolved_device,
                    text_recognition_batch_size=32,
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

    @staticmethod
    def is_cover_title_candidate(page: PageOCR) -> bool:
        return bool(_cover_title_units(page.units, page.width, page.height))

    def refine_cover_title(self, image_path: Path, page: PageOCR) -> PageOCR:
        candidates = _cover_title_units(page.units, page.width, page.height)
        if not candidates:
            return page
        image = Image.open(image_path).convert("RGB")
        box = [
            min(unit.bbox[0] for unit in candidates),
            min(unit.bbox[1] for unit in candidates),
            max(unit.bbox[2] for unit in candidates),
            max(unit.bbox[3] for unit in candidates),
        ]
        pad_x = max(16, round((box[2] - box[0]) * 0.04))
        pad_y = max(16, round((box[3] - box[1]) * 0.02))
        crop_box = [
            max(0, box[0] - pad_x),
            max(0, box[1] - pad_y),
            min(page.width, box[2] + pad_x),
            min(page.height, box[3] + pad_y),
        ]
        refined = self.reader(image.crop(tuple(crop_box))).strip()
        if not JP_RE.search(refined):
            return page
        erase_boxes = [
            erase_box
            for unit in candidates
            for erase_box in (unit.erase_boxes or [unit.bbox])
        ]
        merged = TextUnit(
            id=candidates[0].id,
            bbox=box,
            crop_bbox=crop_box,
            ja=refined,
            score=max(unit.score for unit in candidates),
            is_sfx=False,
            erase_boxes=erase_boxes,
            special="cover_title",
        )
        return PageOCR(page.page, page.file, page.width, page.height, [merged])

    def analyze(self, image_path: Path, page_number: int) -> PageOCR:
        image = Image.open(image_path).convert("RGB")
        results = list(self.detector.predict(str(image_path)))
        payload = self._payload(results[0]) if results else {}
        regions = self._regions(payload)
        if self.recognition_detector is not None:
            recognition_results = list(
                self.recognition_detector.predict(str(image_path))
            )
            recognition_payload = (
                self._payload(recognition_results[0]) if recognition_results else {}
            )
            regions = merge_region_candidates(
                regions, self._regions(recognition_payload)
            )
        for candidate in _light_on_dark_regions(image):
            candidate_box = candidate["box"]
            overlaps_detector = any(
                _axis_overlap(
                    candidate_box[0], candidate_box[2], item["box"][0], item["box"][2]
                )
                * _axis_overlap(
                    candidate_box[1], candidate_box[3], item["box"][1], item["box"][3]
                )
                >= min(_area(candidate_box), _area(item["box"])) * 0.65
                for item in regions
            )
            if not overlaps_detector:
                regions.append(candidate)
        groups = _groups(regions)
        groups.sort(
            key=lambda group: (_union_box(group)[1] // 180, -_union_box(group)[0])
        )
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
                    is_sfx=likely_sfx_text(clean),
                    erase_boxes=[item["box"] for item in group],
                )
            )
        page = PageOCR(page_number, image_path.name, image.width, image.height, units)
        return self.refine_cover_title(image_path, page)

    def recover_missing(
        self, image_path: Path, page: PageOCR, missing: list[dict]
    ) -> PageOCR:
        """Recover VLM-reported omissions through enlarged detector-owned crops.

        Full-page vision is useful for recall but its coordinates are too coarse
        for destructive editing.  Each proposed region is therefore enlarged,
        run through the exact Paddle detector again, and accepted only when
        MangaOCR independently recognizes Japanese in the recovered geometry.
        """
        if not missing:
            return page
        image = Image.open(image_path).convert("RGB")
        recovered_regions: list[dict] = []
        existing_boxes = [
            erase_box
            for unit in page.units
            for erase_box in (unit.erase_boxes or [unit.bbox])
        ]
        for item in missing[:40]:
            try:
                normalized = [int(value) for value in item["bbox"]]
            except (KeyError, TypeError, ValueError):
                continue
            if len(normalized) != 4:
                continue
            coarse = [
                round(normalized[0] * image.width / 1000),
                round(normalized[1] * image.height / 1000),
                round(normalized[2] * image.width / 1000),
                round(normalized[3] * image.height / 1000),
            ]
            if coarse[2] <= coarse[0] or coarse[3] <= coarse[1]:
                continue
            pad_x = max(12, round((coarse[2] - coarse[0]) * 0.12))
            pad_y = max(12, round((coarse[3] - coarse[1]) * 0.12))
            crop_box = [
                max(0, coarse[0] - pad_x),
                max(0, coarse[1] - pad_y),
                min(image.width, coarse[2] + pad_x),
                min(image.height, coarse[3] + pad_y),
            ]
            crop = image.crop(tuple(crop_box))
            scale = max(2.0, min(4.0, 1200 / max(crop.width, crop.height)))
            enlarged = crop.resize(
                (round(crop.width * scale), round(crop.height * scale)),
                Image.Resampling.LANCZOS,
            )
            results = list(self.detector.predict(np.asarray(enlarged)))
            payload = self._payload(results[0]) if results else {}
            for region in self._regions(payload):
                local = region["box"]
                mapped = [
                    crop_box[0] + round(local[0] / scale),
                    crop_box[1] + round(local[1] / scale),
                    crop_box[0] + round(local[2] / scale),
                    crop_box[1] + round(local[3] / scale),
                ]
                if _area(mapped) < 180:
                    continue
                covered = any(
                    _axis_overlap(mapped[0], mapped[2], box[0], box[2])
                    * _axis_overlap(mapped[1], mapped[3], box[1], box[3])
                    >= min(_area(mapped), _area(box)) * 0.65
                    for box in existing_boxes
                )
                if covered:
                    continue
                recovered_regions.append(
                    {"box": mapped, "text": "", "score": region["score"]}
                )

        recovered_regions = merge_region_candidates([], recovered_regions)
        for group in _groups(recovered_regions):
            box = _union_box(group)
            pad_x = max(10, round((box[2] - box[0]) * 0.06))
            pad_y = max(10, round((box[3] - box[1]) * 0.05))
            crop_box = [
                max(0, box[0] - pad_x),
                max(0, box[1] - pad_y),
                min(image.width, box[2] + pad_x),
                min(image.height, box[3] + pad_y),
            ]
            refined = self.reader(image.crop(tuple(crop_box))).strip()
            if not JP_RE.search(refined):
                continue
            page.units.append(
                TextUnit(
                    id="",
                    bbox=box,
                    crop_bbox=crop_box,
                    ja=refined,
                    score=round(max(item["score"] for item in group), 4),
                    is_sfx=likely_sfx_text(refined),
                    erase_boxes=[item["box"] for item in group],
                )
            )
        page.units.sort(key=lambda unit: (unit.bbox[1] // 180, -unit.bbox[0]))
        for index, unit in enumerate(page.units, start=1):
            unit.id = f"p{page.page:03d}u{index:02d}"
        return page


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

    def _chat(self, images: list[bytes], prompt: str, schema: dict) -> dict:
        body = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "think": False,
                "format": schema,
                "options": {"temperature": 0, "num_predict": 4096},
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [
                            base64.b64encode(content).decode("ascii")
                            for content in images
                        ],
                    }
                ],
            },
            ensure_ascii=False,
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
        if payload.get("done_reason") == "length":
            raise RuntimeError(
                "Ollama OCR output was truncated before valid JSON completed"
            )
        content = str(payload.get("message", {}).get("content", ""))
        return self._json_payload(content)

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
        prompt = (
            "Detect every Japanese text region in this manga page, including dialogue, "
            "narration, furigana and sound effects. Return JSON only as "
            '{"regions":[{"bbox":[x0,y0,x1,y1],"text":"...",'
            '"score":0.0,"is_sfx":false}]}. Coordinates must be integer source-image '
            f"pixels within width={width}, height={height}. Do not translate or omit text."
        )
        schema = {
            "type": "object",
            "properties": {
                "regions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "bbox": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            "text": {"type": "string"},
                            "score": {"type": "number"},
                            "is_sfx": {"type": "boolean"},
                        },
                        "required": ["bbox", "text", "score", "is_sfx"],
                    },
                }
            },
            "required": ["regions"],
        }
        regions = self._chat([image_path.read_bytes()], prompt, schema).get(
            "regions", []
        )
        units: list[TextUnit] = []
        for region in regions[:300]:
            try:
                raw = [float(value) for value in region["bbox"]]
                if len(raw) != 4:
                    continue
                if all(0 <= value <= 1 for value in raw):
                    raw = [
                        raw[0] * width,
                        raw[1] * height,
                        raw[2] * width,
                        raw[3] * height,
                    ]
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

    @staticmethod
    def _annotated_image(image: Image.Image, page: PageOCR) -> bytes:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        line_width = max(4, round(min(image.size) * 0.003))
        label_height = max(28, round(min(image.size) * 0.024))
        for index, unit in enumerate(page.units, start=1):
            x0, y0, x1, y1 = unit.bbox
            draw.rectangle((x0, y0, x1, y1), outline="#ff2d2d", width=line_width)
            label = str(index)
            label_box = draw.textbbox((0, 0), label)
            label_width = max(label_height, label_box[2] - label_box[0] + 12)
            label_y = max(0, y0 - label_height)
            draw.rectangle(
                (x0, label_y, min(image.width, x0 + label_width), y0),
                fill="#ff2d2d",
            )
            draw.text((x0 + 6, label_y + 3), label, fill="white")
        output = BytesIO()
        annotated.save(output, "JPEG", quality=92, subsampling=0)
        return output.getvalue()

    def refine(self, image_path: Path, page: PageOCR) -> tuple[PageOCR, list[dict]]:
        """Correct detector-bound OCR using full-page vision without changing geometry.

        The original page and a numbered overlay are sent together. The VLM may
        correct text and classify SFX, but exact bounding/erase boxes remain the
        detector's exclusive responsibility. Unboxed findings are returned for
        audit and can never enter destructive rendering through this method.
        """
        if not page.units:
            return page, []
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        ids = [unit.id for unit in page.units]
        candidates = "\n".join(
            f"{index}. {unit.id}: {unit.ja}"
            for index, unit in enumerate(page.units, start=1)
        )
        prompt = f"""你会收到同一张日语漫画的两幅图：第一幅是无标记原图，第二幅有红色编号框。
逐框结合整页上下文校正 OCR。红框和数字是程序标记，不属于漫画文字。
必须为每个 id 返回一项；不要翻译。text 只写框内日文，保留标点和语气符号。
如果页面上还有未被任何红框覆盖的可见日文，写入 missing，bbox 使用 0 到 1000 的归一化整数；没有则返回空数组。

当前机器 OCR 候选：
{candidates}"""
        item_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "enum": ids},
                "text": {"type": "string"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "is_sfx": {"type": "boolean"},
            },
            "required": ["id", "text", "score", "is_sfx"],
        }
        missing_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "is_sfx": {"type": "boolean"},
            },
            "required": ["text", "bbox", "score", "is_sfx"],
        }
        schema = {
            "type": "object",
            "properties": {
                "regions": {"type": "array", "items": item_schema},
                "missing": {"type": "array", "items": missing_schema},
            },
            "required": ["regions", "missing"],
        }
        payload = self._chat(
            [image_path.read_bytes(), self._annotated_image(image, page)],
            prompt,
            schema,
        )
        by_id = {
            str(item.get("id")): item
            for item in payload.get("regions", [])
            if isinstance(item, dict) and item.get("id") in ids
        }
        for unit in page.units:
            item = by_id.get(unit.id)
            vision_is_sfx = unit.is_sfx
            if item:
                candidate = str(item.get("text", "")).strip()
                score = float(item.get("score", 0.0))
                if prefer_semantic_ocr(unit.ja, candidate, score):
                    unit.ja = candidate
                    unit.score = max(unit.score, round(score, 4))
                vision_is_sfx = bool(item.get("is_sfx", False))
            unit.is_sfx = semantic_sfx_classification(
                unit.ja,
                unit.score,
                vision_is_sfx,
            )
        missing = [
            item
            for item in payload.get("missing", [])
            if isinstance(item, dict)
            and JP_RE.search(str(item.get("text", "")))
            and float(item.get("score", 0.0)) >= 0.75
        ]
        return page, missing

    @staticmethod
    def _local_crop_is_suspicious(unit: TextUnit) -> bool:
        """Flag OCR text whose length is implausible for its exact geometry."""
        text = re.sub(r"\s+", "", unit.ja)
        if not text:
            return True
        area_per_character = _area(unit.bbox) / max(1, len(text))
        return (
            unit.score <= 0.05
            or area_per_character < 350
            or area_per_character > 30_000
        )

    @staticmethod
    def _tight_crop_bytes(image: Image.Image, unit: TextUnit) -> bytes:
        x0, y0, x1, y1 = unit.bbox
        width, height = max(1, x1 - x0), max(1, y1 - y0)
        pad_x = max(12, round(width * 0.08))
        pad_y = max(12, round(height * 0.08))
        crop = image.crop(
            (
                max(0, x0 - pad_x),
                max(0, y0 - pad_y),
                min(image.width, x1 + pad_x),
                min(image.height, y1 + pad_y),
            )
        )
        scale = min(4.0, max(1.0, 800 / max(crop.width, crop.height)))
        if scale > 1:
            crop = crop.resize(
                (round(crop.width * scale), round(crop.height * scale)),
                Image.Resampling.LANCZOS,
            )
        output = BytesIO()
        crop.save(output, "PNG")
        return output.getvalue()

    @staticmethod
    def _normalize_local_crop_text(value: str) -> str:
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        kept = []
        for index, line in enumerate(lines):
            if (
                len(line) <= 2
                and index + 1 < len(lines)
                and lines[index + 1].startswith(line)
            ):
                continue
            kept.append(line)
        return "\n".join(kept)

    def refine_local_crops(
        self, image_path: Path, page: PageOCR, batch_size: int = 8
    ) -> PageOCR:
        """Audit OCR text and semantic role from tight, ordered source crops.

        Full-page VLM passes are useful for story context, but small numbered
        regions can be assigned to the wrong id.  This second pass removes that
        ambiguity: each image contains exactly one detector-owned region.  The
        specialized OCR remains authoritative unless geometry is suspicious or
        the crop auditor strongly disagrees about dialogue versus sound effect.
        """
        if not page.units:
            return page
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        for start in range(0, len(page.units), max(1, batch_size)):
            units = page.units[start : start + max(1, batch_size)]
            ids = [unit.id for unit in units]
            item_schema = {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "enum": ids},
                    "text": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["dialogue", "narration", "sfx", "nontext"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["id", "text", "role", "confidence"],
            }
            schema = {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": item_schema},
                },
                "required": ["items"],
            }
            order = "、".join(ids)
            prompt = f"""你会依次收到 {len(units)} 张漫画文字框的紧密局部放大图，对应 id 顺序为：{order}。
每张图只抄写局部内真正可见的日文，禁止根据故事或相邻图片补全。
role 选择 dialogue（对白）、narration（旁白）、sfx（拟声/效果字）或 nontext（无可替换文字）。看不清就选 nontext。
必须为每个 id 返回一项。"""
            payload = self._chat(
                [self._tight_crop_bytes(image, unit) for unit in units],
                prompt,
                schema,
            )
            by_id = {
                str(item.get("id")): item
                for item in payload.get("items", [])
                if isinstance(item, dict) and item.get("id") in ids
            }
            for unit in units:
                item = by_id.get(unit.id)
                if not item:
                    continue
                role = str(item.get("role", "")).lower()
                confidence = float(item.get("confidence", 0.0))
                candidate = self._normalize_local_crop_text(str(item.get("text", "")))
                suspicious = self._local_crop_is_suspicious(unit)
                if role in {"sfx", "nontext"}:
                    threshold = 0.68 if suspicious or unit.is_sfx else 0.88
                    if confidence >= threshold:
                        unit.is_sfx = True
                    continue
                if role not in {"dialogue", "narration"} or confidence < 0.82:
                    continue
                if not JP_RE.search(candidate):
                    continue
                if unit.is_sfx:
                    semantic = re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", candidate)
                    has_sentence_evidence = (
                        len(semantic) >= 4
                        or bool(re.search(r"[．。…、，,?？]", candidate))
                        or (
                            bool(re.search(r"[\u3400-\u9fff]", semantic))
                            and bool(re.search(r"[ぁ-ゖ]", semantic))
                        )
                    )
                    if not has_sentence_evidence:
                        continue
                # Tight-crop text may replace the detector recognizer only when
                # the old geometry/text contract is already suspect, or when it
                # recovers a region previously classified as an effect.
                if suspicious or unit.is_sfx:
                    unit.ja = candidate
                    unit.score = max(unit.score, round(confidence, 4))
                    unit.is_sfx = False
        for unit in page.units:
            if (
                tiny_low_confidence_nontext(page, unit)
                or malformed_tiny_ocr(page, unit)
                or duplicate_tiny_fragment(page, unit)
            ):
                unit.is_sfx = True
                unit.special = unit.special or "ocr_duplicate"
        return page

    @staticmethod
    def filter_covered_missing(page: PageOCR, missing: list[dict]) -> list[dict]:
        """Discard second-pass VLM omissions already disproved by exact OCR."""

        def normalized_text(value: str) -> str:
            return re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", value)

        known_text = {
            normalized_text(unit.ja)
            for unit in page.units
            if len(normalized_text(unit.ja)) >= 2
        }
        known_boxes = [
            erase_box
            for unit in page.units
            for erase_box in (unit.erase_boxes or [unit.bbox])
        ]
        unresolved = []
        for item in missing:
            text = normalized_text(str(item.get("text", "")))
            if text and text in known_text:
                continue
            try:
                raw = [int(value) for value in item["bbox"]]
            except (KeyError, TypeError, ValueError):
                unresolved.append(item)
                continue
            if len(raw) != 4:
                unresolved.append(item)
                continue
            box = [
                round(raw[0] * page.width / 1000),
                round(raw[1] * page.height / 1000),
                round(raw[2] * page.width / 1000),
                round(raw[3] * page.height / 1000),
            ]
            covered = any(
                _axis_overlap(box[0], box[2], known[0], known[2])
                * _axis_overlap(box[1], box[3], known[1], known[3])
                >= min(_area(box), _area(known)) * 0.5
                for known in known_boxes
            )
            if not covered:
                unresolved.append(item)
        return unresolved


class HybridMangaOCR:
    """Quality OCR: Paddle/MangaOCR geometry plus full-page Ollama semantics."""

    def __init__(
        self,
        paths: AppPaths,
        base_url: str,
        model: str = DEFAULT_OLLAMA_VISION_MODEL,
        device: str = "auto",
        profile: str = "quality",
    ):
        self.geometry = PaddleMangaOCR(paths, device, profile)
        self.semantics = OllamaVisionOCR(base_url, model)
        self.profile = profile
        self.last_missing: list[dict] = []

    @staticmethod
    def _coarse_page_as_missing(page: PageOCR) -> list[dict]:
        return [
            {
                "text": unit.ja,
                "bbox": [
                    round(unit.bbox[0] * 1000 / page.width),
                    round(unit.bbox[1] * 1000 / page.height),
                    round(unit.bbox[2] * 1000 / page.width),
                    round(unit.bbox[3] * 1000 / page.height),
                ],
                "score": unit.score,
                "is_sfx": unit.is_sfx,
            }
            for unit in page.units
        ]

    def analyze(self, image_path: Path, page_number: int) -> PageOCR:
        page = self.geometry.analyze(image_path, page_number)
        if page.units:
            page, self.last_missing = self.semantics.refine(image_path, page)
        else:
            # An empty exact-detector result is not evidence that a page has no
            # text. Use full-page vision only to propose recovery crops; raw VLM
            # coordinates still never become erase geometry.
            coarse = self.semantics.analyze(image_path, page_number)
            self.last_missing = self._coarse_page_as_missing(coarse)
        if self.last_missing:
            page = self.geometry.recover_missing(image_path, page, self.last_missing)
            if page.units:
                page, self.last_missing = self.semantics.refine(image_path, page)
                self.last_missing = self.semantics.filter_covered_missing(
                    page, self.last_missing
                )
        if getattr(self, "profile", "quality") == "quality" and hasattr(
            self.semantics, "refine_local_crops"
        ):
            page = self.semantics.refine_local_crops(image_path, page)
        page.semantic_missing = list(self.last_missing)
        return page

    def refine_cover_title(self, image_path: Path, page: PageOCR) -> PageOCR:
        page = self.geometry.refine_cover_title(image_path, page)
        page, self.last_missing = self.semantics.refine(image_path, page)
        page.semantic_missing = list(self.last_missing)
        return page
