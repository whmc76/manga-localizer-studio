from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from .config import AppPaths, UserSettings
from .name_dictionary import JapaneseNameDictionary
from .ocr import (
    DEFAULT_OLLAMA_VISION_MODEL,
    EXPLICIT_SKIP_REASONS,
    UNRESOLVED_SKIP_REASON,
    HybridMangaOCR,
    PageOCR,
    OllamaVisionOCR,
    PaddleMangaOCR,
    TextUnit,
    likely_sfx_text,
    list_images,
    semantic_sfx_classification,
)
from .renderer import ArtworkPreservingRenderer
from .model_manager import ModelManager
from .translator import (
    DEFAULT_LOCAL_TRANSLATION_MODEL,
    HyMTTranslator,
    LocalQualityTranslator,
    OllamaTranslator,
    OpenAICompatibleTranslator,
    PromptTranslator,
    ensure_ollama_model,
)


ProgressCallback = Callable[[str, int, int, str], None]
OCR_CACHE_VERSION = 4


def unsafe_semantic_missing(page: PageOCR, preserve_sfx: bool) -> list[dict]:
    """Return full-page findings that cannot safely remain outside exact geometry."""
    unsafe = []
    for item in page.semantic_missing:
        text = str(item.get("text", "")).strip()
        japanese_chars = re.findall(r"[ぁ-ゖァ-ヺ\u3400-\u9fff]", text)
        single_kana_fragment = len(japanese_chars) == 1 and bool(
            re.fullmatch(r"[ぁ-ゖァ-ヺ]", japanese_chars[0])
        )
        short_vocalization = len(japanese_chars) <= 3 and bool(
            re.search(r"[っッゃゅょャュョ♡♥]", text)
        )
        preservable = (
            single_kana_fragment
            or short_vocalization
            or semantic_sfx_classification(
                text,
                float(item.get("score", 0.0)),
                bool(item.get("is_sfx", False)),
            )
        )
        if not preserve_sfx or not preservable:
            unsafe.append(item)
    return unsafe


@dataclass
class PipelineRequest:
    source: Path
    output: Path
    target_language: str = "简体中文"
    story_context: bool = True
    context_pages: int = 6
    preserve_sfx: bool = True
    quality_profile: str = "quality"
    output_format: str = "webp"
    device: str = "auto"
    glossary: dict[str, str] | None = None
    inference_backend: str = "ollama"
    ocr_backend: str = "hybrid"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = DEFAULT_LOCAL_TRANSLATION_MODEL
    ollama_ocr_model: str = DEFAULT_OLLAMA_VISION_MODEL
    online_base_url: str = "https://api.openai.com/v1"
    online_model: str = ""
    online_api_key: str = ""
    reviewed_transcript: Path | None = None


class LocalizerPipeline:
    def __init__(self, paths: AppPaths):
        self.paths = paths.ensure()

    @staticmethod
    def _emit(
        callback: ProgressCallback, phase: str, current: int, total: int, message: str
    ):
        callback(phase, current, total, message)

    @staticmethod
    def _write_transcript(
        path: Path,
        source: Path,
        pages: list[PageOCR],
        glossary: dict[str, str] | None = None,
    ) -> None:
        payload = {"source": str(source), "pages": [page.to_dict() for page in pages]}
        if glossary:
            payload["glossary"] = glossary
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _load_cached_ocr(
        work: Path,
        image_path: Path,
        index: int,
        ocr_backend: str = "builtin",
        quality_profile: str = "quality",
        ocr_model: str = "",
    ) -> PageOCR | None:
        cache_path = work / f"{image_path.stem}.ocr.json"
        if not cache_path.is_file():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            page = page_from_dict(payload)
            with Image.open(image_path) as image:
                expected_size = image.size
            source_stat = image_path.stat()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if page.page != index or page.file != image_path.name:
            return None
        if (page.width, page.height) != expected_size:
            return None
        metadata = payload.get("_cache")
        if metadata:
            if int(metadata.get("version", -1)) != OCR_CACHE_VERSION:
                return None
            if metadata.get("ocr_backend") != ocr_backend:
                return None
            if metadata.get("quality_profile") != quality_profile:
                return None
            if (
                ocr_backend in {"hybrid", "ollama"}
                and metadata.get("ocr_model") != ocr_model
            ):
                return None
            if int(metadata.get("source_size", -1)) != source_stat.st_size:
                return None
            if int(metadata.get("source_mtime_ns", -1)) != source_stat.st_mtime_ns:
                return None
        else:
            # Legacy caches predate detector-preserving quality OCR and can
            # silently omit valid Japanese bubbles.  They are never safe to
            # reuse after the cache contract changed.
            return None
        # Text-only classification improvements are safe to apply to an OCR
        # cache without repeating model inference.  This keeps expensive OCR
        # reusable while preventing legacy mixed-kana sound effects from being
        # sent to the translator as dialogue.
        for unit in page.units:
            unit.is_sfx = semantic_sfx_classification(
                unit.ja, unit.score, unit.is_sfx or likely_sfx_text(unit.ja)
            )
        return page

    def run(
        self, request: PipelineRequest, callback: ProgressCallback | None = None
    ) -> dict:
        emit = callback or (lambda _phase, _current, _total, _message: None)
        source = request.source.expanduser().resolve()
        output = request.output.expanduser().resolve()
        if source == output:
            raise ValueError("Output directory must not be the source directory.")
        files = list_images(source)
        if not files:
            raise ValueError("No supported images were found in the source folder.")
        output.mkdir(parents=True, exist_ok=True)
        work = output / ".manga-localizer-work"
        work.mkdir(parents=True, exist_ok=True)

        pages: list[PageOCR] = []
        total = len(files)
        if request.reviewed_transcript:
            transcript_path = request.reviewed_transcript.expanduser().resolve()
            payload = json.loads(transcript_path.read_text(encoding="utf-8"))
            pages = [page_from_dict(item) for item in payload["pages"]]
            self._validate_reviewed_pages(files, pages)
            self._emit(
                emit, "ocr", total, total, f"已导入审校稿 {transcript_path.name}"
            )
        else:
            cached_pages = [
                self._load_cached_ocr(
                    work,
                    image_path,
                    index,
                    request.ocr_backend,
                    request.quality_profile,
                    (
                        request.ollama_ocr_model
                        if request.ocr_backend in {"hybrid", "ollama"}
                        else ""
                    ),
                )
                for index, image_path in enumerate(files, start=1)
            ]
            title_refinements = [
                bool(
                    request.ocr_backend in {"builtin", "hybrid"}
                    and page is not None
                    and PaddleMangaOCR.is_cover_title_candidate(page)
                )
                for page in cached_pages
            ]
            ocr = None
            if any(page is None for page in cached_pages) or any(title_refinements):
                if request.ocr_backend == "builtin":
                    ocr = PaddleMangaOCR(
                        self.paths, request.device, request.quality_profile
                    )
                elif request.ocr_backend == "hybrid":
                    self._emit(emit, "ocr", 0, total, "检查本地视觉语义模型")
                    ensure_ollama_model(
                        request.ollama_base_url, request.ollama_ocr_model
                    )
                    ocr = HybridMangaOCR(
                        self.paths,
                        request.ollama_base_url,
                        request.ollama_ocr_model,
                        request.device,
                        request.quality_profile,
                    )
                elif request.ocr_backend == "ollama":
                    ocr = OllamaVisionOCR(
                        request.ollama_base_url, request.ollama_ocr_model
                    )
                else:
                    raise ValueError(f"Unknown OCR backend: {request.ocr_backend}")
            for index, (image_path, cached_page) in enumerate(
                zip(files, cached_pages), start=1
            ):
                cache_updated = False
                if cached_page is not None:
                    page = cached_page
                    if title_refinements[index - 1]:
                        page = ocr.refine_cover_title(image_path, page)
                        cache_updated = page is not cached_page
                    self._emit(
                        emit, "ocr", index - 1, total, f"复用 OCR {image_path.name}"
                    )
                else:
                    self._emit(emit, "ocr", index - 1, total, f"识别 {image_path.name}")
                    page = ocr.analyze(image_path, index)
                unsafe_missing = unsafe_semantic_missing(page, request.preserve_sfx)
                if unsafe_missing:
                    texts = " / ".join(
                        str(item.get("text", "")).strip() or "未识别文字"
                        for item in unsafe_missing[:4]
                    )
                    raise RuntimeError(
                        f"第 {index} 页发现未被精确检测框覆盖的文字：{texts}。"
                        "质量模式拒绝猜测擦除区域，请更换 OCR 模型或重试。"
                    )
                pages.append(page)
                if cached_page is None or cache_updated:
                    page_payload = page.to_dict()
                    source_stat = image_path.stat()
                    page_payload["_cache"] = {
                        "version": OCR_CACHE_VERSION,
                        "ocr_backend": request.ocr_backend,
                        "quality_profile": request.quality_profile,
                        "ocr_model": (
                            request.ollama_ocr_model
                            if request.ocr_backend in {"hybrid", "ollama"}
                            else ""
                        ),
                        "source_size": source_stat.st_size,
                        "source_mtime_ns": source_stat.st_mtime_ns,
                    }
                    (work / f"{image_path.stem}.ocr.json").write_text(
                        json.dumps(page_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                self._emit(emit, "ocr", index, total, f"已识别 {image_path.name}")

        needs_translation = any(
            not unit.skip
            and not (request.preserve_sfx and unit.is_sfx)
            and not PromptTranslator._valid_translation(unit, unit.zh)
            for page in pages
            for unit in page.units
        )
        self._emit(
            emit,
            "translate",
            0,
            total,
            "加载翻译后端" if needs_translation else "审校稿已包含译文",
        )
        if not needs_translation:
            translator = None
        elif request.inference_backend == "builtin":
            translator = HyMTTranslator(
                self.paths.models / "hy-mt2",
                request.target_language,
                request.device,
                JapaneseNameDictionary(self.paths.cache),
            )
        elif request.inference_backend == "ollama":
            if not request.ollama_model.strip():
                raise ValueError("Ollama model name is required")
            self._emit(emit, "translate", 0, total, "检查本地解除限制翻译模型")
            downloaded = ensure_ollama_model(
                request.ollama_base_url, request.ollama_model
            )
            if downloaded:
                self._emit(emit, "translate", 0, total, "本地翻译模型下载完成")
            editor = OllamaTranslator(
                request.ollama_base_url,
                request.ollama_model,
                request.target_language,
                JapaneseNameDictionary(self.paths.cache),
            )
            if request.quality_profile == "quality":
                model_manager = ModelManager(self.paths)
                if not model_manager.is_ready("hy-mt2"):
                    self._emit(emit, "translate", 0, total, "下载 Hy-MT2 独立翻译候选")
                    model_manager.download("hy-mt2")
                candidate = HyMTTranslator(
                    self.paths.models / "hy-mt2",
                    request.target_language,
                    request.device,
                    JapaneseNameDictionary(self.paths.cache),
                )
                self._emit(emit, "translate", 0, total, "启用 9B 分阶段语义审校")
                translator = LocalQualityTranslator(editor, candidate)
            else:
                translator = editor
        elif request.inference_backend == "online":
            if not request.online_model.strip():
                raise ValueError("Online model name is required")
            translator = OpenAICompatibleTranslator(
                request.online_base_url,
                request.online_model,
                request.online_api_key,
                request.target_language,
                JapaneseNameDictionary(self.paths.cache),
            )
        else:
            raise ValueError(f"Unknown inference backend: {request.inference_backend}")
        if translator is not None:
            try:
                translator.translate_pages(
                    pages,
                    context_pages=max(0, min(12, request.context_pages)),
                    story_context=request.story_context,
                    preserve_sfx=request.preserve_sfx,
                    glossary=request.glossary,
                    progress=lambda current, count: self._emit(
                        emit,
                        "translate",
                        current,
                        count,
                        f"翻译质量进度 {current}/{count}",
                    ),
                )
            finally:
                self._write_transcript(
                    work / "translation-draft.json",
                    source,
                    pages,
                    translator.resolved_glossary,
                )
        else:
            self._write_transcript(
                work / "translation-draft.json", source, pages, request.glossary
            )
        quality = completion_summary(pages, request.preserve_sfx)
        if quality["unresolved_units"]:
            sample = ", ".join(quality["unresolved_ids"][:8])
            raise ValueError(
                f"Localization has {quality['unresolved_units']} unresolved text units"
                f" ({sample}). Translate them or assign an explicit skip_reason."
            )
        self._emit(emit, "translate", total, total, "连贯翻译完成")
        self._write_transcript(
            work / "transcript.json",
            source,
            pages,
            translator.resolved_glossary
            if translator is not None
            else request.glossary,
        )

        if request.quality_profile == "quality":
            manager = ModelManager(self.paths)
            if not manager.lama_weights_path().exists():
                self._emit(emit, "render", 0, total, "首次使用：下载 LaMa 补画模型")
                manager.download(
                    "lama",
                    lambda value, message: self._emit(
                        emit, "render", 0, total, f"{message} {value}%"
                    ),
                )
        manifest = []
        if request.output_format not in {"webp", "png"}:
            raise ValueError(f"Unknown lossless output format: {request.output_format}")
        renderer = ArtworkPreservingRenderer(
            cleanup_profile=request.quality_profile,
            paths=self.paths,
            device=request.device,
        )
        suffix = ".webp" if request.output_format == "webp" else ".png"
        for index, page in enumerate(pages, start=1):
            self._emit(emit, "render", index - 1, total, f"替换第 {index} 页文字")
            source_path = source / page.file
            output_path = output / f"{source_path.stem}{suffix}"
            manifest.append(
                renderer.render_page(
                    source_path, page, output_path, request.preserve_sfx
                )
            )
            self._emit(emit, "render", index, total, f"已生成 {output_path.name}")

        result = {
            "source": str(source),
            "output": str(output),
            "pages": total,
            "images": manifest,
            "transcript": str(work / "transcript.json"),
            "quality": quality,
        }
        (output / "translation_manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._emit(emit, "complete", total, total, "本地化完成")
        return result

    @staticmethod
    def _validate_reviewed_pages(files: list[Path], pages: list[PageOCR]) -> None:
        if len(files) != len(pages):
            raise ValueError(
                f"Reviewed transcript has {len(pages)} pages, source folder has {len(files)}"
            )
        for index, (source, page) in enumerate(zip(files, pages), start=1):
            if source.name != page.file:
                raise ValueError(
                    f"Reviewed transcript page {index} expects {page.file}, got {source.name}"
                )
            with Image.open(source) as image:
                if image.size != (page.width, page.height):
                    raise ValueError(
                        f"Reviewed transcript page {index} dimensions do not match source"
                    )
        ambiguous = [
            unit.id
            for page in pages
            for unit in page.units
            if unit.skip and unit.skip_reason not in EXPLICIT_SKIP_REASONS
        ]
        if ambiguous:
            sample = ", ".join(ambiguous[:8])
            raise ValueError(
                f"Reviewed transcript has {len(ambiguous)} ambiguous skipped units"
                f" ({sample}). Legacy skip=true is unresolved; use one of: "
                f"{', '.join(sorted(EXPLICIT_SKIP_REASONS))}."
            )


def request_from_settings(
    source: Path, output: Path, settings: UserSettings
) -> PipelineRequest:
    return PipelineRequest(
        source=source,
        output=output,
        target_language=settings.target_language,
        story_context=settings.story_context,
        context_pages=settings.context_pages,
        preserve_sfx=settings.preserve_sfx,
        quality_profile=settings.quality_profile,
        output_format=settings.output_format,
        device=settings.device,
        inference_backend=settings.inference_backend,
        ocr_backend=settings.ocr_backend,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        ollama_ocr_model=settings.ollama_ocr_model,
        online_base_url=settings.online_base_url,
        online_model=settings.online_model,
    )


def page_from_dict(payload: dict) -> PageOCR:
    return PageOCR(
        page=payload["page"],
        file=payload["file"],
        width=payload["width"],
        height=payload["height"],
        units=[
            TextUnit(
                id=unit["id"],
                bbox=unit["bbox"],
                crop_bbox=unit.get("crop_bbox", unit["bbox"]),
                ja=unit.get("ja", ""),
                score=float(unit.get("score", max(unit.get("paddle_scores") or [0.0]))),
                is_sfx=bool(unit.get("is_sfx", False)),
                zh=unit.get("zh", ""),
                skip=bool(unit.get("skip", False)),
                skip_reason=(
                    str(unit.get("skip_reason", "")).strip()
                    or (UNRESOLVED_SKIP_REASON if bool(unit.get("skip", False)) else "")
                ),
                erase_boxes=unit.get("erase_boxes", []),
                translation_attempts=list(unit.get("translation_attempts", [])),
                special=str(unit.get("special", "")).strip(),
            )
            for unit in payload["units"]
        ],
        semantic_missing=list(payload.get("semantic_missing", [])),
    )


def completion_summary(pages: list[PageOCR], preserve_sfx: bool = False) -> dict:
    """Classify every OCR unit; unresolved work must remain visible to callers."""
    translated = explicit_skips = preserved_sfx_count = 0
    unresolved_ids: list[str] = []
    invalid_translation_ids: list[str] = []
    for page in pages:
        for unit in page.units:
            if unit.skip:
                if unit.skip_reason in EXPLICIT_SKIP_REASONS:
                    explicit_skips += 1
                else:
                    unresolved_ids.append(unit.id)
            elif preserve_sfx and unit.is_sfx:
                preserved_sfx_count += 1
            elif unit.zh.strip():
                if PromptTranslator._valid_translation(unit, unit.zh):
                    translated += 1
                else:
                    invalid_translation_ids.append(unit.id)
                    unresolved_ids.append(unit.id)
            else:
                unresolved_ids.append(unit.id)
    return {
        "total_units": translated
        + explicit_skips
        + preserved_sfx_count
        + len(unresolved_ids),
        "translated_units": translated,
        "explicitly_skipped_units": explicit_skips,
        "preserved_sfx_units": preserved_sfx_count,
        "unresolved_units": len(unresolved_ids),
        "unresolved_ids": unresolved_ids,
        "invalid_translation_units": len(invalid_translation_ids),
        "invalid_translation_ids": invalid_translation_ids,
    }
