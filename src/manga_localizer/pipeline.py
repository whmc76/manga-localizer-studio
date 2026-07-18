from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from .config import AppPaths, UserSettings
from .ocr import (
    EXPLICIT_SKIP_REASONS,
    UNRESOLVED_SKIP_REASON,
    PageOCR,
    OllamaVisionOCR,
    PaddleMangaOCR,
    TextUnit,
    list_images,
)
from .renderer import ArtworkPreservingRenderer
from .model_manager import ModelManager
from .translator import (
    HyMTTranslator,
    OllamaTranslator,
    OpenAICompatibleTranslator,
    PromptTranslator,
)


ProgressCallback = Callable[[str, int, int, str], None]


@dataclass
class PipelineRequest:
    source: Path
    output: Path
    target_language: str = "简体中文"
    story_context: bool = True
    context_pages: int = 3
    preserve_sfx: bool = False
    quality_profile: str = "quality"
    output_format: str = "webp"
    device: str = "auto"
    glossary: dict[str, str] | None = None
    inference_backend: str = "builtin"
    ocr_backend: str = "builtin"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_ocr_model: str = "qwen2.5vl:7b"
    online_base_url: str = "https://api.openai.com/v1"
    online_model: str = ""
    online_api_key: str = ""
    reviewed_transcript: Path | None = None


class LocalizerPipeline:
    def __init__(self, paths: AppPaths):
        self.paths = paths.ensure()

    @staticmethod
    def _emit(callback: ProgressCallback, phase: str, current: int, total: int, message: str):
        callback(phase, current, total, message)

    def run(self, request: PipelineRequest, callback: ProgressCallback | None = None) -> dict:
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
            self._emit(emit, "ocr", total, total, f"已导入审校稿 {transcript_path.name}")
        else:
            if request.ocr_backend == "builtin":
                ocr = PaddleMangaOCR(self.paths, request.device, request.quality_profile)
            elif request.ocr_backend == "ollama":
                ocr = OllamaVisionOCR(
                    request.ollama_base_url, request.ollama_ocr_model
                )
            else:
                raise ValueError(f"Unknown OCR backend: {request.ocr_backend}")
            for index, image_path in enumerate(files, start=1):
                self._emit(emit, "ocr", index - 1, total, f"识别 {image_path.name}")
                page = ocr.analyze(image_path, index)
                pages.append(page)
                (work / f"{image_path.stem}.ocr.json").write_text(
                    json.dumps(page.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self._emit(emit, "ocr", index, total, f"已识别 {image_path.name}")

        needs_translation = any(
            not unit.skip
            and not (request.preserve_sfx and unit.is_sfx)
            and not PromptTranslator._valid_translation(unit, unit.zh)
            for page in pages
            for unit in page.units
        )
        self._emit(emit, "translate", 0, total, "加载翻译后端" if needs_translation else "审校稿已包含译文")
        if not needs_translation:
            translator = None
        elif request.inference_backend == "builtin":
            translator = HyMTTranslator(
                self.paths.models / "hy-mt2", request.target_language, request.device
            )
        elif request.inference_backend == "ollama":
            if not request.ollama_model.strip():
                raise ValueError("Ollama model name is required")
            translator = OllamaTranslator(
                request.ollama_base_url, request.ollama_model, request.target_language
            )
        elif request.inference_backend == "online":
            if not request.online_model.strip():
                raise ValueError("Online model name is required")
            translator = OpenAICompatibleTranslator(
                request.online_base_url,
                request.online_model,
                request.online_api_key,
                request.target_language,
            )
        else:
            raise ValueError(f"Unknown inference backend: {request.inference_backend}")
        if translator is not None:
            translator.translate_pages(
                pages,
                context_pages=max(0, min(12, request.context_pages)),
                story_context=request.story_context,
                preserve_sfx=request.preserve_sfx,
                glossary=request.glossary,
                progress=lambda current, count: self._emit(
                    emit, "translate", current, count, f"已翻译 {current}/{count} 页"
                ),
            )
        quality = completion_summary(pages, request.preserve_sfx)
        if quality["unresolved_units"]:
            sample = ", ".join(quality["unresolved_ids"][:8])
            raise ValueError(
                f"Localization has {quality['unresolved_units']} unresolved text units"
                f" ({sample}). Translate them or assign an explicit skip_reason."
            )
        self._emit(emit, "translate", total, total, "连贯翻译完成")
        transcript = {"source": str(source), "pages": [page.to_dict() for page in pages]}
        (work / "transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
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
        renderer = ArtworkPreservingRenderer(
            cleanup_profile=request.quality_profile,
            paths=self.paths,
            device=request.device,
        )
        manifest = []
        if request.output_format not in {"webp", "png"}:
            raise ValueError(f"Unknown lossless output format: {request.output_format}")
        suffix = ".webp" if request.output_format == "webp" else ".png"
        for index, page in enumerate(pages, start=1):
            self._emit(emit, "render", index - 1, total, f"替换第 {index} 页文字")
            source_path = source / page.file
            output_path = output / f"{source_path.stem}{suffix}"
            manifest.append(
                renderer.render_page(source_path, page, output_path, request.preserve_sfx)
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


def request_from_settings(source: Path, output: Path, settings: UserSettings) -> PipelineRequest:
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
                special=str(unit.get("special", "")).strip(),
            )
            for unit in payload["units"]
        ],
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
        "total_units": translated + explicit_skips + preserved_sfx_count + len(unresolved_ids),
        "translated_units": translated,
        "explicitly_skipped_units": explicit_skips,
        "preserved_sfx_units": preserved_sfx_count,
        "unresolved_units": len(unresolved_ids),
        "unresolved_ids": unresolved_ids,
        "invalid_translation_units": len(invalid_translation_ids),
        "invalid_translation_ids": invalid_translation_ids,
    }
