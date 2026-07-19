from __future__ import annotations

import importlib.util
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from .config import AppPaths, configure_model_caches


ProgressCallback = Callable[[int, str], None]

LAMA_WEIGHTS_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/"
    "v0.1.0/big-lama.pt"
)
LAMA_WEIGHTS_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"


class ModelDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    role: str
    provider: str
    repository: str
    size: str
    license: str
    required: bool = True


MODEL_REGISTRY = (
    ModelSpec(
        id="paddleocr",
        name="PaddleOCR",
        role="文字检测与行级识别",
        provider="ModelScope via PaddleX",
        repository="PP-OCRv5_mobile_det + PP-OCRv5_server_rec",
        size="~180 MB",
        license="Apache-2.0",
    ),
    ModelSpec(
        id="manga-ocr",
        name="Manga OCR",
        role="日文漫画识别",
        provider="Hugging Face fallback",
        repository="kha-white/manga-ocr-base",
        size="~450 MB",
        license="Apache-2.0",
    ),
    ModelSpec(
        id="lama",
        name="Big-LaMa",
        role="文字区域背景补全",
        provider="Pinned upstream fallback",
        repository=LAMA_WEIGHTS_URL,
        size="~196 MB",
        license="Apache-2.0",
    ),
    ModelSpec(
        id="hy-mt2",
        name="Hy-MT2 1.8B",
        role="连贯翻译",
        provider="ModelScope",
        repository="Tencent-Hunyuan/Hy-MT2-1.8B",
        size="~4.1 GB",
        license="Apache-2.0",
    ),
)


class ModelManager:
    def __init__(self, paths: AppPaths):
        self.paths = paths.ensure()
        configure_model_caches(paths)

    def model_dir(self, model_id: str) -> Path:
        return self.paths.models / model_id

    def _marker(self, model_id: str) -> Path:
        return self.model_dir(model_id) / ".ready.json"

    def is_ready(self, model_id: str) -> bool:
        marker = self._marker(model_id)
        if not marker.exists():
            return False
        if model_id == "lama" and not self.lama_weights_path().exists():
            return False
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            return payload.get("model_id") == model_id
        except (json.JSONDecodeError, OSError):
            return False

    def status(
        self,
        inference_backend: str = "builtin",
        quality_profile: str = "quality",
        ocr_backend: str = "builtin",
    ) -> list[dict]:
        return [
            {
                **asdict(spec),
                "required": (
                    (
                        spec.id != "hy-mt2"
                        or inference_backend == "builtin"
                        or (
                            inference_backend == "ollama"
                            and quality_profile == "quality"
                        )
                    )
                    and (spec.id != "lama" or quality_profile == "quality")
                    and (
                        spec.id not in {"paddleocr", "manga-ocr"}
                        or ocr_backend in {"builtin", "hybrid"}
                    )
                ),
                "ready": self.is_ready(spec.id),
                "path": str(self.model_dir(spec.id)),
            }
            for spec in MODEL_REGISTRY
        ]

    def lama_weights_path(self) -> Path:
        return self.model_dir("lama") / "big-lama.pt"

    def mark_ready(self, model_id: str, metadata: dict | None = None) -> None:
        directory = self.model_dir(model_id)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"model_id": model_id, **(metadata or {})}
        self._marker(model_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _require(self, module: str, extra: str = "ml") -> None:
        if importlib.util.find_spec(module) is None:
            raise ModelDependencyError(
                f"Missing optional dependency '{module}'. Run scripts/bootstrap with "
                f"the {extra} profile, or install 'manga-localizer-studio[{extra}]'."
            )

    def download(self, model_id: str, progress: ProgressCallback | None = None) -> Path:
        callback = progress or (lambda _value, _message: None)
        spec = next((item for item in MODEL_REGISTRY if item.id == model_id), None)
        if spec is None:
            raise KeyError(f"Unknown model: {model_id}")
        target = self.model_dir(model_id)
        target.mkdir(parents=True, exist_ok=True)
        callback(5, f"Preparing {spec.name}")

        if model_id == "lama":
            # ModelScope has damo/cv_fft_inpainting_lama, but its legacy CV
            # pipeline pulls a substantially larger dependency graph and is not
            # currently compatible with the supported Python 3.12 environment.
            # Keep a transparent, pinned fallback rather than silently fetching
            # an unversioned binary.
            weights = self.lama_weights_path()
            partial = weights.with_suffix(".pt.part")
            digest = hashlib.sha256()
            request = Request(
                LAMA_WEIGHTS_URL,
                headers={"User-Agent": "Manga-Localizer-Studio/0.3"},
            )
            try:
                with (
                    urlopen(request, timeout=180) as response,
                    partial.open("wb") as handle,
                ):
                    total_bytes = int(response.headers.get("Content-Length", "0"))
                    received = 0
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                        if total_bytes:
                            callback(
                                min(90, 10 + round(received / total_bytes * 80)),
                                "Downloading Big-LaMa weights",
                            )
                if digest.hexdigest() != LAMA_WEIGHTS_SHA256:
                    raise RuntimeError("Big-LaMa weight checksum mismatch")
                partial.replace(weights)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        elif model_id == "hy-mt2":
            self._require("modelscope")
            from modelscope import snapshot_download

            snapshot_download(spec.repository, local_dir=str(target))
        elif model_id == "manga-ocr":
            self._require("manga_ocr")
            from manga_ocr import MangaOcr

            callback(
                15, "Hugging Face fallback: manga-ocr has no ModelScope equivalent"
            )
            MangaOcr(force_cpu=True)
        elif model_id == "paddleocr":
            self._require("paddleocr")
            from paddleocr import PaddleOCR

            callback(15, "Downloading PaddleOCR models through ModelScope/PaddleX")
            PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                device="cpu",
                enable_mkldnn=False,
            )
        callback(95, f"Verifying {spec.name}")
        self.mark_ready(
            model_id, {"repository": spec.repository, "provider": spec.provider}
        )
        callback(100, f"{spec.name} is ready")
        return target

    def download_all(self, progress: ProgressCallback | None = None) -> None:
        callback = progress or (lambda _value, _message: None)
        total = len(MODEL_REGISTRY)
        for index, spec in enumerate(MODEL_REGISTRY):
            if self.is_ready(spec.id):
                callback(round((index + 1) / total * 100), f"{spec.name} already ready")
                continue
            self.download(
                spec.id,
                lambda value, message, base=index: callback(
                    round((base + value / 100) / total * 100), message
                ),
            )
