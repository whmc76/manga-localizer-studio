import hashlib
import io

from manga_localizer.config import AppPaths
from manga_localizer.model_manager import MODEL_REGISTRY, ModelManager


def paths(tmp_path):
    return AppPaths(
        tmp_path,
        tmp_path / "models",
        tmp_path / "cache",
        tmp_path / "jobs",
        tmp_path / "settings.json",
    )


def test_registry_uses_modelscope_first(tmp_path):
    status = ModelManager(paths(tmp_path)).status()
    assert len(status) == 4
    assert (
        next(item for item in status if item["id"] == "hy-mt2")["provider"]
        == "ModelScope"
    )
    assert (
        "fallback"
        in next(item for item in status if item["id"] == "manga-ocr")["provider"]
    )
    assert (
        "fallback"
        in next(item for item in status if item["id"] == "lama")["provider"].lower()
    )


def test_remote_translation_backend_does_not_require_builtin_translator(tmp_path):
    manager = ModelManager(paths(tmp_path))
    builtin = {item["id"]: item for item in manager.status("builtin")}
    ollama = {item["id"]: item for item in manager.status("ollama")}
    assert builtin["hy-mt2"]["required"] is True
    assert ollama["hy-mt2"]["required"] is True
    ollama_fast = {item["id"]: item for item in manager.status("ollama", "fast")}
    assert ollama_fast["hy-mt2"]["required"] is False
    assert ollama["paddleocr"]["required"] is True
    assert builtin["lama"]["required"] is True
    fast = {item["id"]: item for item in manager.status("builtin", "fast")}
    assert fast["lama"]["required"] is False


def test_ready_marker(tmp_path):
    manager = ModelManager(paths(tmp_path))
    assert not manager.is_ready(MODEL_REGISTRY[0].id)
    manager.mark_ready(MODEL_REGISTRY[0].id, {"test": True})
    assert manager.is_ready(MODEL_REGISTRY[0].id)


def test_lama_download_is_checksum_validated_and_atomic(monkeypatch, tmp_path):
    payload = b"pinned-lama-test-payload"
    monkeypatch.setattr(
        "manga_localizer.model_manager.LAMA_WEIGHTS_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

    monkeypatch.setattr(
        "manga_localizer.model_manager.urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )
    manager = ModelManager(paths(tmp_path))
    target = manager.download("lama")
    weights = target / "big-lama.pt"
    assert weights.read_bytes() == payload
    assert manager.is_ready("lama")
    assert not weights.with_suffix(".pt.part").exists()
