from manga_localizer.config import AppPaths
from manga_localizer.model_manager import MODEL_REGISTRY, ModelManager


def paths(tmp_path):
    return AppPaths(
        tmp_path, tmp_path / "models", tmp_path / "cache",
        tmp_path / "jobs", tmp_path / "settings.json"
    )


def test_registry_uses_modelscope_first(tmp_path):
    status = ModelManager(paths(tmp_path)).status()
    assert len(status) == 3
    assert next(item for item in status if item["id"] == "hy-mt2")["provider"] == "ModelScope"
    assert "fallback" in next(item for item in status if item["id"] == "manga-ocr")["provider"]


def test_remote_translation_backend_does_not_require_builtin_translator(tmp_path):
    manager = ModelManager(paths(tmp_path))
    builtin = {item["id"]: item for item in manager.status("builtin")}
    ollama = {item["id"]: item for item in manager.status("ollama")}
    assert builtin["hy-mt2"]["required"] is True
    assert ollama["hy-mt2"]["required"] is False
    assert ollama["paddleocr"]["required"] is True


def test_ready_marker(tmp_path):
    manager = ModelManager(paths(tmp_path))
    assert not manager.is_ready(MODEL_REGISTRY[0].id)
    manager.mark_ready(MODEL_REGISTRY[0].id, {"test": True})
    assert manager.is_ready(MODEL_REGISTRY[0].id)
