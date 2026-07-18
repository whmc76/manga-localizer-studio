import json
import os

from manga_localizer.config import AppPaths, UserSettings, configure_model_caches


def test_settings_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    expected = UserSettings(context_pages=5, preserve_sfx=False)
    expected.save(path)
    assert UserSettings.load(path) == expected
    assert json.loads(path.read_text(encoding="utf-8"))["context_pages"] == 5


def test_model_caches_are_scoped(monkeypatch, tmp_path):
    for name in ("MODELSCOPE_CACHE", "HF_HOME", "PADDLE_PDX_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    paths = AppPaths(
        tmp_path, tmp_path / "models", tmp_path / "cache",
        tmp_path / "jobs", tmp_path / "settings.json"
    )
    configure_model_caches(paths)
    assert os.environ["MODELSCOPE_CACHE"].startswith(str(tmp_path))
    assert os.environ["PADDLE_PDX_MODEL_SOURCE"] == "modelscope"
