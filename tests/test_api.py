import time

from fastapi.testclient import TestClient
from PIL import Image

from manga_localizer.api import create_app
from manga_localizer.config import AppPaths


class FakePipeline:
    def __init__(self, paths):
        self.paths = paths

    def run(self, request, callback):
        files = sorted(request.source.glob("*.png"))
        request.output.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(files, 1):
            callback("ocr", index, len(files), source.name)
            Image.open(source).save(request.output / source.name)
        callback("complete", len(files), len(files), "done")
        return {"pages": len(files), "output": str(request.output), "images": []}


def make_paths(tmp_path):
    root = tmp_path / "home"
    return AppPaths(root, root / "models", root / "cache", root / "jobs", root / "settings.json")


def test_health_models_and_job_flow(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    assert client.get("/api/health").json()["status"] == "ok"
    assert len(client.get("/api/models").json()["models"]) == 4
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (32, 32), "white").save(source / "1.png")
    output = tmp_path / "output"
    response = client.post("/api/jobs", json={"source": str(source), "output": str(output)})
    assert response.status_code == 200
    job_id = response.json()["id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"complete", "failed"}:
            break
        time.sleep(0.01)
    assert job["status"] == "complete"
    assert job["total"] == 1
    assert output.joinpath("1.png").exists()


def test_rejects_source_as_output(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (16, 16), "white").save(source / "1.png")
    response = client.post("/api/jobs", json={"source": str(source), "output": str(source)})
    assert response.status_code == 422


def test_derives_non_destructive_output_path(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    source = tmp_path / "chapter-01"
    response = client.post("/api/paths/derive-output", json={"initial": str(source)})
    assert response.status_code == 200
    assert response.json()["path"] == str(tmp_path / "chapter-01_localized")


def test_online_secret_is_memory_only_and_builtin_model_becomes_optional(tmp_path):
    app_paths = make_paths(tmp_path)
    client = TestClient(create_app(app_paths, pipeline_factory=FakePipeline))
    response = client.put(
        "/api/settings",
        json={
            "inference_backend": "online",
            "online_base_url": "https://example.test/v1",
            "online_model": "translator-a",
            "online_api_key": "do-not-persist",
        },
    )
    assert response.status_code == 200
    assert response.json()["online_api_key_configured"] is True
    assert "online_api_key" not in response.json()
    assert "do-not-persist" not in app_paths.settings.read_text(encoding="utf-8")
    models = {item["id"]: item for item in client.get("/api/models").json()["models"]}
    assert models["hy-mt2"]["required"] is False


def test_inference_check_lists_remote_models(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "manga_localizer.api.available_remote_models",
        lambda backend, base_url, api_key: ["qwen-test"],
    )
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    response = client.post(
        "/api/inference/check",
        json={
            "inference_backend": "ollama",
            "ollama_base_url": "http://127.0.0.1:11434",
            "ollama_model": "qwen-test",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "backend": "ollama",
        "message": "已连接，发现 1 个模型",
        "models": ["qwen-test"],
    }
