from __future__ import annotations

import asyncio
import os
import platform
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import AppPaths, UserSettings
from .jobs import JobService
from .model_manager import ModelManager
from .ocr import list_images
from .pipeline import PipelineRequest
from .translator import available_remote_models


class FolderRequest(BaseModel):
    initial: str | None = None


class SettingsPayload(BaseModel):
    target_language: str = "简体中文"
    story_context: bool = True
    context_pages: int = Field(3, ge=0, le=12)
    preserve_sfx: bool = False
    quality_profile: Literal["quality", "fast"] = "quality"
    prefer_modelscope: bool = True
    device: str = "auto"
    inference_backend: Literal["builtin", "ollama", "online"] = "builtin"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    online_base_url: str = "https://api.openai.com/v1"
    online_model: str = ""
    online_api_key: str | None = Field(default=None, max_length=2048)


class JobPayload(SettingsPayload):
    source: str
    output: str
    glossary: dict[str, str] = Field(default_factory=dict)
    reviewed_transcript: str | None = None


class BootstrapPayload(BaseModel):
    model_ids: list[str] = Field(default_factory=lambda: ["paddleocr", "manga-ocr", "hy-mt2"])


class InferenceCheckPayload(BaseModel):
    inference_backend: Literal["builtin", "ollama", "online"]
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    online_base_url: str = "https://api.openai.com/v1"
    online_model: str = ""
    online_api_key: str | None = Field(default=None, max_length=2048)


def _pick_folder(initial: str | None = None) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(initialdir=initial or str(Path.home()))
        root.destroy()
        return chosen
    except Exception as exc:
        raise RuntimeError(f"Native folder picker is unavailable: {exc}") from exc


def create_app(paths: AppPaths | None = None, pipeline_factory=None) -> FastAPI:
    paths = (paths or AppPaths.from_env()).ensure()
    models = ModelManager(paths)
    jobs = JobService(paths, pipeline_factory) if pipeline_factory else JobService(paths)
    app = FastAPI(title="Manga Localizer Studio", version=__version__)
    tasks: dict[str, dict] = {}
    task_lock = threading.RLock()
    task_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-bootstrap")
    secrets = {"online_api_key": os.environ.get("MLS_ONLINE_API_KEY", "")}

    def public_settings(active: UserSettings) -> dict:
        return {
            **active.__dict__,
            "online_api_key_configured": bool(secrets["online_api_key"]),
        }

    def save_settings_payload(payload: SettingsPayload) -> UserSettings:
        if payload.online_api_key is not None:
            secrets["online_api_key"] = payload.online_api_key.strip()
        values = payload.model_dump(exclude={"online_api_key"})
        active = UserSettings(**values)
        active.save(paths.settings)
        return active

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/system")
    def system_info():
        active = UserSettings.load(paths.settings)
        return {
            "version": __version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "home": str(paths.root),
            "local_only": active.inference_backend != "online",
            "settings": public_settings(active),
        }

    @app.get("/api/models")
    def model_status(refresh: bool = Query(False)):
        del refresh
        active = UserSettings.load(paths.settings)
        return {
            "models": models.status(active.inference_backend),
            "source_policy": "ModelScope first",
        }

    def bootstrap(task_id: str, model_ids: list[str]):
        try:
            for index, model_id in enumerate(model_ids):
                models.download(
                    model_id,
                    lambda value, message, idx=index: _update_task(
                        task_id,
                        status="running",
                        progress=round((idx + value / 100) / len(model_ids) * 100),
                        message=message,
                    ),
                )
            _update_task(task_id, status="complete", progress=100, message="全部模型已就绪")
        except Exception as exc:
            _update_task(task_id, status="failed", error=str(exc), message="模型准备失败")

    def _update_task(task_id: str, **values):
        with task_lock:
            tasks.setdefault(task_id, {}).update(values)

    @app.post("/api/models/bootstrap")
    def start_bootstrap(payload: BootstrapPayload):
        task_id = uuid.uuid4().hex[:12]
        tasks[task_id] = {
            "id": task_id,
            "status": "queued",
            "progress": 0,
            "message": "等待下载",
            "error": None,
        }
        task_pool.submit(bootstrap, task_id, payload.model_ids)
        return tasks[task_id]

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str):
        with task_lock:
            task = tasks.get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        return task

    @app.get("/api/settings")
    def get_settings():
        return public_settings(UserSettings.load(paths.settings))

    @app.put("/api/settings")
    def put_settings(payload: SettingsPayload):
        return public_settings(save_settings_payload(payload))

    @app.post("/api/inference/check")
    def check_inference(payload: InferenceCheckPayload):
        if payload.inference_backend == "builtin":
            ready = models.is_ready("hy-mt2")
            return {
                "ok": ready,
                "backend": "builtin",
                "message": "Hy-MT2 已就绪" if ready else "Hy-MT2 尚未下载",
                "models": ["hy-mt2"] if ready else [],
            }
        api_key = (
            payload.online_api_key.strip()
            if payload.online_api_key is not None
            else secrets["online_api_key"]
        )
        base_url = (
            payload.ollama_base_url
            if payload.inference_backend == "ollama"
            else payload.online_base_url
        )
        try:
            available = available_remote_models(payload.inference_backend, base_url, api_key)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        selected = (
            payload.ollama_model
            if payload.inference_backend == "ollama"
            else payload.online_model
        )
        return {
            "ok": bool(available) and (not selected or selected in available),
            "backend": payload.inference_backend,
            "message": (
                f"已连接，发现 {len(available)} 个模型"
                if not selected or selected in available
                else f"已连接，但未找到模型 {selected}"
            ),
            "models": available,
        }

    @app.post("/api/dialog/folder")
    async def folder_dialog(payload: FolderRequest):
        try:
            selected = await asyncio.to_thread(_pick_folder, payload.initial)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"path": selected}

    @app.post("/api/paths/derive-output")
    def derive_output(payload: FolderRequest):
        if not payload.initial:
            return {"path": ""}
        source = Path(payload.initial).expanduser()
        return {"path": str(source.parent / f"{source.name}_localized")}

    @app.post("/api/jobs")
    def create_job(payload: JobPayload):
        source = Path(payload.source).expanduser().resolve()
        output = Path(payload.output).expanduser().resolve()
        if not source.is_dir():
            raise HTTPException(422, "Source folder does not exist")
        try:
            image_count = len(list_images(source))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if image_count == 0:
            raise HTTPException(422, "No supported images found")
        if source == output:
            raise HTTPException(422, "Output folder must differ from source")
        settings = SettingsPayload(
            **payload.model_dump(exclude={"source", "output", "glossary", "reviewed_transcript"})
        )
        active = save_settings_payload(settings)
        api_key = payload.online_api_key
        if api_key is None:
            api_key = secrets["online_api_key"]
        request = PipelineRequest(
            source=source,
            output=output,
            target_language=payload.target_language,
            story_context=payload.story_context,
            context_pages=payload.context_pages,
            preserve_sfx=payload.preserve_sfx,
            quality_profile=payload.quality_profile,
            device=payload.device,
            glossary=payload.glossary,
            inference_backend=active.inference_backend,
            ollama_base_url=active.ollama_base_url,
            ollama_model=active.ollama_model,
            online_base_url=active.online_base_url,
            online_model=active.online_model,
            online_api_key=api_key or "",
            reviewed_transcript=(
                Path(payload.reviewed_transcript).expanduser().resolve()
                if payload.reviewed_transcript
                else None
            ),
        )
        return jobs.submit(request, total=image_count)

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": jobs.store.list()}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/api/jobs/{job_id}/preview/{kind}/{page_number}")
    def preview(job_id: str, kind: str, page_number: int):
        job = jobs.store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        source = Path(job["request"]["source"])
        output = Path(job["request"]["output"])
        files = list_images(source)
        if page_number < 1 or page_number > len(files):
            raise HTTPException(404, "Page not found")
        source_file = files[page_number - 1]
        if kind == "source":
            path = source_file
        elif kind == "output":
            path = output / f"{source_file.stem}.png"
        else:
            raise HTTPException(422, "Preview kind must be source or output")
        if not path.exists():
            raise HTTPException(404, "Preview is not ready")
        return FileResponse(path)

    web_dir = Path(__file__).resolve().parent / "web"
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    app.state.paths = paths
    app.state.models = models
    app.state.jobs = jobs
    return app


app = create_app()
