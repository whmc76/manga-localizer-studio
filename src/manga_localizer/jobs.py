from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .config import AppPaths
from .pipeline import LocalizerPipeline, PipelineRequest


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, folder: Path):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, job_id: str) -> Path:
        return self.folder / f"{job_id}.json"

    def create(self, request: PipelineRequest, total: int = 0) -> dict:
        job_id = uuid.uuid4().hex[:12]
        now = utc_now()
        job = {
            "id": job_id,
            "status": "queued",
            "phase": "queued",
            "progress": 0,
            "current": 0,
            "total": total,
            "message": "等待开始",
            "error": None,
            "created_at": now,
            "updated_at": now,
            "request": {
                "source": str(request.source),
                "output": str(request.output),
                "target_language": request.target_language,
                "story_context": request.story_context,
                "context_pages": request.context_pages,
                "preserve_sfx": request.preserve_sfx,
                "device": request.device,
                "glossary": request.glossary or {},
            },
            "result": None,
        }
        self.save(job)
        return job

    def save(self, job: dict) -> None:
        with self._lock:
            job["updated_at"] = utc_now()
            self._path(job["id"]).write_text(
                json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def get(self, job_id: str) -> dict | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        with self._lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        jobs = []
        for path in self.folder.glob("*.json"):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)


class JobService:
    def __init__(self, paths: AppPaths, pipeline_factory=LocalizerPipeline):
        self.paths = paths.ensure()
        self.store = JobStore(paths.jobs)
        self.pipeline_factory = pipeline_factory
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="manga-localizer")

    def submit(self, request: PipelineRequest, total: int = 0) -> dict:
        # Store the page count before the worker can start, avoiding an API-side
        # follow-up save that could race and overwrite live progress fields.
        job = self.store.create(request, total=total)
        self.executor.submit(self._run, job["id"], request)
        return job

    def _run(self, job_id: str, request: PipelineRequest) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        job.update(status="running", phase="prepare", message="准备本地化")
        self.store.save(job)

        def progress(phase: str, current: int, total: int, message: str) -> None:
            active = self.store.get(job_id)
            if active is None:
                return
            value = round(current / total * 100) if total else 0
            # Each phase occupies a stable part of the overall progress bar.
            offsets = {"ocr": (0, 45), "translate": (45, 25), "render": (70, 30)}
            if phase in offsets:
                base, span = offsets[phase]
                value = base + round((current / total if total else 0) * span)
            elif phase == "complete":
                value = 100
            active.update(
                status="running" if phase != "complete" else "complete",
                phase=phase,
                progress=value,
                current=current,
                total=total,
                message=message,
            )
            self.store.save(active)

        try:
            result = self.pipeline_factory(self.paths).run(request, progress)
            job = self.store.get(job_id) or job
            job.update(
                status="complete",
                phase="complete",
                progress=100,
                message="本地化完成",
                result=result,
            )
        except Exception as exc:
            job = self.store.get(job_id) or job
            job.update(status="failed", phase="failed", error=str(exc), message="任务失败")
        self.store.save(job)
