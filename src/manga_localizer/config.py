from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path
    models: Path
    cache: Path
    jobs: Path
    settings: Path

    @classmethod
    def from_env(cls) -> "AppPaths":
        root = Path(
            os.environ.get("MLS_HOME", Path.home() / ".manga-localizer-studio")
        ).expanduser().resolve()
        return cls(
            root=root,
            models=root / "models",
            cache=root / "cache",
            jobs=root / "jobs",
            settings=root / "settings.json",
        )

    def ensure(self) -> "AppPaths":
        for path in (self.root, self.models, self.cache, self.jobs):
            path.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class UserSettings:
    target_language: str = "简体中文"
    story_context: bool = True
    context_pages: int = 3
    preserve_sfx: bool = True
    prefer_modelscope: bool = True
    device: str = "auto"

    @classmethod
    def load(cls, path: Path) -> "UserSettings":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def configure_model_caches(paths: AppPaths) -> None:
    """Keep every downloaded artifact inside the application data directory."""
    paths.ensure()
    os.environ.setdefault("MODELSCOPE_CACHE", str(paths.models / "modelscope-cache"))
    os.environ.setdefault("HF_HOME", str(paths.models / "huggingface"))
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(paths.models / "paddlex"))
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
