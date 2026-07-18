from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import typer

from . import __version__
from .api import create_app
from .config import AppPaths, UserSettings
from .model_manager import MODEL_REGISTRY, ModelManager
from .pipeline import LocalizerPipeline, PipelineRequest
from .renderer import ensure_bold_font, ensure_font, find_bold_font, find_font


def configure_cli_encoding(*streams) -> None:
    """Use UTF-8 at the CLI boundary while keeping redirected streams compatible."""
    active_streams = streams or (sys.stdout, sys.stderr)
    for stream in active_streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def configure_dependency_logging() -> None:
    """Keep optional HTTP capability probes out of the user-facing console."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


configure_cli_encoding()
configure_dependency_logging()


app = typer.Typer(help="Local-first manga localization workspace.", no_args_is_help=True)
models_app = typer.Typer(help="Inspect and download OCR/translation models.")
assets_app = typer.Typer(help="Inspect and download non-model runtime assets.")
app.add_typer(models_app, name="models")
app.add_typer(assets_app, name="assets")


def existing_ui_url(host: str, port: int, timeout: float = 30.0) -> str:
    """Return the local workspace URL when this app already owns the port."""
    url = f"http://{host}:{port}"
    try:
        with urlopen(f"{url}/api/system", timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return ""
    return url if payload.get("local_only") is True and payload.get("version") else ""


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Bind host. Keep localhost for privacy."),
    port: int = typer.Option(8765, min=1, max=65535),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
):
    """Start the local web workspace."""
    running_url = existing_ui_url(host, port)
    if running_url:
        typer.echo(f"Manga Localizer Studio is already running at {running_url}")
        if open_browser:
            webbrowser.open(running_url)
        return

    import uvicorn

    if open_browser:
        threading.Thread(
            target=lambda: (time.sleep(1.0), webbrowser.open(f"http://{host}:{port}")),
            daemon=True,
        ).start()
    uvicorn.run(create_app(), host=host, port=port)


@app.command("run")
def run_pipeline(
    source: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output", "-o"),
    target_language: str = typer.Option("简体中文", "--target"),
    context_pages: int = typer.Option(3, min=0, max=12),
    story_context: bool = typer.Option(True, "--story-context/--no-story-context"),
    preserve_sfx: bool = typer.Option(False, "--preserve-sfx/--translate-sfx"),
    quality_profile: str = typer.Option("quality", "--quality-profile"),
    output_format: str = typer.Option("webp", "--output-format"),
    reviewed_transcript: Path | None = typer.Option(None, "--reviewed-transcript", exists=True),
    device: str = typer.Option("auto"),
    inference_backend: str = typer.Option("builtin", "--backend"),
    ocr_backend: str = typer.Option("builtin", "--ocr-backend"),
    ollama_base_url: str = typer.Option("http://127.0.0.1:11434", "--ollama-url"),
    ollama_model: str = typer.Option("qwen2.5:7b", "--ollama-model"),
    ollama_ocr_model: str = typer.Option("qwen2.5vl:7b", "--ollama-ocr-model"),
    online_base_url: str = typer.Option("https://api.openai.com/v1", "--online-url"),
    online_model: str = typer.Option("", "--online-model"),
):
    """Run the same localization pipeline used by the UI."""
    paths = AppPaths.from_env().ensure()
    request = PipelineRequest(
        source=source,
        output=output,
        target_language=target_language,
        context_pages=context_pages,
        story_context=story_context,
        preserve_sfx=preserve_sfx,
        quality_profile=quality_profile,
        output_format=output_format,
        device=device,
        inference_backend=inference_backend,
        ocr_backend=ocr_backend,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        ollama_ocr_model=ollama_ocr_model,
        online_base_url=online_base_url,
        online_model=online_model,
        online_api_key=os.environ.get("MLS_ONLINE_API_KEY", ""),
        reviewed_transcript=reviewed_transcript,
    )

    def progress(phase: str, current: int, total: int, message: str):
        typer.echo(f"[{phase}] {current}/{total} {message}")

    result = LocalizerPipeline(paths).run(request, progress)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@models_app.command("status")
def model_status():
    manager = ModelManager(AppPaths.from_env().ensure())
    for item in manager.status():
        state = "ready" if item["ready"] else "missing"
        typer.echo(f"{item['id']:<12} {state:<8} {item['provider']}  {item['path']}")


@models_app.command("download")
def model_download(model_id: str = typer.Argument("all")):
    manager = ModelManager(AppPaths.from_env().ensure())
    valid = {item.id for item in MODEL_REGISTRY}
    if model_id != "all" and model_id not in valid:
        raise typer.BadParameter(f"Choose one of: all, {', '.join(sorted(valid))}")

    def progress(value: int, message: str):
        typer.echo(f"[{value:3d}%] {message}")

    if model_id == "all":
        manager.download_all(progress)
    else:
        manager.download(model_id, progress)


@assets_app.command("download")
def asset_download():
    """Download pinned regular and bold Simplified Chinese fonts."""
    paths = AppPaths.from_env().ensure()
    regular = ensure_font(paths, force_managed=True)
    bold = ensure_bold_font(paths, force_managed=True)
    typer.echo(f"CJK regular font ready: {regular}")
    typer.echo(f"CJK bold font ready: {bold}")


def _font_status(finder, paths: AppPaths) -> str:
    try:
        return str(finder(paths))
    except FileNotFoundError:
        return "missing; run manga-localizer assets download"


def torch_pair_compatible(torch_version: str, torchvision_version: str) -> bool:
    return torch_version.split("+", 1)[0] == "2.8.0" and torchvision_version.split("+", 1)[0] == "0.23.0"


def _ml_runtime_status() -> dict:
    try:
        import torch
        import torchvision

        ready = torch_pair_compatible(torch.__version__, torchvision.__version__)
        return {
            "ready": ready,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "error": "" if ready else "Expected Torch 2.8.0 with Torchvision 0.23.0",
        }
    except Exception as exc:
        return {"ready": False, "torch": "missing", "torchvision": "missing", "error": str(exc)}


@app.command()
def doctor(require_ml: bool = typer.Option(False, help="Fail if the pinned ML runtime cannot load.")):
    """Print environment and cache diagnostics."""
    paths = AppPaths.from_env().ensure()
    settings = UserSettings.load(paths.settings)
    ml_runtime = _ml_runtime_status()
    payload = {
        "version": __version__,
        "home": str(paths.root),
        "settings": settings.__dict__,
        "models": ModelManager(paths).status(
            settings.inference_backend, settings.quality_profile, settings.ocr_backend
        ),
        "font": _font_status(find_font, paths),
        "bold_font": _font_status(find_bold_font, paths),
        "ml_runtime": ml_runtime,
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    if require_ml and not ml_runtime["ready"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
