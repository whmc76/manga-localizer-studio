from __future__ import annotations

import json
import threading
import time
import webbrowser
from pathlib import Path

import typer

from . import __version__
from .api import create_app
from .config import AppPaths, UserSettings
from .model_manager import MODEL_REGISTRY, ModelManager
from .pipeline import LocalizerPipeline, PipelineRequest
from .renderer import ensure_font, find_font


app = typer.Typer(help="Local-first manga localization workspace.", no_args_is_help=True)
models_app = typer.Typer(help="Inspect and download OCR/translation models.")
assets_app = typer.Typer(help="Inspect and download non-model runtime assets.")
app.add_typer(models_app, name="models")
app.add_typer(assets_app, name="assets")


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Bind host. Keep localhost for privacy."),
    port: int = typer.Option(8765, min=1, max=65535),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
):
    """Start the local web workspace."""
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
    preserve_sfx: bool = typer.Option(True, "--preserve-sfx/--translate-sfx"),
    device: str = typer.Option("auto"),
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
        device=device,
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
    """Download the pinned Simplified Chinese font into the app data folder."""
    path = ensure_font(AppPaths.from_env().ensure(), force_managed=True)
    typer.echo(f"CJK font ready: {path}")


@app.command()
def doctor():
    """Print environment and cache diagnostics."""
    paths = AppPaths.from_env().ensure()
    settings = UserSettings.load(paths.settings)
    try:
        font = str(find_font(paths))
    except FileNotFoundError:
        font = "missing; run manga-localizer assets download"
    payload = {
        "version": __version__,
        "home": str(paths.root),
        "settings": settings.__dict__,
        "models": ModelManager(paths).status(),
        "font": font,
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
