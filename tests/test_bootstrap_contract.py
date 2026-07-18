from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_uv_owns_locked_setup_and_hardware_overlays():
    powershell = (ROOT / "scripts" / "bootstrap.ps1").read_text(encoding="utf-8")
    shell = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    for script in (powershell, shell):
        assert "--locked" in script
        assert "--extra" in script and "ml" in script
        assert "uv pip install" in script
        assert "download.pytorch.org/whl/cpu" in script
        assert "download.pytorch.org/whl/cu129" in script
        assert "using the compatible venv + pip path" in script

    assert "astral-sh/setup-uv@" in workflow
    assert "uv sync --frozen --extra test" in workflow
    assert "uv run --frozen --no-sync pytest" in workflow
