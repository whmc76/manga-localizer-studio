from pathlib import Path


ROOT = Path(__file__).parents[1]
WEB = ROOT / "src" / "manga_localizer" / "web"


def test_navigation_and_pipeline_contract():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert html.count('class="nav-item') == 4
    assert html.count('class="pipeline-step" data-phase=') == 4
    assert html.count("data-preview=") == 3
    assert 'id="prevPage" aria-label="上一页" disabled' in html
    assert 'id="nextPage" aria-label="下一页" disabled' in html
    assert 'id="inferenceBackend"' in html
    assert 'id="inferenceBackendSetting"' in html
    assert 'id="ocrBackend"' in html
    assert 'id="ollamaOcrModel"' in html
    assert 'id="checkInference"' in html


def test_ocr_phase_uses_zero_based_dom_index():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'if(job?.phase==="ocr")return' in javascript
    assert '>=.5?1:0' in javascript
    assert "const active=phaseIndex(job)" in javascript
    assert 'state.activeBackend!==`${backend}:${ocr}`' in javascript
    assert 'textContent="尚未测试连接"' in javascript


def test_preview_uses_available_height_and_contains_full_page():
    css = (WEB / "app.css").read_text(encoding="utf-8")
    assert ".preview-card{min-height:568px;overflow:hidden;display:grid" in css
    assert "grid-template-rows:52px minmax(458px,1fr) 56px" in css
    assert ".preview-canvas{height:auto;min-height:458px" in css
    assert ".preview-pane img{display:block;width:100%;height:100%;object-fit:contain" in css


def test_output_preview_waits_until_page_has_rendered():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'id="outputPending"' in html
    assert 'function outputPreviewReady(job,page)' in javascript
    assert 'job.phase==="render"&&Number(job.current)>=page' in javascript
    assert 'pending.textContent=pendingPreviewMessage(job)' in javascript
    assert 'image.dataset.previewKey===key' in javascript


def test_ml_bootstrap_installs_matching_torchvision_backend():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    powershell = (ROOT / "scripts" / "bootstrap.ps1").read_text(encoding="utf-8")
    shell = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert '"torchvision>=0.23,<1"' in project
    assert '$TorchPackages = @("torch", "torchvision")' in powershell
    assert "torch torchvision --index-url" in shell
