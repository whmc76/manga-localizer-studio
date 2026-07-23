import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
WEB = ROOT / "src" / "manga_localizer" / "web"


def test_navigation_and_pipeline_contract():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert html.count('class="nav-item') == 4
    assert html.count('class="pipeline-step" data-phase=') == 4
    assert html.count("data-preview=") == 3
    assert 'id="prevPage" aria-label="上一页"' in html
    assert 'data-i18n-aria-label="preview.previous" disabled' in html
    assert 'id="nextPage" aria-label="下一页"' in html
    assert 'data-i18n-aria-label="preview.next" disabled' in html
    assert 'id="inferenceBackend"' in html
    assert 'id="inferenceBackendSetting"' in html
    assert 'id="ocrBackend"' in html
    assert 'id="ollamaOcrModel"' in html
    assert 'id="checkInference"' in html
    assert 'id="glossaryText"' not in html
    assert "huihui_ai/qwen3.5-abliterated:9b" in html


def test_ocr_phase_uses_zero_based_dom_index():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'if(job?.phase==="ocr")return' in javascript
    assert ">=.5?1:0" in javascript
    assert "const active=phaseIndex(job)" in javascript
    assert "state.activeBackend!==`${backend}:${ocr}`" in javascript
    assert 'textContent=t("settings.inference.notChecked")' in javascript


def test_ui_supports_persistent_chinese_japanese_and_english_locales():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    i18n = (WEB / "i18n.js").read_text(encoding="utf-8")
    assert '<script src="/i18n.js" defer></script>' in html
    assert html.index('/i18n.js') < html.index('/app.js')
    assert 'id="uiLanguage"' in html
    for locale in ("zh-CN", "ja-JP", "en-US"):
        assert f'<option value="{locale}">' in html
        assert f'"{locale}": {{' in i18n
    assert 'const STORAGE_KEY = "mls.uiLanguage"' in i18n
    assert "localStorage.setItem(STORAGE_KEY, locale)" in i18n
    assert 'window.addEventListener("mls:localechange",refreshLocalizedUI)' in javascript


def test_target_language_values_are_stable_across_ui_locales():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert '<option value="简体中文" data-i18n="target.zhHans">' in html
    assert '<option value="繁体中文" data-i18n="target.zhHant">' in html
    assert '<option value="English" data-i18n="target.english">' in html


def test_every_static_translation_key_exists_in_all_locales():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    i18n = (WEB / "i18n.js").read_text(encoding="utf-8")
    keys = set(re.findall(r'data-i18n(?:-[a-z-]+)?="([^"]+)"', html))
    keys.update(re.findall(r'\bt\("([^"]+)"', javascript))
    keys.update(
        {
            "models.role.paddleocr",
            "models.role.manga-ocr",
            "models.role.lama",
            "models.role.hy-mt2",
        }
    )
    missing = [key for key in sorted(keys) if i18n.count(f'"{key}":') != 3]
    assert missing == []


def test_preview_uses_available_height_and_contains_full_page():
    css = (WEB / "app.css").read_text(encoding="utf-8")
    assert ".preview-card{min-height:568px;overflow:hidden;display:grid" in css
    assert "grid-template-rows:52px minmax(458px,1fr) 56px" in css
    assert ".preview-canvas{height:auto;min-height:458px" in css
    assert (
        ".preview-pane img{display:block;width:100%;height:100%;object-fit:contain"
        in css
    )


def test_output_preview_waits_until_page_has_rendered():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'id="outputPending"' in html
    assert "function outputPreviewReady(job,page)" in javascript
    assert 'job.phase==="render"&&Number(job.current)>=page' in javascript
    assert "pending.textContent=pendingPreviewMessage(job)" in javascript
    assert "image.dataset.previewKey===key" in javascript


def test_ml_bootstrap_installs_matching_torchvision_backend():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    powershell = (ROOT / "scripts" / "bootstrap.ps1").read_text(encoding="utf-8")
    shell = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert '"torch==2.8.0"' in project
    assert '"torchvision==0.23.0"' in project
    assert '"torch==2.8.0+cu129", "torchvision==0.23.0+cu129"' in powershell
    assert "'torch==2.8.0+cu129' 'torchvision==0.23.0+cu129'" in shell
    assert "doctor --require-ml" in powershell
    assert "doctor --require-ml" in shell
