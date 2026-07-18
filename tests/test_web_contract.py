from pathlib import Path


WEB = Path(__file__).parents[1] / "src" / "manga_localizer" / "web"


def test_navigation_and_pipeline_contract():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert html.count('class="nav-item') == 4
    assert html.count('class="pipeline-step" data-phase=') == 4
    assert html.count("data-preview=") == 3
    assert 'id="prevPage" aria-label="上一页" disabled' in html
    assert 'id="nextPage" aria-label="下一页" disabled' in html


def test_ocr_phase_uses_zero_based_dom_index():
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'if(job?.phase==="ocr")return' in javascript
    assert '>=.5?1:0' in javascript
    assert "const active=phaseIndex(job)" in javascript
