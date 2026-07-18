# Verification report

Date: 2026-07-18

## Automated checks

- 30 unit/API/contract tests passed on Windows with Python 3.12.
- Source distribution and wheel built successfully.
- Python source compiled with `compileall`.
- PowerShell bootstrap parsed without errors; Git Bash accepted both shell scripts.
- Renderer regressions confirm pixels outside the bounded cleanup area remain unchanged and edge-connected panel art survives grouped-text cleanup.
- Managed-font download is header/size validated and atomically installed.

## Full-book acceptance test

- Ran the built-in backend end to end on a 125-page, 2126×3661 source book (final page 2800×3808).
- Produced 125 PNG files and a 125-page transcript; every output dimension matched its source.
- Detected 898 text units: 741 translated units, 157 preserved sound-effect units, and zero missing non-SFX translations.
- Quality-gate audit found zero remaining kana-bearing or context-leaking translations after bounded retry.
- Pixel comparison on pages 1, 3, 25, 50, 75, 100, and 125 found zero changed pixels outside the declared cleanup/typesetting regions.

## Browser checks

The real FastAPI application was inspected with Playwright, not a static mock.

| Check | Desktop 1536×1024 | Mobile 390×844 |
|---|---:|---:|
| Document scroll width / client width | 1536 / 1536 | 390 / 390 |
| Sidebar width / position | 232px | fixed 64px bottom navigation |
| Topbar height | 72px | 62px |
| Model rows from API | 3 | 3 |
| Inspector before preview | n/a | 474px / 1057px top |
| Console errors | 0 | 0 |

Interactions checked: all four navigation views, quick-start dialog, model refresh,
backend switching, conditional Ollama/online fields, failed-connection recovery,
keyboard focus, empty preview boundaries, and phase-to-step state mapping.

## Adversarial user complaints

- “模型明明没下，为什么显示可用？” — the UI renders `未下载` from `/api/models`
  and disables Start until all required model markers exist.
- “手机上还没设置就先看到一大块预览。” — the mobile inspector is ordered before
  preview; measured tops are 474px and 1057px.
- “刚开始 OCR，怎么第二步已经亮了？” — the zero/one-based phase mapping was fixed;
  browser simulation verifies detect → OCR → translate → render.
- “我担心它覆盖原图。” — equal source/output paths are rejected by API and pipeline,
  and renderer output is always PNG in a distinct directory.
