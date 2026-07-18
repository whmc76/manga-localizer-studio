# Verification report

Date: 2026-07-18

## Automated checks

- 14 unit/API/contract tests passed on Windows with Python 3.12.
- Source distribution and wheel built successfully.
- Python source compiled with `compileall`.
- PowerShell bootstrap parsed without errors; Git Bash accepted both shell scripts.
- The renderer regression test confirms every pixel outside the OCR box is unchanged.
- Managed-font download is header/size validated and atomically installed.

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
