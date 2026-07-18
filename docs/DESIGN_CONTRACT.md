# Design contract

## Authoritative artifact

- Desktop: `docs/design/design-desktop.png` at 1536 × 1024.
- Mobile: `docs/design/design-mobile.png` at 390 × 844.
- The screenshots above supersede the failed image-generation attempts and are the implementation contract.

## Product and page job

Manga Localizer Studio is a local-first creative workspace. A non-technical user selects an image folder, prepares OCR/translation models, starts a batch localization job, and reviews source/output pages without uploading artwork.

## Shell and grid

- Desktop: fixed 232px sidebar, 72px topbar, content width fills the remaining viewport.
- Main workspace: preview column plus a 265–360px inspector rail.
- Mobile: no desktop sidebar; four navigation items become a fixed 64px bottom bar. Inspector precedes preview.
- Primary setup, pipeline status, model status, settings, start action, preview, and execution status must all be reachable without hidden routes.

## Visual system

- Warm off-white canvas `#f2f0eb`, near-white surfaces `#fffdfa`, charcoal text `#262320`.
- Coral primary `#e66450`; mint success `#278c72` only for semantic readiness.
- 1px warm-gray borders, 10–12px radii, restrained shadows, 8px spacing rhythm.
- Dense professional workspace; no marketing hero, glassmorphism, generic dashboard mosaic, or decorative gradients.

## Module order

1. Sidebar / mobile bottom navigation.
2. Compact topbar.
3. Project setup: source, output, target language.
4. Four-step pipeline.
5. Preview with source/output/compare segmented control.
6. Model status with explicit download/readiness states.
7. Translation settings.
8. Start action with local-processing disclosure.
9. Execution status and progress.

## Data lifecycle

- Initial state comes from `/api/system`, `/api/models`, and local persisted settings. No sample job is inserted.
- Folder buttons open a local server-side directory picker when supported; paths remain manually editable.
- Model status is read from the model registry and cache inspection. Download actions create a real background bootstrap task and are polled.
- Start validates paths and model readiness, creates a persisted job, then polls real progress.
- Preview remains an explicit empty state until a real job/source page exists.
- Refresh re-reads model caches; it never fabricates readiness.

## Interactions

- Navigation switches visible workspace sections without replacing the shared shell.
- Model download, refresh, source/output selection, target selection, context-page controls, story mode, SFX preservation, job start, pause, preview mode, and page navigation have real handlers or an explicit disabled state.
- Start is disabled while required paths are missing or a job is running.

## Module-to-code map

| Design module | Owner | Action | Verification |
| --- | --- | --- | --- |
| App shell and navigation | `web/index.html`, `web/app.css` | create | `[data-testid=app-shell]`; nav count 4 |
| Project setup | `web/index.html`, `web/app.js` | create + wire API | source/output inputs and picker buttons |
| Four-step pipeline | `web/index.html`, `web/app.js` | create + job state | pipeline step count 4 |
| Preview split pane | `web/index.html`, `web/app.js` | create + real preview URLs | two preview slots in compare mode |
| Model status | `web/index.html`, `web/app.js`, `/api/models` | create + real cache state | model row count equals registry count |
| Translation settings | `web/index.html`, `web/app.js` | create + persist | context counter and toggles update settings |
| Primary action | `web/index.html`, `/api/jobs` | create + run/disabled states | creates a persisted job |
| Execution log | `web/index.html`, `/api/jobs/{id}` | create + poll | progress and phase reflect job state |
| Desktop sidebar | `web/app.css` | create | width 232px at 1536px viewport |
| Mobile bottom nav | `web/app.css` | create | fixed 64px bar at 390px viewport |

## Protected behavior

- Bind to `127.0.0.1` by default.
- Preserve input image dimensions and never overwrite source files.
- ModelScope is the preferred model source; Hugging Face is an explicit fallback only where no equivalent exists.
- Translation inference is selectable: bundled Hy-MT2, local Ollama, or an online OpenAI-compatible API.
- Changing the translation backend never changes OCR/render boundaries or the coherent page-context contract.
- Online mode visibly discloses that OCR text is sent remotely; API keys are never persisted.
- CLI and UI call the same application services and pipeline.

## Verification selectors

See `docs/PARITY_LEDGER.md`. The final implementation is not complete while any row is TODO, FAIL, or BLOCKED.
