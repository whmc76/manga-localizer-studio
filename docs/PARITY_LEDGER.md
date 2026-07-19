# UI parity ledger

Verified 2026-07-19 with the real FastAPI application at 1536×1024 and 390×844.
Evidence artifacts: `docs/screenshots/app-desktop.png` and `docs/screenshots/app-mobile.png`.

| # | Design requirement | Initial implementation | Required action | Evidence selector/check | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | 232px desktop sidebar with four icon+text items | implemented | verify | measured 232px; `.nav-item` count 4 | PASS |
| 2 | Mobile fixed bottom navigation | implemented | verify responsive shell | bottom edge 844 at 390×844; height 64px | PASS |
| 3 | Compact title topbar | implemented | verify | measured 72px | PASS |
| 4 | Source folder field and picker | implemented | verify API wiring | semantic textbox/button; `/api/dialog/folder` handler | PASS |
| 5 | Output field and auto-detect/picker | implemented | verify API wiring | `/api/paths/derive-output` and picker handlers | PASS |
| 6 | Target language selector | implemented | verify request | `collectSettings()` included in job payload | PASS |
| 7 | Exactly four pipeline steps | implemented | verify | `.pipeline-step` count 4 | PASS |
| 8 | Pipeline reflects live job phase | implemented | simulate persisted phases | detect/OCR/translate/render activated in order | PASS |
| 9 | Preview has 原图/译文/对比 controls | implemented | verify | `[data-preview]` count 3 | PASS |
| 10 | Compare mode is a split pane, not cards | implemented | verify | two `.preview-pane` figures; both real images loaded at 2126×3661 | PASS |
| 11 | Page navigation uses real page count | implemented | verify boundaries | completed job displayed 1/125; previous disabled and next enabled | PASS |
| 12 | One model row per registry entry | implemented | compare API and DOM | 4 API models; 4 `.model-row` elements | PASS |
| 13 | Model rows show downloading/ready/error | implemented | verify data rendering | chips render API/task state; no fake ready state | PASS |
| 14 | Model refresh is wired | implemented | verify handler | click calls `/api/models?refresh=true` | PASS |
| 15 | Model bootstrap button/action is real | implemented | verify task endpoint | POST task plus task-status polling | PASS |
| 16 | Story-context toggle and helper text | implemented | verify persistence | checkbox included in settings and job request | PASS |
| 17 | Context-page counter | implemented | verify bounds | 0–12 client bounds and Pydantic validation | PASS |
| 18 | SFX-preservation toggle | implemented | verify persistence | request value reaches translator and renderer | PASS |
| 19 | Coral start action | implemented | verify states/API | disabled without paths/models; POST `/api/jobs` | PASS |
| 20 | Local-processing privacy disclosure | implemented | verify visibility | visible beside action and in desktop sidebar | PASS |
| 21 | Execution status with phase and progress | implemented | verify persisted job | API-driven progress 0–100 and current/total | PASS |
| 22 | Pause control is real or explicitly disabled | implemented | verify semantics | disabled with explanatory title | PASS |
| 23 | Empty/loading/error states use real data | implemented | verify initial app | missing models and empty history shown from API | PASS |
| 24 | No horizontal overflow | implemented | measure both viewports | desktop 1521=1521 and mobile 375=375 scroll/client widths after scrollbar | PASS |
| 25 | First desktop viewport matches module density | implemented | visual comparison | actual 1536×1024 screenshot reviewed; preview canvas 736px high and uses inspector-created height | PASS |
| 26 | Mobile inspector precedes preview | implemented | measure layout | inspector top 497px; preview top 1315px | PASS |
| 27 | All primary controls keyboard accessible | implemented | keyboard check | native controls and visible focus; Tab reached input | PASS |
| 28 | No old/placeholder module remains | implemented | inspect DOM | `[data-placeholder]` count 0 | PASS |
| 29 | Inference backend selector | implemented | API + DOM contract | built-in, Ollama, and online modes persist | PASS |
| 30 | Backend-aware model requirements | implemented | model manager regression | Hy-MT2 optional outside built-in mode | PASS |
| 31 | Online privacy disclosure | implemented | verify backend UI state | images stay local; OCR text disclosure shown | PASS |
| 32 | Connection test and model discovery | implemented | API adapter tests | Ollama and compatible `/models` supported | PASS |
