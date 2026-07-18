# Manga Localizer Studio 0.4.5

This launcher patch makes double-click startup idempotent.

- When a Manga Localizer Studio instance already owns the configured local port, `manga-localizer ui` opens that workspace and exits successfully instead of attempting a conflicting bind.
- A different service on the port is never mistaken for the workspace; the `/api/system` identity contract must match.
- If startup genuinely fails, `start-windows.bat` now keeps the window open and shows an actionable error instead of flashing closed.

The fix was reproduced against the real background 125-page test instance on port 8765 without interrupting its OCR job.
