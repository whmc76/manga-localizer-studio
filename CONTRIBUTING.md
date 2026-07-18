# Contributing

1. Create a Python 3.12 environment with `scripts/bootstrap.ps1 -SkipModels -Dev`
   or `./scripts/bootstrap.sh --skip-models --dev`.
   With uv installed, both commands use the committed `uv.lock` automatically.
2. Keep source images immutable. Render only to a distinct output folder.
3. Run `uv run --frozen --no-sync pytest` before opening a pull request.
4. When changing the UI, update `docs/PARITY_LEDGER.md` and verify desktop and
   390 px mobile layouts.

By contributing, you agree that your contribution is licensed under MIT.
