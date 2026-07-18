@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\bootstrap.ps1" -SkipModels
  if errorlevel 1 exit /b %errorlevel%
)
".venv\Scripts\python.exe" -m manga_localizer.cli ui
if errorlevel 1 (
  echo.
  echo Manga Localizer Studio failed to start. Review the message above.
  pause
  exit /b %errorlevel%
)
