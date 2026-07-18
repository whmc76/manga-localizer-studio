@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\bootstrap.ps1"
  if errorlevel 1 exit /b %errorlevel%
)
".venv\Scripts\python.exe" -m manga_localizer.cli ui
