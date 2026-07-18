[CmdletBinding()]
param(
    [ValidateSet("auto", "cpu", "cuda129")]
    [string]$Profile = "auto",
    [switch]$SkipModels,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

Set-Location $ProjectRoot
if (-not (Test-Path -LiteralPath $PythonPath)) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv $VenvPath --python 3.12
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.12 -m venv $VenvPath
    } else {
        python -m venv $VenvPath
    }
}

& $PythonPath -m pip install --upgrade pip wheel
if ($Profile -eq "auto") {
    $Profile = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "cuda129" } else { "cpu" }
}

if ($Profile -eq "cuda129") {
    & $PythonPath -m pip install torch --index-url https://download.pytorch.org/whl/cu129
    & $PythonPath -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
} else {
    & $PythonPath -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    & $PythonPath -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
}

$ProjectExtra = if ($Dev) { ".[ml,test]" } else { ".[ml]" }
& $PythonPath -m pip install -e $ProjectExtra
& $PythonPath -m manga_localizer.cli assets download
if (-not $SkipModels) {
    & $PythonPath -m manga_localizer.cli models download all
}
& $PythonPath -m manga_localizer.cli doctor
Write-Host "Ready. Start with: .\start-windows.bat" -ForegroundColor Green
