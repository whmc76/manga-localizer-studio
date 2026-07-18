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
$PaddleGpuPackages = @(
    "paddlepaddle-gpu",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nvjitlink-cu12"
)

Set-Location $ProjectRoot
if ($Profile -eq "auto") {
    $Profile = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "cuda129" } else { "cpu" }
}

$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($UvCommand) {
    $SyncArgs = @("sync", "--locked", "--extra", "ml", "--python", "3.12")
    if ($Dev) { $SyncArgs += @("--extra", "test") }
    & uv @SyncArgs
    & uv pip uninstall --python $PythonPath @PaddleGpuPackages

    if ($Profile -eq "cuda129") {
        & uv pip install --python $PythonPath --reinstall torch --index-url https://download.pytorch.org/whl/cu129
    } else {
        & uv pip install --python $PythonPath --reinstall torch --index-url https://download.pytorch.org/whl/cpu
    }
} else {
    Write-Warning "uv is not installed; using the compatible venv + pip path. Install uv for locked, faster setup."
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            py -3.12 -m venv $VenvPath
        } else {
            python -m venv $VenvPath
        }
    }
    & $PythonPath -m pip install --upgrade pip wheel
    $ProjectExtra = if ($Dev) { ".[ml,test]" } else { ".[ml]" }
    & $PythonPath -m pip install -e $ProjectExtra
    & $PythonPath -m pip uninstall -y @PaddleGpuPackages
    if ($Profile -eq "cuda129") {
        & $PythonPath -m pip install torch --index-url https://download.pytorch.org/whl/cu129
    } else {
        & $PythonPath -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    }
}

& $PythonPath -m manga_localizer.cli assets download
if (-not $SkipModels) {
    & $PythonPath -m manga_localizer.cli models download all
}
& $PythonPath -m manga_localizer.cli doctor
Write-Host "Ready. Start with: .\start-windows.bat" -ForegroundColor Green
