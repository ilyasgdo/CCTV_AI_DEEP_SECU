$ErrorActionPreference = 'Stop'
Write-Host "[Sentinel-AI] Installation start"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python 3.10+ is required"
}

python -m venv venv
& .\venv\Scripts\python.exe -m pip install --upgrade pip
& .\venv\Scripts\python.exe -m pip install -r requirements.txt

$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    Write-Host "GPU detected: installing CUDA wheels"
    & .\venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
}
else {
    Write-Host "No GPU detected: installing CPU wheels"
    & .\venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

$dirs = @(
    "data/whitelist",
    "data/snapshots",
    "data/clips",
    "data/reports",
    "logs"
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host ".env created from .env.example"
}

Write-Host "[Sentinel-AI] Installation done"
Write-Host "Run: .\\venv\\Scripts\\Activate.ps1 ; python scripts/run.py"
