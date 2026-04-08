#!/usr/bin/env bash
set -euo pipefail

echo "[Sentinel-AI] Installation start"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python3 is required (3.10+)"
  exit 1
fi

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected: installing CUDA wheels"
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
  echo "No GPU detected: installing CPU wheels"
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

python - <<'PY'
from pathlib import Path
for path in [
    Path("data/whitelist"),
    Path("data/snapshots"),
    Path("data/clips"),
    Path("data/reports"),
    Path("logs"),
]:
    path.mkdir(parents=True, exist_ok=True)
print("Data directories ready")
PY

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo ".env created from .env.example"
fi

echo "[Sentinel-AI] Installation done"
echo "Run: source venv/bin/activate ; python scripts/run.py"
