@echo off
echo ============================================
echo   CCTV AI DEEP SECU — Setup Environment
echo ============================================
echo.

REM Verifier Python
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo X Python non trouve ! Installer Python 3.10 ou 3.11
    pause
    exit /b 1
)

REM Verifier CUDA
nvcc --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo ! CUDA non trouve. Les modeles tourneront sur CPU (tres lent)
)

REM Creer l'environnement virtuel
if not exist venv (
    echo [1/4] Creation de l'environnement virtuel...
    python -m venv venv
) else (
    echo [1/4] Environnement virtuel existant trouve
)

REM Activer
call venv\Scripts\activate.bat

REM Installer les dependances
echo [2/4] Installation de PyTorch avec CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [3/4] Installation des dependances...
pip install -r requirements.txt

echo [4/4] Verification...
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"

echo.
echo ============================================
echo   Setup termine !
echo   Pour lancer : python src/main.py
echo ============================================
pause
