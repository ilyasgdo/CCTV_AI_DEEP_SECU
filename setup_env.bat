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
echo [2/5] Installation de PyTorch avec CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [3/5] Installation des dependances principales...
pip install -r requirements.txt

echo [4/5] Installation des modules complementaires (IA, voix, dashboard)...
pip install flask edge-tts openai-whisper pyaudio requests fpdf2 scipy matplotlib

echo [5/5] Verification...
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
python -c "import flask; import edge_tts; print('Modules complementaires OK')"

echo.
echo ============================================
echo   Setup termine !
echo   Pour lancer : start.bat
echo ============================================
pause
