@echo off
chcp 65001 >NUL 2>&1
title CCTV AI DEEP SECU
echo.
echo  ============================================================
echo  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
echo  ░░  CCTV AI DEEP SECU — Systeme de Videosurveillance IA  ░░
echo  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
echo  ============================================================
echo.

REM === Verifier Python ===
python --version >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  [ERREUR] Python n'est pas installe !
    echo  Telechargez Python 3.10+ depuis : https://www.python.org/downloads/
    echo  IMPORTANT : Cochez "Add Python to PATH" lors de l'installation
    pause
    exit /b 1
)
echo  [OK] Python detecte

REM === Creer le venv si necessaire ===
if not exist "venv\Scripts\python.exe" (
    echo.
    echo  [1/4] Creation de l'environnement virtuel...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo  [ERREUR] Impossible de creer le venv
        pause
        exit /b 1
    )
    echo  [OK] Environnement virtuel cree
) else (
    echo  [OK] Environnement virtuel existe deja
)

REM === Activer le venv ===
call venv\Scripts\activate.bat

REM === Installer les dependances si necessaire ===
pip show torch >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [2/4] Installation de PyTorch avec CUDA...
    echo  (Cela peut prendre quelques minutes)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if %ERRORLEVEL% NEQ 0 (
        echo  [ATTENTION] Echec CUDA, tentative version CPU...
        pip install torch torchvision torchaudio
    )
    echo  [OK] PyTorch installe
) else (
    echo  [OK] PyTorch deja installe
)

pip show ultralytics >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [3/4] Installation des dependances du projet...
    pip install -r requirements.txt
    echo  [OK] Dependances installees
) else (
    echo  [OK] Dependances deja installees
)

REM === Verifier onnxruntime ===
pip show onnxruntime >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
    pip show onnxruntime-gpu >NUL 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo  [4/4] Installation de onnxruntime...
        pip install onnxruntime
    )
)

REM === Creer les dossiers necessaires ===
if not exist "data\whitelist_photos" mkdir "data\whitelist_photos"
if not exist "data\videos" mkdir "data\videos"

REM === Verification finale ===
echo.
echo  ============================================================
echo  [VERIFICATION] Test des modules...
echo  ============================================================
python -c "import torch; print(f'  PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('  YOLOv8: OK')"
echo.

REM === Choisir la source ===
echo  ============================================================
echo  CHOISIR LA SOURCE VIDEO :
echo  ============================================================
echo.
echo    [1] Webcam du PC (defaut)
echo    [2] Camera d'un telephone (IP Webcam)
echo    [3] Fichier video
echo.
set /p CHOIX="  Votre choix (1/2/3) : "

if "%CHOIX%"=="2" (
    echo.
    echo  Entrez l'URL du flux video de votre telephone.
    echo  Exemple: http://192.168.1.42:8080/video
    echo.
    set /p URL_TEL="  URL : "
    echo.
    echo  [LANCEMENT] Camera telephone : %URL_TEL%
    echo  Appuyez sur 'q' dans la fenetre pour quitter
    echo.
    python src/main.py --source "%URL_TEL%"
) else if "%CHOIX%"=="3" (
    echo.
    echo  Entrez le chemin du fichier video.
    echo  Exemple: C:\Users\ilyas\Videos\test.mp4
    echo.
    set /p FICHIER="  Chemin : "
    echo.
    echo  [LANCEMENT] Fichier : %FICHIER%
    echo  Appuyez sur 'q' dans la fenetre pour quitter
    echo.
    python src/main.py --source "%FICHIER%"
) else (
    echo.
    echo  [LANCEMENT] Webcam du PC
    echo  Appuyez sur 'q' dans la fenetre pour quitter
    echo.
    python src/main.py --source 0
)

echo.
echo  ============================================================
echo  SESSION TERMINEE
echo  ============================================================
pause
