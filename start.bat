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

REM === Verification ===
echo.
python -c "import torch; print(f'  PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"

:MENU
echo.
echo  ============================================================
echo  MENU PRINCIPAL :
echo  ============================================================
echo.
echo    [1] Lancer la surveillance (webcam PC)
echo    [2] Lancer la surveillance (camera telephone USB)
echo    [3] Lancer la surveillance (camera telephone Wi-Fi)
echo    [4] Lancer la surveillance (fichier video)
echo    [5] Capturer des visages (construire la whitelist)
echo    [6] Benchmark performance
echo    [7] Quitter
echo.
set /p CHOIX="  Votre choix (1-7) : "

if "%CHOIX%"=="1" (
    echo.
    echo  [LANCEMENT] Webcam du PC
    echo  Raccourcis : [Q] Quitter  [S] Statistiques
    echo.
    python src/main.py --source 0
    goto MENU
)

if "%CHOIX%"=="2" (
    echo.
    echo  ============================================================
    echo  CAMERA TELEPHONE VIA USB
    echo  ============================================================
    echo.
    echo  1. Installer DroidCam (Android) ou EpocCam (iPhone)
    echo  2. Installer le client DroidCam/EpocCam sur le PC
    echo  3. Brancher le telephone en USB
    echo  4. Lancer DroidCam Client sur le PC ^(USB mode^)
    echo.
    echo  Le telephone apparait comme webcam index 1, 2 ou 3.
    echo.
    set /p USB_IDX="  Index de la camera USB (1, 2 ou 3) [defaut: 1] : "
    if "%USB_IDX%"=="" set USB_IDX=1
    echo.
    echo  [LANCEMENT] Camera USB (index %USB_IDX%)
    echo  Raccourcis : [Q] Quitter  [S] Statistiques
    echo.
    python src/main.py --source %USB_IDX%
    goto MENU
)

if "%CHOIX%"=="3" (
    echo.
    echo  ============================================================
    echo  CAMERA TELEPHONE VIA WI-FI
    echo  ============================================================
    echo.
    echo  1. Installer "IP Webcam" ou "DroidCam" sur le telephone
    echo  2. Ouvrir l'app et demarrer le serveur
    echo  3. PC et telephone sur le meme Wi-Fi
    echo.
    echo  Exemples d'URL :
    echo    IP Webcam  : http://192.168.1.42:8080/video
    echo    DroidCam   : http://192.168.1.42:4747/video
    echo.
    set /p URL_TEL="  URL du flux : "
    echo.
    echo  [LANCEMENT] Camera Wi-Fi
    echo  Raccourcis : [Q] Quitter  [S] Statistiques
    echo.
    python src/main.py --source "%URL_TEL%"
    goto MENU
)

if "%CHOIX%"=="4" (
    echo.
    set /p FICHIER="  Chemin du fichier video : "
    echo.
    echo  [LANCEMENT] Fichier video
    echo.
    python src/main.py --source "%FICHIER%"
    goto MENU
)

if "%CHOIX%"=="5" (
    echo.
    echo  ============================================================
    echo  CAPTURE DE VISAGES — Whitelist
    echo  ============================================================
    echo.
    echo    [A] Capturer avec la webcam du PC
    echo    [B] Capturer avec camera telephone (USB index)
    echo    [C] Capturer avec camera telephone (Wi-Fi URL)
    echo    [D] Construire la whitelist (sans camera)
    echo.
    set /p WL_CHOIX="  Votre choix (A/B/C/D) : "
    
    if /i "%WL_CHOIX%"=="A" (
        python tools/whitelist_capture.py --source 0
    ) else if /i "%WL_CHOIX%"=="B" (
        set /p WL_USB="  Index USB (1, 2 ou 3) : "
        python tools/whitelist_capture.py --source %WL_USB%
    ) else if /i "%WL_CHOIX%"=="C" (
        set /p WL_URL="  URL du flux : "
        python tools/whitelist_capture.py --source "%WL_URL%"
    ) else if /i "%WL_CHOIX%"=="D" (
        python tools/whitelist_capture.py --build
    )
    goto MENU
)

if "%CHOIX%"=="6" (
    echo.
    echo  [BENCHMARK] Test de performance...
    echo.
    python tests/benchmark.py
    echo.
    pause
    goto MENU
)

if "%CHOIX%"=="7" (
    echo.
    echo  Au revoir !
    exit /b 0
)

echo  Choix invalide, reessayez.
goto MENU
