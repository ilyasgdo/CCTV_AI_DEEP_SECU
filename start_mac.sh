#!/bin/bash
# ============================================================
#  CCTV AI DEEP SECU — Lanceur macOS (Apple Silicon M1/M2/M3)
#  Installation automatique + Menu interactif
# ============================================================

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  CCTV AI DEEP SECU — Système de Vidéosurveillance IA${NC}"
echo -e "${CYAN}  Compatible Apple Silicon (M1/M2/M3/M4)${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# === Détecter le répertoire du script ===
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# === Vérifier Python ===
PYTHON=""
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo -e "${RED}❌ Python non trouvé !${NC}"
    echo "  Installez Python 3.10+ :"
    echo "    brew install python@3.12"
    echo "  ou téléchargez depuis https://www.python.org/downloads/"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1)
echo -e "${GREEN}✅ $PY_VERSION${NC}"

# === Vérifier / Créer l'environnement virtuel ===
if [ ! -d "venv" ]; then
    echo ""
    echo -e "${YELLOW}[SETUP] Première exécution — Installation automatique...${NC}"
    echo ""

    # Créer le venv
    echo -e "  ${CYAN}[1/4]${NC} Création de l'environnement virtuel..."
    $PYTHON -m venv venv
    echo -e "  ${GREEN}✅ venv créé${NC}"

    # Activer le venv
    source venv/bin/activate

    # Mettre à jour pip
    echo -e "  ${CYAN}[2/4]${NC} Mise à jour de pip..."
    pip install --upgrade pip --quiet

    # Installer PyTorch pour Apple Silicon (MPS)
    echo -e "  ${CYAN}[3/4]${NC} Installation de PyTorch (Apple Silicon MPS)..."
    pip install torch torchvision --quiet
    echo -e "  ${GREEN}✅ PyTorch installé${NC}"

    # Installer les dépendances
    echo -e "  ${CYAN}[4/4]${NC} Installation des dépendances..."
    pip install ultralytics opencv-python numpy insightface==0.2.1 onnxruntime --quiet
    echo -e "  ${GREEN}✅ Toutes les dépendances installées${NC}"

    # Créer les dossiers nécessaires
    mkdir -p data/whitelist_photos
    mkdir -p data/videos
    mkdir -p src/database

    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}  ✅ INSTALLATION TERMINÉE${NC}"
    echo -e "${GREEN}============================================================${NC}"
else
    # Activer le venv existant
    source venv/bin/activate
fi

# === Vérifier MPS ===
echo ""
$PYTHON -c "
import torch
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('\033[0;32m✅ Apple Silicon MPS détecté — accélération GPU active\033[0m')
elif torch.cuda.is_available():
    print('\033[0;32m✅ NVIDIA CUDA détecté — accélération GPU active\033[0m')
else:
    print('\033[1;33m⚠️  Pas de GPU détecté — mode CPU (plus lent)\033[0m')
" 2>/dev/null || echo -e "${YELLOW}⚠️ Impossible de vérifier le GPU${NC}"

# === Menu principal ===
while true; do
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}  MENU PRINCIPAL :${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
    echo "    [1] Lancer la surveillance (webcam Mac)"
    echo "    [2] Lancer la surveillance (caméra téléphone USB)"
    echo "    [3] Lancer la surveillance (caméra téléphone Wi-Fi)"
    echo "    [4] Lancer la surveillance (fichier vidéo)"
    echo "    [5] Capturer des visages (construire la whitelist)"
    echo "    [6] Benchmark performance"
    echo "    [7] Quitter"
    echo ""
    read -p "  Votre choix (1-7) : " CHOIX

    case $CHOIX in
        1)
            echo ""
            echo -e "  ${GREEN}[LANCEMENT]${NC} Webcam du Mac"
            echo "  Raccourcis : [Q] Quitter  [S] Statistiques  [P] Panneau"
            echo ""
            $PYTHON src/main.py --source 0
            ;;

        2)
            echo ""
            echo -e "${CYAN}============================================================${NC}"
            echo -e "${CYAN}  CAMÉRA TÉLÉPHONE VIA USB${NC}"
            echo -e "${CYAN}============================================================${NC}"
            echo ""
            echo "  Pour Android :"
            echo "    1. Installer DroidCam sur le téléphone + DroidCam Client sur Mac"
            echo "    2. Activer le débogage USB sur le téléphone"
            echo "    3. Brancher en USB → DroidCam crée une webcam virtuelle"
            echo ""
            echo "  Pour iPhone :"
            echo "    1. Continuity Camera (macOS Ventura+) : automatique en USB"
            echo "    2. Ou installer EpocCam sur iPhone + driver Mac"
            echo ""
            read -p "  Index de la caméra USB (1, 2 ou 3) [défaut: 1] : " USB_IDX
            USB_IDX=${USB_IDX:-1}
            echo ""
            echo -e "  ${GREEN}[LANCEMENT]${NC} Caméra USB (index $USB_IDX)"
            echo ""
            $PYTHON src/main.py --source $USB_IDX
            ;;

        3)
            echo ""
            echo -e "${CYAN}============================================================${NC}"
            echo -e "${CYAN}  CAMÉRA TÉLÉPHONE VIA WI-FI${NC}"
            echo -e "${CYAN}============================================================${NC}"
            echo ""
            echo "  1. Installer 'IP Webcam' (Android) ou 'DroidCam' sur le téléphone"
            echo "  2. Ouvrir l'app et démarrer le serveur"
            echo "  3. Mac et téléphone sur le même Wi-Fi"
            echo ""
            echo "  Exemples d'URL :"
            echo "    IP Webcam  : http://192.168.1.42:8080/video"
            echo "    DroidCam   : http://192.168.1.42:4747/video"
            echo ""
            read -p "  URL du flux : " URL_TEL
            echo ""
            echo -e "  ${GREEN}[LANCEMENT]${NC} Caméra Wi-Fi"
            echo ""
            $PYTHON src/main.py --source "$URL_TEL"
            ;;

        4)
            echo ""
            read -p "  Chemin du fichier vidéo : " FICHIER
            echo ""
            echo -e "  ${GREEN}[LANCEMENT]${NC} Fichier vidéo"
            echo ""
            $PYTHON src/main.py --source "$FICHIER"
            ;;

        5)
            echo ""
            echo -e "${CYAN}============================================================${NC}"
            echo -e "${CYAN}  CAPTURE DE VISAGES — Whitelist${NC}"
            echo -e "${CYAN}============================================================${NC}"
            echo ""
            echo "    [A] Capturer avec la webcam du Mac"
            echo "    [B] Capturer avec caméra téléphone (USB index)"
            echo "    [C] Capturer avec caméra téléphone (Wi-Fi URL)"
            echo "    [D] Construire la whitelist (sans caméra)"
            echo ""
            read -p "  Votre choix (A/B/C/D) : " WL_CHOIX

            case ${WL_CHOIX,,} in
                a)
                    $PYTHON tools/whitelist_capture.py --source 0
                    ;;
                b)
                    read -p "  Index USB (1, 2 ou 3) : " WL_USB
                    $PYTHON tools/whitelist_capture.py --source $WL_USB
                    ;;
                c)
                    read -p "  URL du flux : " WL_URL
                    $PYTHON tools/whitelist_capture.py --source "$WL_URL"
                    ;;
                d)
                    $PYTHON tools/whitelist_capture.py --build
                    ;;
            esac
            ;;

        6)
            echo ""
            echo -e "  ${GREEN}[BENCHMARK]${NC} Test de performance..."
            echo ""
            $PYTHON tests/benchmark.py
            echo ""
            read -p "  Appuyez sur Entrée pour continuer..."
            ;;

        7)
            echo ""
            echo "  Au revoir !"
            exit 0
            ;;

        *)
            echo -e "  ${RED}Choix invalide, réessayez.${NC}"
            ;;
    esac
done
