# 🏗️ Étape 0 — La Fondation (L'environnement enrichi)

## 📋 Summary (À lire AVANT de commencer)

**Objectif** : Préparer un environnement de développement complet et stable sur Windows avec GPU NVIDIA RTX 3080 Ti, capable de faire tourner PyTorch, YOLOv8-Pose, InsightFace et ST-GCN.

**Durée estimée** : 1-2 heures

**Prérequis matériel** :
- PC Windows 10/11
- GPU NVIDIA RTX 3080 Ti (12 Go VRAM)
- Minimum 16 Go RAM
- 20 Go d'espace disque libre

**Ce que vous aurez à la fin** :
- ✅ Pilotes NVIDIA à jour
- ✅ CUDA Toolkit installé et vérifié
- ✅ cuDNN installé
- ✅ Environnement virtuel Python fonctionnel
- ✅ Toutes les bibliothèques installées et vérifiées avec support GPU
- ✅ Structure de dossiers du projet créée

---

## 📝 Étapes Détaillées

### 0.1 — Vérifier et Mettre à Jour les Pilotes NVIDIA

> [!IMPORTANT]
> Les pilotes NVIDIA doivent être à jour AVANT d'installer CUDA. Un pilote obsolète causera des erreurs silencieuses.

**Actions :**

1. Ouvrir un terminal PowerShell et exécuter :
   ```powershell
   nvidia-smi
   ```

2. **Vérifier la sortie** :
   - `Driver Version` doit être ≥ **535.xx** (pour CUDA 12.1) ou ≥ **520.xx** (pour CUDA 11.8)
   - `CUDA Version` affichée en haut à droite (c'est la version maximale supportée par le driver)

3. Si le driver est trop ancien :
   - Télécharger le dernier driver depuis [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx)
   - Sélectionner : GeForce > RTX 3080 Ti > Windows 10/11 64-bit
   - Installer et **redémarrer** le PC

**✅ Critère de validation 0.1** :
```powershell
nvidia-smi
# DOIT afficher :
# - "NVIDIA GeForce RTX 3080 Ti" (ou similaire)
# - Driver Version: 5xx.xx ou supérieur
# - CUDA Version: 11.8 ou 12.1
# - Aucune erreur
```

---

### 0.2 — Installer CUDA Toolkit

> [!WARNING]
> Ne PAS installer la dernière version de CUDA si PyTorch ne la supporte pas encore. Vérifier la compatibilité sur [pytorch.org](https://pytorch.org/get-started/locally/).

**Actions :**

1. **Choix de la version CUDA** :
   - **Recommandé : CUDA 12.1** (support PyTorch 2.x le plus récent)
   - Alternative : CUDA 11.8 (plus stable, plus testé)

2. Télécharger depuis : [developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

3. **Installation** :
   - Choisir "Custom Installation"
   - Cocher **uniquement** : CUDA > Runtime, Development, Documentation
   - Décocher : Driver components (déjà installé), Visual Studio Integration (non nécessaire)

4. Vérifier l'installation :
   ```powershell
   nvcc --version
   ```

**✅ Critère de validation 0.2** :
```powershell
nvcc --version
# DOIT afficher : "Cuda compilation tools, release 12.1" (ou 11.8)
# ET :
where nvcc
# DOIT retourner un chemin valide (ex: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe)
```

---

### 0.3 — Installer cuDNN

**Actions :**

1. Télécharger cuDNN depuis : [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) (nécessite un compte NVIDIA Developer gratuit)

2. Choisir la version compatible avec votre CUDA :
cda es t 13.1
   - CUDA 12.1 → cuDNN 8.9.x
   - CUDA 11.8 → cuDNN 8.7.x ou 8.9.x

3. **Installation manuelle** :
   ```
   Extraire le ZIP cuDNN et copier les fichiers dans le dossier CUDA :
   
   cudnn-xxx/bin/cudnn*.dll      → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\
   cudnn-xxx/include/cudnn*.h    → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include\
   cudnn-xxx/lib/x64/cudnn*.lib  → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64\
   ```

4. Vérifier que les variables d'environnement PATH contiennent :
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\libnvvp
   ```

**✅ Critère de validation 0.3** :
```powershell
where cudnn64*.dll
# OU
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\cudnn64*.dll"
# DOIT trouver le fichier cudnn64_8.dll (ou similaire)
```

---

### 0.4 — Créer l'Environnement Virtuel Python

> [!TIP]
> On utilise `venv` pour la simplicité. Conda est aussi valable mais plus lourd.

**Actions :**

1. Vérifier la version Python (3.10 ou 3.11 recommandé) :
   ```powershell
   python --version
   # Doit être 3.10.x ou 3.11.x
   ```

2. Créer l'environnement virtuel dans le dossier du projet :
   ```powershell
   cd C:\Users\ilyas\Documents\CCTV_AI_DEEP_SECU
   python -m venv venv
   ```

3. Activer l'environnement :
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   > Si erreur "Execution policy", exécuter d'abord :
   > ```powershell
   > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   > ```

4. Mettre à jour pip :
   ```powershell
   python -m pip install --upgrade pip setuptools wheel
   ```

**✅ Critère de validation 0.4** :
```powershell
# Le prompt doit afficher (venv) au début :
# (venv) PS C:\Users\ilyas\Documents\CCTV_AI_DEEP_SECU>

python --version
# 3.10.x ou 3.11.x

pip --version
# pip 2x.x.x from ...\venv\...
```

---

### 0.5 — Installer les Bibliothèques (Ordre Critique)

> [!CAUTION]
> L'ordre d'installation est **CRITIQUE**. Installer PyTorch en premier garantit que les autres bibliothèques détectent correctement le GPU. Ne PAS changer l'ordre.

**Actions (dans cet ordre strict) :**

**Étape 1 — PyTorch avec CUDA** :
```powershell
# Pour CUDA 12.1 :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# OU pour CUDA 11.8 :
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Étape 2 — Ultralytics (YOLO)** :
```powershell
pip install ultralytics
```

**Étape 3 — InsightFace + ONNX Runtime GPU** :
```powershell
pip install insightface onnxruntime-gpu
```

**Étape 4 — OpenCV et Numpy** :
```powershell
pip install opencv-python numpy
```

**Étape 5 — Utilitaires supplémentaires** :
```powershell
pip install scipy matplotlib tqdm
```

**✅ Critère de validation 0.5** :
```python
# Exécuter ce script Python de vérification :
python -c "
import torch
print('=== VERIFICATION ENVIRONNEMENT ===')
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA disponible : {torch.cuda.is_available()}')
print(f'GPU detecte     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"AUCUN\"}')
print(f'CUDA version    : {torch.version.cuda}')

import ultralytics
print(f'Ultralytics     : {ultralytics.__version__}')

import insightface
print(f'InsightFace     : {insightface.__version__}')

import cv2
print(f'OpenCV          : {cv2.__version__}')

import numpy
print(f'NumPy           : {numpy.__version__}')

import onnxruntime as ort
providers = ort.get_available_providers()
print(f'ONNX Providers  : {providers}')
print()
if torch.cuda.is_available() and 'CUDAExecutionProvider' in providers:
    print('✅ TOUT EST OK — Environnement prêt !')
else:
    print('❌ PROBLÈME DÉTECTÉ — Vérifier CUDA et les installations')
"
```

**Résultat attendu** :
```
=== VERIFICATION ENVIRONNEMENT ===
PyTorch version : 2.x.x+cu121
CUDA disponible : True
GPU detecte     : NVIDIA GeForce RTX 3080 Ti
CUDA version    : 12.1
Ultralytics     : 8.x.x
InsightFace     : 0.7.x
OpenCV          : 4.x.x
NumPy           : 1.x.x
ONNX Providers  : ['CUDAExecutionProvider', 'CPUExecutionProvider']

✅ TOUT EST OK — Environnement prêt !
```

---

### 0.6 — Créer la Structure de Dossiers du Projet

**Actions :**

```powershell
cd C:\Users\ilyas\Documents\CCTV_AI_DEEP_SECU

# Créer toute l'arborescence
mkdir src\pipeline
mkdir src\models\stgcn\weights
mkdir src\models\yolo\weights
mkdir src\face_recognition\whitelist
mkdir src\database
mkdir src\behavior
mkdir src\utils
mkdir data\videos
mkdir data\whitelist_photos
mkdir tests
```

**Créer les fichiers `__init__.py`** (pour que Python reconnaisse les packages) :
```powershell
# Fichiers __init__.py vides
New-Item -ItemType File -Path src\__init__.py -Force
New-Item -ItemType File -Path src\pipeline\__init__.py -Force
New-Item -ItemType File -Path src\models\__init__.py -Force
New-Item -ItemType File -Path src\models\stgcn\__init__.py -Force
New-Item -ItemType File -Path src\models\yolo\__init__.py -Force
New-Item -ItemType File -Path src\face_recognition\__init__.py -Force
New-Item -ItemType File -Path src\database\__init__.py -Force
New-Item -ItemType File -Path src\behavior\__init__.py -Force
New-Item -ItemType File -Path src\utils\__init__.py -Force
New-Item -ItemType File -Path tests\__init__.py -Force
```

**✅ Critère de validation 0.6** :
```powershell
# Vérifier que l'arborescence existe :
tree /F src
# Doit afficher la structure complète avec tous les sous-dossiers et __init__.py
```

---

### 0.7 — Créer le fichier `requirements.txt`

**Actions :**

Créer le fichier `requirements.txt` à la racine du projet :

```txt
# GPU / CUDA (installer PyTorch séparément avec --index-url)
# torch
# torchvision
# torchaudio

# Detection + Pose Estimation
ultralytics>=8.0.0

# Reconnaissance Faciale
insightface>=0.7.0
onnxruntime-gpu>=1.15.0

# Vision et Calcul
opencv-python>=4.8.0
numpy>=1.24.0

# Utilitaires
scipy>=1.10.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

**✅ Critère de validation 0.7** :
```powershell
type requirements.txt
# Doit afficher le contenu du fichier ci-dessus
```

---

### 0.8 — Créer le fichier de configuration centralisé `config.py`

**Actions :**

Créer `src/config.py` avec toutes les constantes du projet :

```python
"""
CCTV AI DEEP SECU — Configuration Centralisée
Toutes les constantes et paramètres du système sont ici.
"""
import os
from pathlib import Path

# === CHEMINS ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
WHITELIST_DIR = DATA_DIR / "whitelist_photos"
DB_PATH = PROJECT_ROOT / "src" / "database" / "cctv_records.db"
STGCN_WEIGHTS = PROJECT_ROOT / "src" / "models" / "stgcn" / "weights"
YOLO_WEIGHTS = PROJECT_ROOT / "src" / "models" / "yolo" / "weights"

# === YOLO-POSE ===
YOLO_MODEL = "yolov8m-pose.pt"  # Medium. Changer en "yolov8l-pose.pt" pour Large
YOLO_CONFIDENCE = 0.5            # Seuil de confiance minimum
YOLO_DEVICE = 0                  # 0 = premier GPU, "cpu" pour CPU

# === BYTETRACK (Suivi) ===
TRACKER_TYPE = "bytetrack"        # Tracker intégré à Ultralytics

# === INSIGHTFACE ===
FACE_RECOGNITION_THRESHOLD = 0.55  # Seuil de similarité cosinus
FACE_RECOGNITION_INTERVAL = 60     # Frames entre chaque scan pour un INCONNU (≈2s à 30fps)
FACE_CONFIDENCE_LOCK = 0.95        # Au-dessus, on arrête de scanner

# === ST-GCN ===
STGCN_BUFFER_SIZE = 30            # Nombre de frames dans le buffer temporel
STGCN_INFERENCE_INTERVAL = 5     # Inférence toutes les N frames
STGCN_NUM_KEYPOINTS = 17          # Keypoints COCO (sortie YOLOv8-Pose)
STGCN_IN_CHANNELS = 2             # X, Y (ou 3 si on ajoute la confiance)

# === ACTIONS RECONNUES ===
ACTION_LABELS = [
    "marcher",
    "courir",
    "s'asseoir",
    "se_lever",
    "chute",
    "donner_un_coup",
    "immobile",
    "se_pencher",
]

# === ALERTES ===
ALERT_ACTIONS = ["chute", "donner_un_coup"]  # Actions déclenchant une alerte
LOITERING_TIMEOUT = 300            # Secondes avant alerte de maraudage (5 min)
PERSON_LOST_TIMEOUT = 300          # Secondes avant de considérer la personne partie

# === PERFORMANCE ===
TARGET_FPS = 30
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
```

**✅ Critère de validation 0.8** :
```powershell
python -c "from src.config import *; print('✅ config.py importé avec succès'); print(f'  YOLO model: {YOLO_MODEL}'); print(f'  ST-GCN buffer: {STGCN_BUFFER_SIZE} frames')"
```

---

## ✅ Checklist de Validation Finale — Étape 0

Avant de passer à l'**Étape 1**, TOUS ces critères doivent être vérifiés :

| # | Critère | Commande de vérification | Status |
|---|---------|--------------------------|--------|
| 0.1 | Pilotes NVIDIA à jour | `nvidia-smi` → driver ≥ 535 | ⬜ |
| 0.2 | CUDA Toolkit installé | `nvcc --version` → 11.8 ou 12.1 | ⬜ |
| 0.3 | cuDNN installé | `where cudnn64*.dll` → fichier trouvé | ⬜ |
| 0.4 | Environnement virtuel actif | `(venv)` visible dans le prompt | ⬜ |
| 0.5 | Toutes les libs installées + GPU | Script de vérification → ✅ | ⬜ |
| 0.6 | Structure de dossiers créée | `tree /F src` → arborescence complète | ⬜ |
| 0.7 | `requirements.txt` créé | `type requirements.txt` | ⬜ |
| 0.8 | `config.py` importable | Import Python réussi | ⬜ |

> [!CAUTION]
> **NE PASSEZ PAS À L'ÉTAPE 1 si un seul critère est ⬜.** Résolvez chaque problème avant de continuer.

---

**➡️ Étape suivante : [etape_1.md](etape_1.md) — Détection, Suivi et Extraction Squelettique**
