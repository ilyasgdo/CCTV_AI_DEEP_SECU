# 🛡️ CCTV AI DEEP SECU — Système de Vidéosurveillance Intelligente

Système de vidéosurveillance intelligent utilisant l'IA en temps réel : détection de personnes, suivi multi-cibles, reconnaissance faciale, détection d'objets portés, analyse comportementale et alertes automatiques.

**Compatible Windows (NVIDIA CUDA) + macOS (Apple Silicon M1/M2/M3/M4)**

---

## 🚀 Lancement Rapide (1 seule commande)

### 🖥️ Windows (NVIDIA GPU)

**Double-clic** sur `start.bat` — ou en terminal :
```powershell
cd C:\Users\ilyas\Documents\CCTV_AI_DEEP_SECU
.\start.bat
```

### 🍎 macOS (Apple Silicon M1/M2/M3/M4)

```bash
cd /chemin/vers/CCTV_AI_DEEP_SECU
chmod +x start_mac.sh
./start_mac.sh
```

> **C'est tout !** Le script détecte automatiquement le GPU (CUDA ou MPS), crée l'environnement virtuel, installe toutes les dépendances et lance le système.

---

## 📸 Sources Vidéo (2 options)

### Option A — Caméra du PC (Webcam)

C'est l'option par défaut. Le système utilise automatiquement la webcam intégrée :

```powershell
# Lancement automatique avec webcam
.\start.bat

# OU manuellement :
.\venv\Scripts\Activate.ps1
python src/main.py
```

### Option B — Caméra d'un Téléphone (Wi-Fi ou USB)

Transformez votre téléphone Android/iPhone en caméra de surveillance :

#### 📲 Étape 1 — Installer l'App sur le Téléphone

| Plateforme | Application | Wi-Fi | USB | Lien |
|------------|-------------|-------|-----|------|
| **Android** | **DroidCam** | ✅ | ✅ | [Google Play](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam) |
| **Android** | **IP Webcam** | ✅ | ❌ | [Google Play](https://play.google.com/store/apps/details?id=com.pas.webcam) |
| **iPhone** | **EpocCam** | ✅ | ✅ | App Store |
| **iPhone** | **DroidCam** | ✅ | ✅ | App Store |

#### 📶 Méthode A — Via Wi-Fi (sans câble)

1. PC et téléphone sur le **même réseau Wi-Fi**
2. Ouvrir l'app sur le téléphone → **démarrer le serveur**
3. Noter l'adresse affichée (ex: `http://192.168.1.42:8080`)
4. Lancer le système :

```powershell
# IP Webcam (Android)
python src/main.py --source "http://192.168.1.42:8080/video"

# DroidCam (Android/iPhone)
python src/main.py --source "http://192.168.1.42:4747/video"
```

#### � Méthode B — Via USB (plus stable, recommandé)

**Avec DroidCam (Android) :**

1. Installer **DroidCam Client** sur le PC : [droidcam.app](https://www.dev47apps.com/)
2. Activer le **débogage USB** sur le téléphone :
   - `Paramètres → À propos → Appuyer 7x sur "Numéro de build"`
   - `Paramètres → Options développeur → Débogage USB → Activer`
3. Brancher le téléphone au PC via USB
4. Ouvrir **DroidCam Client** sur le PC → Sélectionner **USB** → **Start**
5. DroidCam crée une **webcam virtuelle** (index 1 ou 2) :

```powershell
# DroidCam USB = webcam virtuelle (essayer index 1, 2, ou 3)
python src/main.py --source 1

# Si index 1 ne marche pas, essayer :
python src/main.py --source 2
```

**Avec EpocCam (iPhone) :**

1. Installer **EpocCam** sur iPhone + le driver sur PC
2. Brancher l'iPhone en USB
3. EpocCam apparaît comme webcam virtuelle :

```powershell
python src/main.py --source 1
```

> **💡 Astuce USB :** Pour trouver le bon index de caméra, lancez le menu (`start.bat`) option [1] avec la webcam du PC, puis option [2] avec l'USB branché en essayant les index 1, 2, 3.

---

## 🖥️ Interface à l'Écran

Le système affiche en temps réel :

```
┌──────────────────────────────────────────────────────────┐
│  FPS: 30.2                                               │
│  Personnes: 2 | DB: 2 present(s) | Alertes: 0           │
│                                                          │
│    ┌─────────┐                                           │
│    │  ID:1   │  ← Boîte englobante verte                 │
│    │  Thomas │  ← Nom (reconnu par InsightFace)          │
│    │  marcher│  ← Action (prédite par ST-GCN)            │
│    │  🦴     │  ← Squelette 17 keypoints                 │
│    └─────────┘                                           │
│                                                          │
│    ┌─────────┐                                           │
│    │  ID:2   │                                           │
│    │ INCONNU │  ← Personne non reconnue                  │
│    │ immobile│                                           │
│    └─────────┘                                           │
│                                                          │
│  ░░░░░░░ Zone de Maraudage (orange) ░░░░░░░              │
│                                                          │
│  🚨 ALERTE: MARAUDAGE ID:2 (312s)  ← Si timeout dépassé │
└──────────────────────────────────────────────────────────┘
```

**Informations affichées :**
- **FPS** en haut à gauche (objectif ≥ 25 FPS)
- **Compteurs** : nombre de personnes, présences en BDD, alertes
- **Par personne** : boîte englobante, ID de suivi, nom, action en cours
- **Squelette** : 17 points anatomiques reliés (COCO format)
- **Zones de maraudage** : polygones orange semi-transparents
- **Alertes** : bandeau rouge en cas de chute, coup ou maraudage

**Raccourcis clavier :**

| Touche | Action |
|--------|--------|
| `q` | Quitter le système proprement |
| `s` | Afficher les statistiques détaillées dans la console |

---

## ⚙️ Options de Lancement

```powershell
# Webcam (défaut)
python src/main.py

# Fichier vidéo
python src/main.py --source "C:\chemin\vers\video.mp4"

# Caméra téléphone (IP Webcam)
python src/main.py --source "http://192.168.1.42:8080/video"

# Flux RTSP (caméra IP pro)
python src/main.py --source "rtsp://user:pass@192.168.1.100:554/stream"

# Mode headless (sans affichage, uniquement BDD)
python src/main.py --no-display

# Désactiver la reconnaissance faciale
python src/main.py --no-face

# Désactiver l'analyse ST-GCN
python src/main.py --no-stgcn
```

---

## 📋 Prérequis Système

### Windows
| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| **OS** | Windows 10 | Windows 10/11 |
| **Python** | 3.10 | 3.13 |
| **GPU** | NVIDIA GTX 1060 (6 Go) | RTX 3080 Ti (12 Go) |
| **RAM** | 8 Go | 16 Go |
| **CUDA** | 11.8 | 12.1 |

### macOS
| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| **OS** | macOS 13 Ventura | macOS 14 Sonoma+ |
| **Python** | 3.10 | 3.12 |
| **Puce** | Apple M1 (8 Go) | Apple M2 Pro+ (16 Go) |
| **RAM** | 8 Go | 16 Go |

> **⚠️ Le système fonctionne aussi sur CPU**, mais les performances seront très réduites (~5 FPS au lieu de 30+).

---

## 🏗️ Architecture du Système

```
┌── Thread 1 ──┐    Queue    ┌── Thread 2 ──┐              ┌── Analyzer ───┐
│              │             │              │              │               │
│  OpenCV      │──[frames]──▸│  YOLOv8-Pose │──[detects]──▸│  ST-GCN       │
│  Capture     │             │  + ByteTrack │              │  InsightFace  │
│  (async)     │             │              │              │  (lazy scan)  │
└──────────────┘             └──────────────┘              └───────┬───────┘
                                                                  │
                                                                  ▼
                                                          ┌───────────────┐
                                                          │   SQLite DB   │
                                                          │   Historique  │
                                                          │   + Alertes   │
                                                          └───────────────┘
```

### Modules Principaux

| Module | Fichier | Rôle |
|--------|---------|------|
| **Détection** | `src/pipeline/detector.py` | YOLOv8-Pose : détection de personnes + extraction des 17 keypoints COCO |
| **Suivi** | ByteTrack intégré | Suivi multi-cibles avec IDs persistants entre frames |
| **Capture** | `src/pipeline/capture.py` | Thread de capture vidéo asynchrone (faible latence) |
| **Reconnaissance** | `src/face_recognition/` | InsightFace : encodage facial + comparaison avec liste blanche |
| **Analyse** | `src/behavior/` | ST-GCN : classification d'actions (marcher, chute, coup...) |
| **Maraudage** | `src/behavior/loitering_detector.py` | Détection de stationnement prolongé dans une zone |
| **Base de Données** | `src/database/db_manager.py` | SQLite : historique des présences + alertes |
| **Analyseur** | `src/pipeline/analyzer.py` | Orchestre ST-GCN + InsightFace + maraudage + BDD |
| **Affichage** | `src/utils/drawing.py` | Visualisation : boîtes, squelettes, labels, alertes |
| **Config** | `src/config.py` | Configuration centralisée (seuils, chemins, paramètres) |

---

## 🧠 Technologies & IA

### 1. Détection de Personnes — YOLOv8-Pose
- **Modèle** : `yolov8m-pose.pt` (Medium, 26M paramètres)
- **Sortie** : Boîte englobante + 17 keypoints COCO par personne
- **Performance** : ~25ms/frame sur RTX 3080 Ti

### 2. Suivi Multi-Cibles — ByteTrack
- **Algorithme** : Association par IoU + Kalman Filter
- **Avantage** : IDs stables même en cas d'occultation temporaire

### 3. Reconnaissance Faciale — InsightFace
- **Détection** : SCRFD (détecteur de visages haute précision)
- **Encodage** : ArcFace (embedding 512D)
- **Stratégie** : Scan paresseux (1 scan / 60 frames pour les inconnus)
- **Whitelist** : Photos de référence dans `data/whitelist_photos/`

### 4. Analyse Comportementale — Analyse Géométrique des Poses
- **Méthode** : Règles géométriques sur les 17 keypoints COCO
- **Entrée** : Buffer de 15 frames de positions
- **Sortie** : 9 actions classifiées + alertes automatiques

**Actions détectées :**
| Action | Description |
|--------|-------------|
| `immobile` | Personne stationnaire |
| `marcher` | Marche normale |
| `courir` | ⚠️ Course / mouvement rapide (ALERTE) |
| `s'asseoir` | Position assise |
| `se_pencher` | Flexion du corps |
| `chute` | 🚨 Chute détectée (ALERTE) |
| `donner_un_coup` | 🚨 Violence détectée (ALERTE) |
| `mains_en_l_air` | 🚨 Mains en l'air (ALERTE) |

### 5. Détection d'Objets Portés — YOLOv8n
- **Modèle** : `yolov8n.pt` (Nano, ultra-rapide) sur les crops de personnes
- **Classes** : 80 classes COCO (téléphone, sac, couteau, bouteille...)
- **Cadence** : Toutes les 3 frames
- **Alertes** : Objets dangereux (🔪 couteau, ✂️ ciseaux, 🏓 batte)

### 6. Détection de Maraudage
- **Méthode** : Polygone spatial + timer
- **Seuil** : 5 minutes (300s) dans la zone → ALERTE
- **Zones** : Configurables (par défaut : 60% central de l'image)

### 6. Base de Données — SQLite
- **Tables** : `presence_records` (entrées/sorties) + `alerts` (incidents)
- **Fichier** : `src/database/cctv_records.db`

---

## 📁 Structure du Projet

```
CCTV_AI_DEEP_SECU/
├── start.bat                    ← 🚀 LANCER ICI (Windows, double-clic)
├── start_mac.sh                 ← 🍎 LANCER ICI (macOS Apple Silicon)
├── setup_env.bat                ← Installation seule (Windows)
├── requirements.txt             ← Dépendances Python
├── README.md                    ← Ce fichier
│
├── src/
│   ├── main.py                  ← Point d'entrée principal
│   ├── config.py                ← Configuration centralisée
│   │
│   ├── pipeline/
│   │   ├── capture.py           ← Thread capture vidéo async
│   │   ├── detector.py          ← YOLOv8-Pose + ByteTrack
│   │   ├── object_detector.py   ← YOLOv8n détection d'objets portés
│   │   └── analyzer.py          ← Orchestrateur d'analyse
│   │
│   ├── face_recognition/
│   │   ├── encoder.py           ← SCRFD + ArcFace (InsightFace)
│   │   └── matcher.py           ← Comparaison avec whitelist
│   │
│   ├── behavior/
│   │   ├── skeleton_buffer.py   ← Buffer temporel (deque 30 frames)
│   │   ├── action_classifier.py ← Interface ST-GCN
│   │   └── loitering_detector.py← Maraudage par polygone
│   │
│   ├── models/
│   │   └── stgcn/
│   │       └── model.py         ← Architecture ST-GCN (PyTorch)
│   │
│   ├── database/
│   │   └── db_manager.py        ← CRUD SQLite
│   │
│   └── utils/
│       └── drawing.py           ← Visualisation OpenCV
│
├── tests/
│   ├── test_detector.py         ← Test détection + suivi
│   ├── test_face_recognition.py ← Test reconnaissance faciale
│   ├── test_database.py         ← Test base de données
│   ├── test_stgcn.py            ← Test analyse comportementale
│   └── benchmark.py             ← Benchmark performance GPU
│
├── data/
│   ├── videos/                  ← Vidéos de test
│   └── whitelist_photos/        ← Photos pour la reconnaissance
│       ├── Thomas/              ← Dossier par personne
│       │   ├── photo1.jpg
│       │   └── photo2.jpg
│       └── Marie/
│           └── photo1.jpg
│
└── PROMPT/
    └── ETAPES/                  ← Documentation technique détaillée
        ├── overview.md
        ├── etape_0.md → etape_5.md
```

---

## 👤 Reconnaissance Faciale — Ajouter des Personnes

### Méthode 1 — Outil de Capture Intégré (Recommandé) 🎥

L'outil capture les visages directement depuis la caméra :

```powershell
# Via le menu start.bat → Option [4]
.\start.bat

# OU directement :
.\venv\Scripts\Activate.ps1
python tools/whitelist_capture.py
```

**Comment ça marche :**

1. 📷 La caméra s'ouvre avec détection de visages en direct
2. 🟢 Cadrez le visage de la personne (boîte verte = visage détecté)
3. ⌨️ Appuyez sur **`C`** pour capturer → tapez le **nom** → **Entrée**
4. 🔄 Répétez pour 3-5 photos (angles légèrement différents)
5. 🔨 Appuyez sur **`B`** pour construire la whitelist automatiquement
6. ✅ Relancez le système — les personnes seront reconnues !

| Touche | Action |
|--------|--------|
| `C` | Capturer le visage visible |
| `B` | Construire la whitelist (quand terminé) |
| `Q` | Quitter |

> **� Conseils :** Capturez 3-5 photos par personne avec des angles légèrement variés (face, 3/4) et un bon éclairage.

### Méthode 2 — Photos Manuelles

Vous pouvez aussi ajouter des photos manuellement :

```
data/whitelist_photos/
├── thomas_1.jpg        ← Format : nom_numero.jpg
├── thomas_2.jpg
├── marie_1.jpg
└── marie_2.jpg
```

Puis construire la whitelist :
```powershell
python tools/whitelist_capture.py --build
```

---

## 🧪 Tests Individuels

Chaque module peut être testé indépendamment :

```powershell
.\venv\Scripts\Activate.ps1

# Test détection + suivi (webcam)
python tests/test_detector.py

# Test reconnaissance faciale
python tests/test_face_recognition.py

# Test base de données
python tests/test_database.py

# Test analyse comportementale (ST-GCN)
python tests/test_stgcn.py

# Benchmark performance GPU
python tests/benchmark.py
```

---

## ⚡ Performance & Optimisation

### Benchmark sur RTX 3080 Ti

| Métrique | Valeur |
|----------|--------|
| YOLO-Pose | ~20 ms/frame |
| FPS global | 25-35 FPS |
| Mémoire GPU | ~3-4 Go / 12 Go |

### Ajuster les Performances

Modifier `src/config.py` :

```python
# Réduire la résolution (plus rapide)
VIDEO_WIDTH = 1280     # Au lieu de 1920
VIDEO_HEIGHT = 720     # Au lieu de 1080

# Réduire la fréquence d'analyse ST-GCN
STGCN_INFERENCE_INTERVAL = 10  # Au lieu de 5

# Réduire la fréquence de scan facial
FACE_RECOGNITION_INTERVAL = 120  # Au lieu de 60
```

---

## 🗄️ Base de Données

Le système enregistre automatiquement tout dans `src/database/cctv_records.db`.

### Tables

**`presence_records`** — Historique des présences :
| Champ | Description |
|-------|-------------|
| `track_id` | ID de suivi |
| `name` | Nom (ou INCONNU) |
| `entry_time` | Date/heure d'entrée |
| `exit_time` | Date/heure de sortie |
| `duration_s` | Durée de présence (secondes) |
| `status` | `PRESENT` ou `SORTI` |
| `alert_flag` | 1 si alerte déclenchée |

**`alerts`** — Journal des alertes :
| Champ | Description |
|-------|-------------|
| `alert_type` | Type (chute, donner_un_coup, MARAUDAGE) |
| `confidence` | Score de confiance |
| `name` | Nom de la personne |
| `timestamp` | Date/heure de l'alerte |

### Consulter les données
```powershell
.\venv\Scripts\Activate.ps1
python -c "
from src.database.db_manager import DatabaseManager
db = DatabaseManager()
print('Présences:', db.get_history(limit=10))
print('Alertes:', db.get_alerts(limit=10))
print('Stats:', db.get_stats())
db.close()
"
```

---

## 🔧 Dépannage

| Problème | Solution |
|----------|----------|
| `CUDA not available` | Installer les pilotes NVIDIA + CUDA Toolkit 12.1 |
| `No module named 'torch'` | Lancer `start.bat` ou `setup_env.bat` |
| Webcam non détectée | Vérifier les permissions caméra dans Paramètres Windows |
| FPS très bas (<10) | Réduire la résolution à 720p dans `config.py` |
| `InsightFace error` | Normal si pas de photos dans `data/whitelist_photos/` |
| Flux téléphone ne se connecte pas | Vérifier que PC + téléphone sont sur le même Wi-Fi |
| `ONNX Runtime error` | Exécuter : `pip install onnxruntime` |

---

## 📜 Licence

MIT — Voir [LICENSE](LICENSE)

---

## 👨‍💻 Auteur

**Ilyas** — Projet CCTV AI Deep Security

---

*Développé avec PyTorch, YOLOv8, InsightFace, et beaucoup de ☕*