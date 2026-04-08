# 🚀 ÉTAPE 1 — Initialisation de l'Environnement et Fondations

> **Durée estimée : 1-2 jours**
> **Priorité : CRITIQUE — Aucune autre étape ne peut commencer sans celle-ci**
> **Prérequis : Aucun (étape de départ)**

---

## 🎯 Objectif

Mettre en place l'environnement de développement complet, créer l'arborescence du projet, configurer les outils de base, et s'assurer que chaque développeur peut cloner et démarrer le projet en moins de 10 minutes.

---

## 📋 Tâches détaillées

### 1.1 — Création de l'arborescence du projet

Créer la structure de dossiers **exactement** telle que définie dans `00_REGLES_DEVELOPPEMENT.md` :

```bash
mkdir -p src/{core,cognitive,audio,effector,dashboard/{static/{css,js,assets},templates},utils}
mkdir -p data/{whitelist,snapshots,clips,reports}
mkdir -p tests
mkdir -p config/prompts
mkdir -p scripts
mkdir -p DOCS/PLAN/RESUMES
```

**Fichiers `__init__.py`** : Créer un fichier vide `__init__.py` dans chaque sous-dossier de `src/`.

### 1.2 — Configuration Python et dépendances

1. **Vérifier la version Python** : Python 3.10+ requis.
2. **Créer l'environnement virtuel** :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou: venv\Scripts\activate  # Windows
   ```
3. **Mettre à jour `requirements.txt`** avec les versions exactes (pinned) après installation.
4. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
5. **Installer PyTorch avec CUDA** (si GPU disponible) :
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

### 1.3 — Configuration centralisée (`src/core/config.py`)

Créer le module de configuration qui charge les paramètres depuis `config/settings.yaml` :

```python
"""
Module de configuration centralisée pour Sentinel-AI.
Charge les paramètres depuis settings.yaml et les variables d'environnement.
"""
```

**Paramètres à gérer :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `camera.source` | `str` | `"0"` | Source vidéo (0=webcam, URL RTSP, etc.) |
| `camera.width` | `int` | `1280` | Largeur de capture |
| `camera.height` | `int` | `720` | Hauteur de capture |
| `camera.fps` | `int` | `30` | FPS cible |
| `llm.api_url` | `str` | `"http://localhost:11434"` | URL de l'API Ollama |
| `llm.model_name` | `str` | `"gemma4"` | Nom du modèle |
| `llm.timeout` | `int` | `30` | Timeout requête (secondes) |
| `llm.analysis_interval` | `int` | `5` | Intervalle d'analyse (secondes) |
| `detection.model_path` | `str` | `"yolo26n.pt"` | Chemin modèle YOLO |
| `detection.confidence` | `float` | `0.5` | Seuil de confiance |
| `face.similarity_threshold` | `float` | `0.6` | Seuil reconnaissance faciale |
| `audio.tts_enabled` | `bool` | `true` | Activer TTS |
| `audio.stt_enabled` | `bool` | `true` | Activer STT |
| `audio.tts_voice` | `str` | `"fr-FR-HenriNeural"` | Voix TTS |
| `dashboard.host` | `str` | `"0.0.0.0"` | Hôte du dashboard |
| `dashboard.port` | `int` | `5000` | Port du dashboard |
| `alerts.email_enabled` | `bool` | `false` | Activer alertes email |
| `alerts.email_to` | `str` | `""` | Email destinataire |

### 1.4 — Logging centralisé (`src/utils/logger.py`)

Créer le module de logging avec :

- Rotation automatique des fichiers log (max 10MB, 5 fichiers).
- Double sortie : console (colorée) + fichier.
- Format : `[2026-04-08 12:00:00] [core.detector] [INFO] Personne détectée à (320, 240)`.
- Un logger par module obtenu via `get_logger(__name__)`.

### 1.5 — Event Bus (`src/utils/event_bus.py`)

Créer un bus d'événements simple basé sur le pattern **Observer/Pub-Sub** :

```python
# Événements prévus:
# "person_detected"     -> {bbox, confidence, frame_id}
# "face_recognized"     -> {name, confidence, bbox}
# "face_unknown"        -> {bbox, snapshot_path}
# "analysis_ready"      -> {snapshot, context}
# "llm_response"        -> {action_vocale, outils}
# "alert_triggered"     -> {type, severity, details}
# "tool_executed"       -> {tool_name, result}
# "audio_transcribed"   -> {text, confidence}
```

### 1.6 — Fichier de configuration YAML

Créer `config/settings.yaml` avec toutes les valeurs par défaut.

### 1.7 — Fichiers d'environnement

Créer `.env.example` :
```env
# Sentinel-AI Environment Variables
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL_TO=security@example.com
OLLAMA_API_KEY=  # Optionnel
```

### 1.8 — Point d'entrée (`main.py`)

Créer un `main.py` minimaliste qui :
1. Charge la configuration.
2. Initialise le logger.
3. Affiche un message de bienvenue ASCII art "SENTINEL-AI".
4. Vérifie les dépendances critiques (OpenCV, YOLO, etc.).
5. Quitte proprement si une dépendance manque.

### 1.9 — README.md

Créer un README.md professionnel avec :
- Description du projet.
- Badges (Python version, License).
- Instructions d'installation rapide.
- Architecture simplifiée.
- Lien vers la documentation complète.

---

## 🧪 Tests de validation (Étape 1)

| # | Test | Commande | Résultat attendu |
|---|------|----------|-------------------|
| 1 | Structure créée | `find src -name "*.py" \| head -20` | Tous les `__init__.py` existent |
| 2 | Config charge | `python -c "from src.core.config import Config; c=Config(); print(c.camera.source)"` | Affiche `0` |
| 3 | Logger fonctionne | `python -c "from src.utils.logger import get_logger; log=get_logger('test'); log.info('Test OK')"` | Log formaté affiché |
| 4 | Event Bus | `python -c "from src.utils.event_bus import EventBus; eb=EventBus(); print('Bus OK')"` | `Bus OK` |
| 5 | Main démarre | `python main.py --check` | ASCII art + check des dépendances |
| 6 | Tests passent | `pytest tests/ -v` | 0 failures |

---

## 📦 Livrables de l'étape

- [ ] Arborescence complète créée
- [ ] `src/core/config.py` fonctionnel
- [ ] `src/utils/logger.py` fonctionnel
- [ ] `src/utils/event_bus.py` fonctionnel
- [ ] `config/settings.yaml` avec valeurs par défaut
- [ ] `.env.example` créé
- [ ] `main.py` avec checks de base
- [ ] `README.md` professionnel
- [ ] Tests unitaires pour config, logger, event_bus
- [ ] `RESUME_ETAPE_01.md` rédigé

---

## ⚠️ Points d'attention

- **Ne PAS installer de modèle YOLO** à cette étape — on vérifie seulement que la lib `ultralytics` s'importe.
- **Ne PAS configurer Ollama** — juste vérifier que `requests` est disponible.
- **Ne PAS toucher au dashboard** — c'est pour l'étape 6.

---

## 🔗 Dépendances pour l'étape suivante

L'étape 2 nécessite :
- ✅ `src/core/config.py` opérationnel
- ✅ `src/utils/logger.py` opérationnel
- ✅ `src/utils/event_bus.py` opérationnel
- ✅ Environnement virtuel avec toutes les dépendances installées
