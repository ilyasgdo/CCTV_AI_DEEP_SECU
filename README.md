# 🛡️ Sentinel-AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](LICENSE)
[![Status: Release Ready](https://img.shields.io/badge/Status-Release%20Ready-brightgreen.svg)]()

> **Gardien de Sécurité Autonome Multimodal basé sur l'Intelligence Artificielle**

Sentinel-AI est un agent IA autonome qui surveille un flux vidéo en temps réel, identifie les personnes, analyse le contexte de la scène via un modèle de langage (Gemma 4), et interagit vocalement avec les interlocuteurs.

---

## ✨ Fonctionnalités

- 📷 **Surveillance vidéo** multi-source (webcam, RTSP, IP, DroidCam)
- 🧑 **Reconnaissance faciale** avec whitelist (InsightFace)
- 🤖 **Analyse sémantique** toutes les 5s via LLM (Gemma 4 / Ollama)
- 🔊 **Interaction vocale** bidirectionnelle (TTS + STT)
- 🛠️ **Exécution d'outils** (email, alarme, snapshots)
- 🖥️ **Dashboard web** avec live view et gestion

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Dashboard Web (Flask)              │
├──────────────┬──────────────┬────────────────────────┤
│   Pipeline   │   Pipeline   │     Pipeline           │
│   CPU (Edge) │   GPU (LLM)  │     Effecteur          │
│              │              │                        │
│  • Caméra    │  • Ollama    │  • TTS (edge-tts)      │
│  • YOLO      │  • Gemma 4   │  • Email               │
│  • Tracking  │  • Prompts   │  • Alarme              │
│  • InsightFace│ • Parser    │  • Snapshots           │
│  • Whisper   │              │                        │
└──────────────┴──────────────┴────────────────────────┘
```

---

## 🚀 Installation rapide

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-repo/CCTV_AI_DEEP_SECU.git
cd CCTV_AI_DEEP_SECU

# 2. Créer l'environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (Optionnel) PyTorch avec GPU CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Configurer l'environnement
cp .env.example .env
# Éditez .env avec vos vraies valeurs

# 6. Vérifier l'installation
python main.py --check

# 7. Démarrer
python main.py
```

---

## ⚙️ Configuration

La configuration se fait via `config/settings.yaml`. Les paramètres clés :

| Section | Paramètre | Description |
|---------|-----------|-------------|
| `camera.source` | `"0"` | Source vidéo (webcam, URL RTSP, etc.) |
| `llm.api_url` | `"http://localhost:11434"` | URL de l'API Ollama |
| `llm.model_name` | `"gemma4"` | Modèle LLM à utiliser |
| `detection.confidence` | `0.5` | Seuil de confiance YOLO |
| `audio.tts_voice` | `"fr-FR-HenriNeural"` | Voix de synthèse |
| `dashboard.port` | `5000` | Port du dashboard web |

Les secrets (SMTP, clés API) sont gérés dans `.env`.

---

## 📁 Structure du projet

```
CCTV_AI_DEEP_SECU/
├── src/
│   ├── core/          # Perception CPU (caméra, YOLO, tracking, visage)
│   ├── cognitive/     # Intelligence LLM (client, prompts, parser)
│   ├── audio/         # TTS (edge-tts) + STT (Whisper)
│   ├── effector/      # Outils (email, alarme, snapshots)
│   ├── dashboard/     # Interface Web (Flask)
│   └── utils/         # Logger, Event Bus
├── config/            # Configuration YAML + System Prompts
├── data/              # Whitelist, snapshots, clips, rapports
├── tests/             # Tests unitaires et d'intégration
├── scripts/           # Scripts utilitaires
├── DOCS/              # Documentation complète et plan de développement
├── main.py            # Point d'entrée
└── requirements.txt   # Dépendances Python
```

---

## 📖 Documentation

- [Document de Vision (MVP)](DOCS/MVP)
- [Plan de développement](DOCS/PLAN/INDEX.md)
- [Règles de développement](DOCS/PLAN/00_REGLES_DEVELOPPEMENT.md)
- [Guide utilisateur](DOCS/USER_GUIDE.md)
- [Guide de déploiement](DOCS/DEPLOYMENT.md)

---

## 🐳 Docker (Étape 10)

```bash
# Build image
docker build -t sentinel-ai:latest .

# Run stack (Sentinel + Ollama)
docker compose up -d
```

Fichiers associés:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

---

## 📄 Licence

Ce projet est sous licence [BSD 3-Clause](LICENSE).

Copyright (c) 2026, Ilyas Ghandaoui.
