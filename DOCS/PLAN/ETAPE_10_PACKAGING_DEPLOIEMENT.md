# 🚢 ÉTAPE 10 — Packaging, Déploiement et Livraison

> **Durée estimée : 2-3 jours**
> **Priorité : HAUTE**
> **Prérequis : ÉTAPES 1-9 complétées (8 minimum)**

---

## 🎯 Objectif

Préparer Sentinel-AI pour le déploiement en production et la distribution. Cette étape finale couvre le packaging, la containerisation Docker, la documentation utilisateur, et la préparation de la première release.

---

## 📋 Tâches détaillées

### 10.1 — Dockerisation

Créer un Dockerfile optimisé :

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Dépendances système (OpenCV, audio)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Port du dashboard
EXPOSE 5000

# Point d'entrée
CMD ["python", "main.py"]
```

**Docker Compose :**
```yaml
# docker-compose.yml
version: '3.8'
services:
  sentinel:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    devices:
      - /dev/video0:/dev/video0     # Webcam
    env_file: .env
    restart: unless-stopped
    
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    restart: unless-stopped

volumes:
  ollama_data:
```

### 10.2 — Script d'installation automatisé

Créer `scripts/install.py` (ou `install.sh`) :

```bash
#!/bin/bash
# install.sh — Installation automatique de Sentinel-AI

echo "🛡️ Installation de Sentinel-AI..."

# 1. Vérifier Python
python3 --version || { echo "❌ Python 3.10+ requis"; exit 1; }

# 2. Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Installer PyTorch (GPU detection)
if nvidia-smi &>/dev/null; then
    echo "✅ GPU détecté, installation CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "⚠️ Pas de GPU, installation CPU..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 5. Télécharger les modèles
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

# 6. Créer les dossiers data
mkdir -p data/{whitelist,snapshots,clips,reports}

# 7. Copier .env
cp .env.example .env

echo "✅ Installation terminée !"
echo "🚀 Lancez: python main.py"
```

### 10.3 — Documentation utilisateur

Créer un guide utilisateur complet :

**`DOCS/USER_GUIDE.md` :**
1. **Installation** — Pas à pas avec captures d'écran.
2. **Premier démarrage** — Configuration initiale.
3. **Enrôlement** — Comment ajouter des personnes à la whitelist.
4. **Dashboard** — Description de chaque page et fonctionnalité.
5. **Configuration** — Tous les paramètres expliqués.
6. **Dépannage** — Problèmes courants et solutions.
7. **FAQ** — Questions fréquentes.

### 10.4 — Préparation de la Release

#### Versioning Sémantique
```
v1.0.0 — MVP fonctionnel (Étapes 1-8)
v1.1.0 — Fonctionnalités avancées (Étape 9)
v1.2.0 — Améliorations et corrections
```

#### Fichiers de release
- `CHANGELOG.md` — Historique des changements.
- `CONTRIBUTING.md` — Guide de contribution.
- `SECURITY.md` — Politique de sécurité.

#### Checklist de release

- [ ] Tous les tests passent (`pytest tests/ -v`).
- [ ] Couverture ≥ 80%.
- [ ] Docker build réussit.
- [ ] Docker compose démarre correctement.
- [ ] README à jour avec badges.
- [ ] Documentation utilisateur complète.
- [ ] `.env.example` à jour.
- [ ] `requirements.txt` avec versions pinnées.
- [ ] Pas de secrets dans le code.
- [ ] License BSD-3 en place.
- [ ] CHANGELOG rédigé.
- [ ] Tag Git créé (`git tag v1.0.0`).

### 10.5 — Démo vidéo

Préparer une démonstration vidéo de 3-5 minutes montrant :
1. Installation et démarrage.
2. Dashboard en action (live view).
3. Reconnaissance d'une personne connue.
4. Interaction avec un inconnu (dialogue TTS/STT).
5. Réception d'une alerte email.
6. Configuration à la volée (changement de LLM).

---

## 🧪 Tests de validation (Étape 10)

| # | Test | Résultat attendu |
|---|------|-------------------|
| 1 | Docker build | Image construite sans erreur |
| 2 | Docker compose up | Sentinel + Ollama démarrent |
| 3 | Install script | Installation complète en < 10 min |
| 4 | Fresh start | Nouveau user peut démarrer en suivant le README |
| 5 | Release checklist | Tous les items cochés |

---

## 📦 Livrables de l'étape

- [ ] `Dockerfile` optimisé
- [ ] `docker-compose.yml` complet
- [ ] `scripts/install.sh` (Linux/Mac) et `scripts/install.ps1` (Windows)
- [ ] `DOCS/USER_GUIDE.md` — Guide utilisateur
- [ ] `CHANGELOG.md`
- [ ] `CONTRIBUTING.md`
- [ ] Tag Git `v1.0.0`
- [ ] Vidéo de démonstration (optionnel)
- [ ] `RESUME_ETAPE_10.md`

---

## 🎉 Fin du Plan de Développement

À la fin de cette étape, Sentinel-AI est :
- ✅ **Fonctionnel** : Tous les modules opérationnels.
- ✅ **Testé** : 80%+ de couverture, tests E2E validés.
- ✅ **Documenté** : Architecture, API, guide utilisateur.
- ✅ **Déployable** : Docker, scripts d'installation.
- ✅ **Sécurisé** : Auth, secrets, validation d'entrées.
- ✅ **Extensible** : Architecture modulaire, Event Bus, plugins.

> **"Sentinel-AI n'est pas un simple script de surveillance. C'est un agent de sécurité intelligent, autonome, et modulaire. Chaque décision architecturale a été pensée pour la qualité, la maintenabilité, et l'évolutivité."**
