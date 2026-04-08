# 🛡️ SENTINEL-AI — Plan de Développement Complet

> **Projet : CCTV AI Deep Security — "Sentinel-AI"**
> **Auteur : Ilyas Ghandaoui**
> **Date de création : 08 Avril 2026**
> **Licence : BSD 3-Clause**

---

## 📖 Vue d'ensemble

Sentinel-AI est un agent d'Intelligence Artificielle autonome qui agit comme un gardien de sécurité physique et interactif. Il surveille un flux vidéo en temps réel, identifie les personnes, analyse le contexte via un LLM (Gemma 4), et interagit vocalement.

**Architecture clé :**
- 💻 **CPU (Edge)** : Perception temps réel (caméra, YOLO, Whisper, TTS)
- 🎮 **GPU (Cognitif)** : Analyse sémantique via Ollama (local ou distant)
- 🖥️ **Web** : Dashboard Flask pour le monitoring et la configuration

---

## 📋 Index des étapes

| # | Étape | Fichier | Durée | Priorité |
|---|-------|---------|-------|----------|
| 📜 | **Règles de développement** | [00_REGLES_DEVELOPPEMENT.md](00_REGLES_DEVELOPPEMENT.md) | — | OBLIGATOIRE |
| 1 | **Initialisation Environnement** | [ETAPE_01_INITIALISATION_ENVIRONNEMENT.md](ETAPE_01_INITIALISATION_ENVIRONNEMENT.md) | 1-2j | CRITIQUE |
| 2 | **Perception Visuelle (CPU)** | [ETAPE_02_PERCEPTION_VISUELLE.md](ETAPE_02_PERCEPTION_VISUELLE.md) | 3-4j | CRITIQUE |
| 3 | **Reconnaissance Faciale** | [ETAPE_03_RECONNAISSANCE_FACIALE.md](ETAPE_03_RECONNAISSANCE_FACIALE.md) | 3-4j | HAUTE |
| 4 | **Pipeline Cognitif (LLM)** | [ETAPE_04_PIPELINE_COGNITIF.md](ETAPE_04_PIPELINE_COGNITIF.md) | 4-5j | CRITIQUE |
| 5 | **Audio & Outils** | [ETAPE_05_AUDIO_ET_OUTILS.md](ETAPE_05_AUDIO_ET_OUTILS.md) | 3-4j | HAUTE |
| 6 | **Dashboard Web** | [ETAPE_06_DASHBOARD_WEB.md](ETAPE_06_DASHBOARD_WEB.md) | 5-6j | HAUTE |
| 7 | **Intégration Complète** | [ETAPE_07_INTEGRATION_COMPLETE.md](ETAPE_07_INTEGRATION_COMPLETE.md) | 4-5j | CRITIQUE |
| 8 | **Tests & Optimisation** | [ETAPE_08_TESTS_OPTIMISATION.md](ETAPE_08_TESTS_OPTIMISATION.md) | 3-4j | HAUTE |
| 9 | **Fonctionnalités Avancées** | [ETAPE_09_FONCTIONNALITES_AVANCEES.md](ETAPE_09_FONCTIONNALITES_AVANCEES.md) | 5-7j | MOYENNE |
| 10 | **Packaging & Déploiement** | [ETAPE_10_PACKAGING_DEPLOIEMENT.md](ETAPE_10_PACKAGING_DEPLOIEMENT.md) | 2-3j | HAUTE |

**Durée totale estimée : 30-42 jours de développement**

---

## 🗺️ Diagramme de dépendances

```
ÉTAPE 1 (Init)
    │
    ▼
ÉTAPE 2 (Perception) ──────────────────┐
    │                                    │
    ▼                                    │
ÉTAPE 3 (Reconnaissance Faciale)        │
    │                                    │
    ▼                                    │
ÉTAPE 4 (Pipeline Cognitif LLM)        │
    │                                    │
    ▼                                    │
ÉTAPE 5 (Audio + Outils) ──────────────┤
    │                                    │
    ▼                                    │
ÉTAPE 6 (Dashboard Web) ◄──────────────┘
    │
    ▼
ÉTAPE 7 (Intégration Complète)
    │
    ▼
ÉTAPE 8 (Tests & Optimisation)
    │
    ▼
ÉTAPE 9 (Fonctionnalités Avancées)  ← Optionnel pour MVP
    │
    ▼
ÉTAPE 10 (Packaging & Déploiement)
```

---

## 🏗️ Architecture du Projet

```
CCTV_AI_DEEP_SECU/
├── 📁 src/
│   ├── 📁 core/          ← Perception CPU (Étapes 1-3)
│   ├── 📁 cognitive/     ← Intelligence LLM (Étape 4)
│   ├── 📁 audio/         ← TTS + STT (Étape 5)
│   ├── 📁 effector/      ← Outils / Actions (Étape 5)
│   ├── 📁 dashboard/     ← Interface Web (Étape 6)
│   └── 📁 utils/         ← Utilitaires (Étape 1)
├── 📁 config/            ← Configuration YAML + Prompts
├── 📁 data/              ← Whitelist, Snapshots, Clips
├── 📁 tests/             ← Tests unitaires + intégration
├── 📁 scripts/           ← Scripts d'installation et utilitaires
├── 📁 DOCS/PLAN/         ← CE PLAN
├── 📄 main.py            ← Point d'entrée
└── 📄 requirements.txt   ← Dépendances Python
```

---

## ⚡ Démarrage rapide (pour les développeurs)

```bash
# 1. Lire les règles (OBLIGATOIRE)
cat DOCS/PLAN/00_REGLES_DEVELOPPEMENT.md

# 2. Lire l'étape en cours
cat DOCS/PLAN/ETAPE_01_INITIALISATION_ENVIRONNEMENT.md

# 3. Lire le résumé de l'étape précédente (si applicable)
cat DOCS/PLAN/RESUMES/RESUME_ETAPE_XX.md

# 4. Développer, tester, résumer

# 5. Passer à l'étape suivante
```

---

## 📌 Règle d'or

> **"Lis. Comprends. Développe. Teste. Résume. Passe à la suite."**
> 
> — Aucun raccourci, aucune improvisation, chaque étape est un bloc autonome et validé.
