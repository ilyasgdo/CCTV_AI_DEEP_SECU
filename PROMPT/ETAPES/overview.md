# 🎯 CCTV AI DEEP SECU — Vue d'Ensemble du Projet

## Résumé Exécutif

Ce projet implémente un système de **vidéosurveillance intelligente** capable de :
- **Détecter** et **suivre** les personnes en temps réel (YOLOv8-Pose)
- **Identifier** les visages via une liste blanche (InsightFace)
- **Archiver** les présences avec horodatage (SQLite)
- **Analyser le comportement** via réseau de neurones spatio-temporel (ST-GCN)
- **Fonctionner en temps réel** à 30 FPS sur RTX 3080 Ti (Pipeline Asynchrone)

---

## Architecture Technique

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE PRINCIPAL                            │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ Thread 1 │───▸│   Thread 2   │───▸│      Thread 3        │   │
│  │ OpenCV   │    │  YOLOv8-Pose │    │  ST-GCN (1x/5 imgs)  │   │
│  │ Capture  │    │  + ByteTrack │    │  + InsightFace lazy   │   │
│  └──────────┘    └──────┬───────┘    └──────────┬────────────┘   │
│                         │                       │                │
│                         ▼                       ▼                │
│                  ┌──────────────┐      ┌─────────────────┐      │
│                  │   SQLite DB  │      │  Alertes Temps   │      │
│                  │  (Présences) │      │  Réel (Chute,    │      │
│                  │              │      │   Maraudage...)  │      │
│                  └──────────────┘      └─────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## Stack Technologique

| Composant | Technologie | Rôle |
|---|---|---|
| Détection + Pose | **YOLOv8-Pose** (Medium/Large) | Boîtes englobantes + 17 keypoints |
| Suivi | **ByteTrack** (intégré à Ultralytics) | ID persistant par personne |
| Reconnaissance faciale | **InsightFace** + ONNX Runtime GPU | Liste blanche / INCONNU |
| Analyse comportementale | **ST-GCN** (PyTorch) | Classification d'actions temporelles |
| Base de données | **SQLite** | Historique des présences |
| GPU Acceleration | **CUDA 11.8/12.1** + cuDNN | RTX 3080 Ti |
| Framework ML | **PyTorch** | Inférence ST-GCN |
| Vision | **OpenCV** | Capture et affichage vidéo |

## Phases d'Implémentation

| Phase | Fichier | Description | Dépendances |
|---|---|---|---|
| **Phase 0** | [etape_0.md](etape_0.md) | Fondation — Environnement & Dépendances | Aucune |
| **Phase 1** | [etape_1.md](etape_1.md) | Détection, Suivi & Squelette (YOLOv8-Pose) | Phase 0 |
| **Phase 2** | [etape_2.md](etape_2.md) | Identification Visuelle (InsightFace) | Phase 1 |
| **Phase 3** | [etape_3.md](etape_3.md) | Historique & Temps (SQLite) | Phase 1 |
| **Phase 4** | [etape_4.md](etape_4.md) | Analyse Comportementale (ST-GCN) | Phase 1 |
| **Phase 5** | [etape_5.md](etape_5.md) | Optimisation RTX 3080 Ti (Pipeline Async) | Phases 0-4 |

## Structure du Projet (Cible Finale)

```
CCTV_AI_DEEP_SECU/
├── PROMPT/
│   ├── plan.txt                    # Plan original
│   └── ETAPES/                     # Guides d'implémentation
│       ├── overview.md
│       ├── etape_0.md ... etape_5.md
├── src/
│   ├── main.py                     # Point d'entrée principal
│   ├── config.py                   # Configuration centralisée
│   ├── pipeline/
│   │   ├── capture.py              # Thread 1 - Capture vidéo
│   │   ├── detector.py             # Thread 2 - YOLOv8-Pose + ByteTrack
│   │   └── analyzer.py             # Thread 3 - ST-GCN + InsightFace
│   ├── models/
│   │   ├── stgcn/                  # Modèle ST-GCN
│   │   │   ├── model.py
│   │   │   └── weights/            # Poids pré-entraînés
│   │   └── yolo/                   # Config YOLO
│   │       └── weights/            # yolov8m-pose.pt
│   ├── face_recognition/
│   │   ├── encoder.py              # Encodage des visages
│   │   ├── matcher.py              # Comparaison des vecteurs
│   │   └── whitelist/              # Photos de référence + .npy
│   ├── database/
│   │   ├── db_manager.py           # CRUD SQLite
│   │   └── cctv_records.db         # Base de données
│   ├── behavior/
│   │   ├── skeleton_buffer.py      # Buffer temporel (deque)
│   │   ├── action_classifier.py    # Interface ST-GCN
│   │   └── loitering_detector.py   # Détection de maraudage (polygone)
│   └── utils/
│       ├── drawing.py              # Affichage / overlays
│       └── alerts.py               # Système d'alertes
├── data/
│   ├── videos/                     # Vidéos de test
│   └── whitelist_photos/           # Photos pour la liste blanche
├── tests/
│   ├── test_detector.py
│   ├── test_face_recognition.py
│   ├── test_stgcn.py
│   └── test_database.py
├── requirements.txt
├── setup_env.bat                   # Script d'installation Windows
└── README.md
```

---

> [!IMPORTANT]
> **Chaque étape doit être validée intégralement avant de passer à la suivante.**
> Les critères de validation sont détaillés dans chaque fichier `etape_X.md`.
