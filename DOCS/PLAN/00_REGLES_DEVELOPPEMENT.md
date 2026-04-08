# 📜 RÈGLES DE DÉVELOPPEMENT — Sentinel-AI

> **Ce fichier est OBLIGATOIRE. Tout développeur travaillant sur ce projet DOIT le lire et le respecter AVANT de toucher au code.**

---

## 🔴 RÈGLE 0 — Lire avant d'agir

**Avant de commencer TOUTE étape, le développeur DOIT :**

1. Relire intégralement le fichier `00_REGLES_DEVELOPPEMENT.md` (ce fichier).
2. Relire le fichier de l'étape qu'il s'apprête à réaliser (ex: `ETAPE_03_...md`).
3. Relire le résumé de l'étape précédente (`RESUME_ETAPE_XX.md`) pour comprendre l'état actuel du projet.
4. Vérifier que toutes les dépendances de l'étape précédente sont satisfaites.

**❌ Il est INTERDIT de commencer une étape sans avoir validé les 4 points ci-dessus.**

---

## 🔴 RÈGLE 1 — Respecter le plan, rien que le plan

- Chaque étape a un scope précis défini dans son fichier `.md`.
- **Il est INTERDIT d'implémenter des fonctionnalités qui ne sont PAS listées dans l'étape en cours.**
- Si une idée d'amélioration surgit en cours de route, elle doit être notée dans un fichier `BACKLOG.md` et traitée ultérieurement.
- Ne jamais "anticiper" une étape future en ajoutant du code prématurément.

---

## 🔴 RÈGLE 2 — Résumé obligatoire après chaque étape

À la fin de chaque étape, le développeur **DOIT** créer un fichier `RESUME_ETAPE_XX.md` dans `DOCS/PLAN/RESUMES/` contenant :

```markdown
# Résumé Étape XX — [Titre]

## ✅ Tâches complétées
- [ liste des tâches réalisées ]

## ⚠️ Problèmes rencontrés
- [ description des bugs, blocages, erreurs ]
- [ solutions appliquées ou contournements ]

## 📁 Fichiers créés / modifiés
- [ liste exhaustive des fichiers touchés ]

## 🧪 Tests effectués
- [ résultats des tests unitaires et d'intégration ]

## 📊 État du projet
- [ ce qui fonctionne ]
- [ ce qui reste à faire pour cette étape ]

## 🔗 Dépendances pour l'étape suivante
- [ pré-requis pour passer à l'étape XX+1 ]
```

**❌ Il est INTERDIT de passer à l'étape suivante sans avoir rédigé ce résumé.**

---

## 🔴 RÈGLE 3 — Architecture et structure des fichiers

### Arborescence obligatoire du projet

```
CCTV_AI_DEEP_SECU/
├── DOCS/
│   ├── MVP                          # Document de vision
│   └── PLAN/
│       ├── 00_REGLES_DEVELOPPEMENT.md
│       ├── ETAPE_01_*.md
│       ├── ETAPE_02_*.md
│       ├── ...
│       └── RESUMES/
│           ├── RESUME_ETAPE_01.md
│           └── ...
├── src/
│   ├── core/                        # Pipeline CPU principal
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration centralisée
│   │   ├── detector.py              # YOLO détection
│   │   ├── face_manager.py          # Reconnaissance faciale + whitelist
│   │   ├── tracker.py               # Suivi d'objets
│   │   └── camera.py                # Gestion caméra (webcam/RTSP/IP)
│   ├── cognitive/                   # Pipeline GPU / LLM
│   │   ├── __init__.py
│   │   ├── llm_client.py            # Client API agnostique (Ollama/etc)
│   │   ├── prompt_manager.py        # Gestion du system prompt
│   │   └── response_parser.py       # Nettoyage <think> + extraction JSON
│   ├── audio/                       # Pipeline Audio
│   │   ├── __init__.py
│   │   ├── tts_engine.py            # Text-to-Speech (edge-tts)
│   │   └── stt_engine.py            # Speech-to-Text (Whisper)
│   ├── effector/                    # Actions / Outils
│   │   ├── __init__.py
│   │   ├── tool_executor.py         # Exécution des tools JSON
│   │   ├── email_tool.py            # Envoi d'emails
│   │   └── alarm_tool.py            # Déclenchement alarme
│   ├── dashboard/                   # Front-End Web
│   │   ├── app.py                   # Serveur Flask
│   │   ├── static/
│   │   │   ├── css/
│   │   │   ├── js/
│   │   │   └── assets/
│   │   └── templates/
│   └── utils/                       # Utilitaires transverses
│       ├── __init__.py
│       ├── logger.py                # Logging centralisé
│       └── event_bus.py             # Communication inter-modules
├── data/
│   ├── whitelist/                   # Photos de la whitelist
│   ├── snapshots/                   # Captures d'écran
│   ├── clips/                       # Clips vidéo d'alertes
│   └── reports/                     # Rapports PDF générés
├── tests/
│   ├── test_detector.py
│   ├── test_face_manager.py
│   ├── test_llm_client.py
│   └── ...
├── config/
│   ├── settings.yaml                # Configuration principale
│   └── prompts/
│       └── sentinel_system.txt      # System prompt Sentinel-AI
├── scripts/
│   ├── setup.py                     # Script d'installation
│   └── run.py                       # Point d'entrée principal
├── main.py                          # Entry point
├── requirements.txt
├── .env.example
└── README.md
```

**❌ Il est INTERDIT de créer des fichiers en dehors de cette structure sans modification préalable de ce document.**

---

## 🔴 RÈGLE 4 — Conventions de code

### Python
- **Python 3.10+** minimum.
- Utiliser des **type hints** sur toutes les fonctions.
- Docstrings obligatoires (format Google-style) sur chaque classe et méthode publique.
- Noms de variables et fonctions en `snake_case`.
- Noms de classes en `PascalCase`.
- Constantes en `UPPER_SNAKE_CASE`.
- Maximum 100 caractères par ligne.
- Imports organisés : stdlib → third-party → local (séparés par une ligne vide).

### JavaScript (Dashboard)
- **ES6+** minimum.
- Noms de variables en `camelCase`.
- Composants en `PascalCase`.
- Toujours utiliser `const` / `let`, jamais `var`.

### Logs
- Utiliser **exclusivement** le module `src/utils/logger.py` pour tout logging.
- Niveaux : `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- Format : `[TIMESTAMP] [MODULE] [LEVEL] Message`.

---

## 🔴 RÈGLE 5 — Tests obligatoires

- Chaque module nouveau **DOIT** avoir au minimum un fichier de test unitaire.
- Les tests doivent être exécutables via `pytest` depuis la racine du projet.
- **Couverture minimale : 70%** pour chaque module.
- Avant de valider une étape, exécuter : `pytest tests/ -v --tb=short`.
- Les tests d'intégration sont définis dans chaque étape et doivent être exécutés.

---

## 🔴 RÈGLE 6 — Gestion de la configuration

- **AUCUN** mot de passe, clé API ou secret ne doit être codé en dur dans le code source.
- Utiliser le fichier `config/settings.yaml` pour la configuration.
- Les secrets vont dans `.env` (qui est dans `.gitignore`).
- Un fichier `.env.example` avec des valeurs factices **DOIT** être maintenu à jour.

---

## 🔴 RÈGLE 7 — Commits et versioning

- **Un commit par sous-tâche** (pas un commit géant par étape).
- Format du message de commit :
  ```
  [ETAPE-XX] type: description courte
  
  Exemples:
  [ETAPE-01] feat: ajout du module config centralisé
  [ETAPE-03] fix: correction du parsing JSON pour modèles thinking
  [ETAPE-05] test: ajout tests unitaires pour tts_engine
  ```
- Types autorisés : `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

---

## 🔴 RÈGLE 8 — Communication inter-modules

- Les modules communiquent via l'**Event Bus** (`src/utils/event_bus.py`).
- **Aucun import circulaire** n'est toléré.
- Les interfaces entre modules sont définies par des **classes abstraites** ou des **Protocol** (typing).
- Le couplage doit rester **minimal** : chaque module doit pouvoir fonctionner et se tester indépendamment.

---

## 🔴 RÈGLE 9 — Gestion des erreurs

- **Jamais** de `except: pass` ou `except Exception: pass` silencieux.
- Toute exception doit être **loguée** avec le traceback complet.
- Les modules doivent définir leurs propres exceptions personnalisées dans un fichier `exceptions.py`.
- Les erreurs réseau (API Ollama, caméra IP) doivent déclencher un **mécanisme de retry** avec backoff exponentiel.

---

## 🔴 RÈGLE 10 — Performance et ressources

- Le pipeline CPU doit maintenir **≥15 FPS** en détection.
- Les requêtes LLM sont **asynchrones** — elles ne doivent JAMAIS bloquer le flux vidéo.
- Monitorer la mémoire RAM : alerter si usage > 80%.
- Les modèles YOLO doivent être chargés **une seule fois** au démarrage.
- Utiliser le **threading** pour séparer capture vidéo, détection et audio.

---

## 🟡 RÈGLE BONUS — Checklist avant de passer à l'étape suivante

Avant de passer à l'étape N+1, vérifier :

- [ ] Le `RESUME_ETAPE_XX.md` est rédigé et complet.
- [ ] Tous les tests de l'étape passent (`pytest`).
- [ ] Le code est committé avec les bons messages.
- [ ] Aucun `TODO` ou `FIXME` critique ne reste dans le code de l'étape.
- [ ] Le fichier de l'étape suivante a été relu et compris.
- [ ] Les dépendances listées dans le résumé sont satisfaites.

---

> **Ce document est vivant. Il peut être mis à jour, mais toute modification doit être tracée par un commit `[RULES] docs: ...`.**
