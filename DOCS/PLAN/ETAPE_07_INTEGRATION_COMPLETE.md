# 🔗 ÉTAPE 7 — Intégration Complète et Pipeline Principal

> **Durée estimée : 4-5 jours**
> **Priorité : CRITIQUE**
> **Prérequis : ÉTAPES 1-6 complétées et validées**

---

## 🎯 Objectif

Assembler tous les modules développés dans les étapes précédentes en un système unifié et fonctionnel. C'est l'étape d'**intégration** : connecter le pipeline de perception (étape 2-3), le pipeline cognitif (étape 4), le pipeline audio (étape 5), le dashboard (étape 6), et valider le **workflow principal des 5 secondes** tel que décrit dans le MVP.

---

## 📋 Tâches détaillées

### 7.1 — Point d'entrée principal (`main.py`)

Refactoriser `main.py` pour orchestrer le démarrage complet :

```python
"""
Sentinel-AI — Point d'entrée principal
Orchestre le démarrage et l'arrêt propre de tous les sous-systèmes.
"""

class SentinelAI:
    """
    Classe principale qui initialise et coordonne tous les modules.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        # 1. Configuration
        self.config = Config(config_path)
        self.logger = get_logger("sentinel")
        self.event_bus = EventBus()
        
        # 2. Pipeline Perception (CPU)
        self.camera = Camera(self.config, self.event_bus)
        self.detector = ObjectDetector(self.config)
        self.tracker = Tracker(self.config, self.event_bus)
        self.face_manager = FaceManager(self.config, self.event_bus)
        
        # 3. Pipeline Cognitif (GPU distant/local)
        self.llm_client = LLMClient(self.config)
        self.prompt_manager = PromptManager(self.config)
        self.response_parser = ResponseParser()
        self.orchestrator = AnalysisOrchestrator(...)
        
        # 4. Pipeline Audio
        self.tts = TTSEngine(self.config)
        self.stt = STTEngine(self.config)
        
        # 5. Effecteurs
        self.tool_executor = ToolExecutor(self.config, self.tts)
        
        # 6. Dashboard
        self.dashboard = DashboardApp(self.config, ...)
    
    async def start(self):
        """Démarre tous les sous-systèmes dans le bon ordre."""
        self.logger.info("🛡️ Démarrage de Sentinel-AI...")
        
        # Ordre de démarrage important:
        # 1. Config + Logger (déjà fait)
        # 2. Event Bus (déjà fait)
        # 3. Caméra
        # 4. Détecteur + Tracker + Face Manager
        # 5. LLM Client (vérifier connexion)
        # 6. Audio (TTS + STT)
        # 7. Tool Executor
        # 8. Dashboard (dernier, dans un thread séparé)
        # 9. Boucle principale
```

### 7.2 — Le Workflow des 5 Secondes (Implémentation complète)

Implémenter le workflow exact décrit dans le MVP :

```
[T=0.0s à 4.9s] — PHASE PERCEPTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CPU Thread 1: Capture caméra continue (30 FPS)
  CPU Thread 2: Détection YOLO + Tracking + Face Recognition
  CPU Thread 3: STT écoute continue (transcription)
  CPU Thread 4: TTS joue les messages en queue

[T=5.0s] — PHASE SNAPSHOT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Capturer la frame actuelle
  2. Compiler le contexte:
     - Liste des détections YOLO
     - État du tracker (qui est là, depuis combien de temps)
     - Résultat reconnaissance faciale (connus/inconnus)
     - Transcription STT des 5 dernières secondes
     - Contexte conversationnel (mémoire des échanges précédents)
  3. Encoder la frame en base64

[T=5.1s] — PHASE REQUÊTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Construire le prompt complet (PromptManager)
  2. POST vers l'API LLM (async, non-bloquant)
  3. Le pipeline de perception continue pendant l'attente

[T=5.5s - 6.5s] — PHASE COGNITION (GPU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Ollama/LLM génère:
  <think>
  [Raisonnement interne...]
  </think>
  {
    "action_vocale": "...",
    "niveau_alerte": "...",
    "outils_a_lancer": [...]
  }

[T=6.6s] — PHASE PARSING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Réception de la réponse brute
  2. Nettoyage des balises <think> (ResponseParser)
  3. Extraction et validation du JSON
  4. Création de l'ActionResponse

[T=6.7s] — PHASE EXÉCUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. TTS lit l'action_vocale
  2. ToolExecutor exécute les outils
  3. Events émis vers le dashboard (WebSocket)
  4. Logging de l'événement complet

[T=7.0s - 9.9s] — RETOUR À LA PHASE PERCEPTION
```

### 7.3 — Gestion du Threading

**Architecture multi-thread :**

```
MainThread
  ├── Thread: Camera Capture (boucle continue)
  ├── Thread: Detection + Tracking (boucle continue)
  ├── Thread: STT Listening (boucle continue)
  ├── AsyncLoop: Analysis Orchestrator (cycle 5s)
  │     ├── LLM Request (async)
  │     ├── Response Parse (sync, rapide)
  │     └── Tool Execution (async)
  ├── Thread: TTS Playback (queue-based)
  └── Thread: Flask Dashboard (serveur web)
```

**Synchronisation :**
- `threading.Event` pour les signaux start/stop.
- `queue.Queue` pour la communication inter-threads.
- `asyncio.Event` pour la coordination async.
- **Aucun lock global** — chaque module a sa propre synchronisation.

### 7.4 — Mode dégradé

Implémenter des modes de fonctionnement dégradés :

| Situation | Comportement |
|-----------|-------------|
| LLM indisponible | Surveillance silencieuse, alertes uniquement visuelles |
| Caméra déconnectée | Tentative de reconnexion, notification dashboard |
| Micro indisponible | Fonctionnement sans STT, saisie texte via dashboard |
| Haut-parleur absent | Actions textuelles uniquement (dashboard + email) |
| GPU absent | Tout sur CPU (plus lent mais fonctionnel) |

### 7.5 — Arrêt propre (Graceful Shutdown)

```python
async def shutdown(self):
    """Arrêt propre de tous les sous-systèmes."""
    self.logger.info("🛑 Arrêt de Sentinel-AI...")
    
    # 1. Arrêter l'analyse
    self.orchestrator.stop()
    
    # 2. Arrêter l'audio
    self.stt.stop_listening()
    await self.tts.speak("Système Sentinel en cours d'arrêt. Au revoir.")
    
    # 3. Arrêter le dashboard
    self.dashboard.stop()
    
    # 4. Arrêter la caméra
    self.camera.release()
    
    # 5. Sauvegarder l'état
    self.save_state()
    
    self.logger.info("✅ Sentinel-AI arrêté proprement.")
```

### 7.6 — Script de lancement

Créer `scripts/run.py` avec des arguments CLI :

```bash
python scripts/run.py                          # Démarrage normal
python scripts/run.py --config custom.yaml     # Config custom
python scripts/run.py --no-audio               # Sans audio
python scripts/run.py --no-dashboard           # Sans dashboard (headless)
python scripts/run.py --camera 0               # Spécifier la caméra
python scripts/run.py --llm-url http://...     # Override URL LLM
python scripts/run.py --check                  # Vérifier les dépendances
python scripts/run.py --demo                   # Mode demo avec vidéo test
```

---

## 🧪 Tests de validation (Étape 7)

| # | Test End-to-End | Description | Résultat attendu |
|---|----------------|-------------|-------------------|
| 1 | Startup complet | Lancer `main.py` | Tous les modules OK |
| 2 | Workflow 5s | Personne devant caméra | Analyse + réponse vocale |
| 3 | Connu reconnu | Personne whitelist | "Bonjour [Nom]" en TTS |
| 4 | Inconnu détecté | Personne non-whitelist | Interrogation + snapshot |
| 5 | Tool Email | LLM demande d'envoyer email | Email reçu |
| 6 | Dashboard Live | Ouvrir dashboard pendant surveillance | Flux vidéo + events live |
| 7 | Change LLM URL | Modifier URL dans settings | Reconnexion au nouveau LLM |
| 8 | Graceful Stop | Ctrl+C pendant fonctionnement | Arrêt propre, pas de crash |
| 9 | No LLM | Arrêter Ollama | Mode dégradé activé |
| 10 | Camera Lost | Débrancher webcam | Reconnexion auto + notification |
| 11 | Multi-person | 2 personnes (1 connue, 1 inconnue) | Comportement différencié |
| 12 | Conversation | Dialogue multi-tours | Cohérence contextuelle |

---

## 📦 Livrables de l'étape

- [ ] `main.py` — Point d'entrée complet
- [ ] `scripts/run.py` — Lanceur CLI avec arguments
- [ ] Workflow 5s intégralement fonctionnel
- [ ] Threading stable et documenté
- [ ] Mode dégradé implémenté
- [ ] Graceful shutdown
- [ ] Tests end-to-end (12 scénarios)
- [ ] `RESUME_ETAPE_07.md`

---

## ⚠️ Points d'attention

- **C'est l'étape la plus complexe** — prendre le temps de tester chaque connexion inter-module.
- **Deadlocks** : Attention aux locks croisés entre threads.
- **Mémoire** : Surveiller les fuites mémoire (frames non libérées, etc.).
- **Le threading est délicat** : Utiliser `ThreadPoolExecutor` pour simplifier.
- **Tester chaque mode dégradé** individuellement.

---

## 🔗 Dépendances pour l'étape suivante

L'étape 8 nécessite :
- ✅ Système complet fonctionnel en mode nominal
- ✅ Dashboard accessible et affichant le live
- ✅ Workflow 5s validé end-to-end
- ✅ Modes dégradés testés
