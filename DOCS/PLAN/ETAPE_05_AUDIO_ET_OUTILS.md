# 🔊 ÉTAPE 5 — Audio Bidirectionnel et Système d'Outils (Effecteurs)

> **Durée estimée : 3-4 jours**
> **Priorité : HAUTE**
> **Prérequis : ÉTAPE 4 complétée et validée**

---

## 🎯 Objectif

Donner une voix à Sentinel-AI (TTS) et des oreilles (STT), puis implémenter le système d'exécution d'outils (Tool Use) qui permet à l'IA d'agir concrètement (envoyer des emails, déclencher des alarmes, etc.) en réponse à ses analyses.

---

## 📋 Tâches détaillées

### 5.1 — Text-to-Speech (`src/audio/tts_engine.py`)

Implémenter la synthèse vocale avec **edge-tts** (Microsoft Neural TTS, gratuit) :

```python
class TTSEngine:
    """
    Moteur de synthèse vocale utilisant edge-tts.
    Tourne sur CPU, aucune dépendance GPU.
    """
    
    def __init__(self, config: Config):
        self.voice = config.audio.tts_voice    # ex: "fr-FR-HenriNeural"
        self.enabled = config.audio.tts_enabled
        self.speaking = False                   # Flag anti-overlap
    
    async def speak(self, text: str) -> None:
        """Synthétise et joue le texte. Non-bloquant."""
        ...
    
    async def speak_priority(self, text: str) -> None:
        """Interrompt la parole en cours et dit le texte en priorité."""
        ...
```

**Voix disponibles FR :**
| Voix | Genre | Style |
|------|-------|-------|
| `fr-FR-HenriNeural` | Masculin | Neutre, professionnel |
| `fr-FR-DeniseNeural` | Féminin | Neutre |
| `fr-FR-EloiseNeural` | Féminin | Doux |

**Fonctionnalités :**
- File d'attente de messages (FIFO) avec priorité.
- Anti-overlap : Ne pas parler en même temps que le STT écoute.
- Volume configurable.
- Cache audio pour les phrases récurrentes (salutations, alertes standard).
- Gestion asynchrone : la parole ne bloque pas le pipeline.

**Événements :**
- `tts_speaking` : L'IA commence à parler.
- `tts_done` : L'IA a fini de parler.

### 5.2 — Speech-to-Text (`src/audio/stt_engine.py`)

Implémenter la transcription vocale avec **Faster-Whisper** :

```python
class STTEngine:
    """
    Moteur de transcription vocale basé sur Faster-Whisper.
    Optimisé pour CPU, supporte le streaming.
    """
    
    def __init__(self, config: Config):
        self.model_size = "base"    # tiny, base, small, medium, large
        self.language = "fr"
        self.model = WhisperModel(self.model_size, device="cpu")
    
    def start_listening(self) -> None:
        """Démarre l'écoute en continu dans un thread."""
        ...
    
    def stop_listening(self) -> None:
        """Arrête l'écoute."""
        ...
    
    def get_transcript(self) -> Optional[str]:
        """Retourne la dernière transcription disponible."""
        ...
```

**Fonctionnalités :**
- **Détection d'activité vocale (VAD)** : Ne transcrire que quand quelqu'un parle.
- Buffer audio circulaire de 10 secondes.
- Transcription par segments de 3-5 secondes.
- Filtrage du bruit ambiant.
- Mode alternatif : saisie texte manuelle via le dashboard.

**Gestion du conflit TTS/STT :**
- Quand le TTS parle, le STT se met en **pause** pour ne pas transcrire la voix de l'IA.
- Reprise de l'écoute 500ms après la fin du TTS.

**Événements :**
- `audio_transcribed` : `{text, confidence, duration}`
- `audio_silence` : Aucune activité vocale détectée.

### 5.3 — Système d'exécution d'outils (`src/effector/tool_executor.py`)

Implémenter le dispatcher d'outils qui exécute les commandes du LLM :

```python
class ToolExecutor:
    """
    Exécute les outils demandés par le LLM via le JSON de réponse.
    Chaque outil est un plugin indépendant et testable.
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_tools()
    
    def _register_tools(self):
        """Enregistre tous les outils disponibles."""
        self.tools["send_email"] = EmailTool(config)
        self.tools["trigger_alarm"] = AlarmTool(config)
        self.tools["save_snapshot"] = SnapshotTool(config)
        self.tools["log_event"] = EventLogTool(config)
        self.tools["announce"] = AnnounceTool(tts_engine)
    
    async def execute(self, actions: List[ToolAction]) -> List[ToolResult]:
        """Exécute une liste d'actions outils séquentiellement."""
        results = []
        for action in actions:
            tool = self.tools.get(action.tool_name)
            if tool:
                result = await tool.run(action.parametres)
                results.append(result)
                self.event_bus.emit("tool_executed", result)
        return results
```

### 5.4 — Outil Email (`src/effector/email_tool.py`)

```python
class EmailTool(BaseTool):
    """Envoie un email d'alerte avec snapshot en pièce jointe."""
    
    async def run(self, params: dict) -> ToolResult:
        """
        Params attendus:
        - urgence: str ("faible", "moyenne", "haute", "critique")
        - sujet: str
        - description: str (optionnel)
        - attach_snapshot: bool (optionnel, défaut True)
        """
```

**Configuration SMTP** via `.env` :
```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=sentinel@example.com
SMTP_PASSWORD=app_password
```

### 5.5 — Outil Alarme (`src/effector/alarm_tool.py`)

```python
class AlarmTool(BaseTool):
    """Déclenche une alarme sonore locale."""
    
    async def run(self, params: dict) -> ToolResult:
        """
        Params:
        - type: str ("intrusion", "incendie", "medical")
        - duration: int (secondes, défaut 10)
        """
        # Jouer un son d'alarme via le haut-parleur
```

### 5.6 — Outil Snapshot (`src/effector/snapshot_tool.py`)

```python
class SnapshotTool(BaseTool):
    """Sauvegarde un snapshot horodaté de la caméra."""
    
    async def run(self, params: dict) -> ToolResult:
        """Capture et sauvegarde la frame actuelle dans data/snapshots/."""
```

### 5.7 — Outil Event Log (`src/effector/event_log_tool.py`)

```python
class EventLogTool(BaseTool):
    """Enregistre un événement dans le journal de bord."""
    
    async def run(self, params: dict) -> ToolResult:
        """Ajoute une entrée dans data/event_log.json."""
```

### 5.8 — Interface de base des outils

```python
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Interface de base pour tous les outils Sentinel-AI."""
    
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @property
    @abstractmethod
    def description(self) -> str: ...
    
    @abstractmethod
    async def run(self, params: dict) -> ToolResult: ...
```

---

## 🧪 Tests de validation (Étape 5)

| # | Test | Description | Résultat attendu |
|---|------|-------------|-------------------|
| 1 | TTS Basic | `speak("Bonjour, bienvenue")` | Audio joué |
| 2 | TTS Queue | 3 messages rapides | Joués dans l'ordre |
| 3 | TTS Priority | Message prioritaire pendant parole | Interruption + nouveau message |
| 4 | STT Basic | Dire "Bonjour" au micro | Transcription "bonjour" |
| 5 | STT VAD | Silence pendant 10s | Pas de transcription parasite |
| 6 | TTS/STT Sync | TTS parle → STT doit se taire | Pas d'écho |
| 7 | Email Tool | Envoi email test | Email reçu avec snapshot |
| 8 | Alarm Tool | Déclencher alarme 3s | Son joué pendant 3s |
| 9 | Snapshot Tool | Sauvegarder frame | Fichier JPEG dans data/snapshots/ |
| 10 | Tool Executor | ActionResponse avec 2 outils | 2 outils exécutés |
| 11 | Tool Unknown | Outil inexistant demandé | Erreur loguée, pas de crash |

---

## 📦 Livrables de l'étape

- [ ] `src/audio/tts_engine.py` — Synthèse vocale
- [ ] `src/audio/stt_engine.py` — Transcription vocale
- [ ] `src/effector/tool_executor.py` — Dispatcher d'outils
- [ ] `src/effector/base_tool.py` — Interface abstraite
- [ ] `src/effector/email_tool.py` — Envoi d'emails
- [ ] `src/effector/alarm_tool.py` — Alarme sonore
- [ ] `src/effector/snapshot_tool.py` — Capture de snapshots
- [ ] `src/effector/event_log_tool.py` — Journal d'événements
- [ ] `.env.example` mis à jour avec SMTP
- [ ] Tests unitaires
- [ ] `RESUME_ETAPE_05.md`

---

## ⚠️ Points d'attention

- **PyAudio** peut être difficile à installer sur certains systèmes. Prévoir un fallback.
- **Faster-Whisper** télécharge un modèle au premier lancement (~150MB pour "base").
- **Le TTS est asynchrone** — il ne doit jamais bloquer la boucle principale.
- **Les emails requièrent un mot de passe d'application** Gmail (pas le mot de passe normal).
- **Ne PAS implémenter le dashboard web** — c'est l'étape 6.

---

## 🔗 Dépendances pour l'étape suivante

L'étape 6 nécessite :
- ✅ TTS fonctionnel pour les annonces vocales
- ✅ STT fonctionnel pour capter les réponses
- ✅ Tool Executor capable d'exécuter les outils
- ✅ Tous les événements audio émis correctement
