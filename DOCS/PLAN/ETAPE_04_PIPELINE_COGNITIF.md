# 🧠 ÉTAPE 4 — Pipeline Cognitif (LLM / Gemma 4 via Ollama)

> **Durée estimée : 4-5 jours**
> **Priorité : CRITIQUE**
> **Prérequis : ÉTAPE 3 complétée et validée**

---

## 🎯 Objectif

Implémenter le cœur intelligent de Sentinel-AI : le pipeline cognitif qui prépare le "paquet de 5 secondes", l'envoie au LLM (Gemma 4 via Ollama), reçoit et parse la réponse (gestion spéciale des modèles "Thinking"), et retourne les actions à exécuter. L'API est **agnostique** : elle doit fonctionner que Ollama soit local ou distant.

---

## 📋 Tâches détaillées

### 4.1 — Client LLM Agnostique (`src/cognitive/llm_client.py`)

Créer un client HTTP flexible pour communiquer avec l'API Ollama (ou API compatible) :

**Architecture :**
```python
class LLMClient:
    """
    Client agnostique pour API de LLM (Ollama, vLLM, API custom).
    Supporte les requêtes multimodales (texte + image).
    
    L'URL de l'API est configurable dans settings.yaml,
    permettant de pointer vers un serveur local ou distant.
    """
    
    def __init__(self, config: Config):
        self.api_url = config.llm.api_url        # ex: http://localhost:11434
        self.model = config.llm.model_name        # ex: gemma4
        self.timeout = config.llm.timeout
    
    async def generate(self, prompt: str, image: Optional[bytes] = None) -> str:
        """Envoie une requête au LLM et retourne la réponse brute."""
        ...
    
    async def health_check(self) -> bool:
        """Vérifie que l'API LLM est accessible."""
        ...
```

**Endpoints Ollama à supporter :**
| Endpoint | Méthode | Usage |
|----------|---------|-------|
| `/api/generate` | POST | Génération texte (principal) |
| `/api/chat` | POST | Mode conversation (optionnel) |
| `/api/tags` | GET | Lister les modèles disponibles |
| `/api/show` | POST | Info sur un modèle spécifique |

**Requête type (multimodale) :**
```json
{
  "model": "gemma4",
  "prompt": "[System Prompt]\n[Context]\n[User Input]",
  "images": ["base64_encoded_frame"],
  "stream": false,
  "options": {
    "temperature": 0.3,
    "num_predict": 1024,
    "top_p": 0.9
  }
}
```

**Gestion de la résilience :**
- Timeout configurable (défaut: 30s).
- Retry automatique (3 tentatives avec backoff exponentiel).
- Fallback gracieux si le LLM est indisponible (mode "veille").
- Logging de chaque requête/réponse (en mode DEBUG).
- Métriques : temps de réponse, tokens utilisés, taux d'erreur.

### 4.2 — Gestionnaire de Prompts (`src/cognitive/prompt_manager.py`)

Gérer les prompts système et la construction du contexte :

**System Prompt (à stocker dans `config/prompts/sentinel_system.txt`) :**
```
Tu es Sentinel-AI, un agent de sécurité autonome, observateur, ferme mais courtois.
Tu reçois des données de capteurs toutes les 5 secondes :
- Image de la caméra (analyse visuelle)
- Objets détectés (liste YOLO)
- Personnes identifiées (whitelist)
- Contexte auditif (transcription STT si disponible)

Instructions Opérationnelles :
1. IDENTITÉ : Salue les personnes de la Whitelist par leur nom. Interroge les inconnus.
2. COHÉRENCE : Si un inconnu parle, compare ses dires avec la scène visuelle.
3. SÉCURITÉ : Ne valide JAMAIS un accès non autorisé. En cas de menace, utilise tes outils.
4. MÉMOIRE : Réfère-toi au contexte précédent pour maintenir la cohérence du dialogue.

Protocole de Réponse OBLIGATOIRE :
Utilise <think>...</think> pour ton raisonnement, puis fournis un JSON strict :
<think>(ton raisonnement ici)</think>
{
  "action_vocale": "phrase TTS",
  "niveau_alerte": "normal|attention|alerte|critique",
  "outils_a_lancer": [{"tool_name": "...", "parametres": {...}}]
}
```

**Construction du prompt contextuel :**
```python
class PromptManager:
    def build_analysis_prompt(
        self,
        detections: List[Detection],
        tracked_entities: List[TrackedEntity],
        audio_transcript: Optional[str],
        previous_context: Optional[str]
    ) -> str:
        """
        Construit le prompt complet avec le contexte de la scène.
        
        Exemple de sortie:
        ---
        [RAPPORT CAPTEURS - 2026-04-08 12:05:00]
        
        OBJETS DÉTECTÉS:
        - Personne #1 (confiance: 0.92) - Position: centre
        - Personne #2 (confiance: 0.87) - Position: gauche
        - Sac à dos (confiance: 0.76) - Position: sol, centre
        
        IDENTIFICATION:
        - Personne #1: CONNU - Ilyas Ghandaoui (Propriétaire) - Score: 0.89
        - Personne #2: INCONNU - Temps de présence: 45s
        
        CONTEXTE AUDITIF:
        "Bonjour, je suis le livreur, j'ai un colis pour M. Ghandaoui"
        
        CONTEXTE PRÉCÉDENT:
        L'inconnu est entré dans le champ il y a 50s. Première interaction.
        ---
        """
```

### 4.3 — Parseur de Réponse (`src/cognitive/response_parser.py`)

**CRITIQUE** — Ce module est vital car les modèles "Thinking" (Gemma 4) enveloppent leur réponse dans des balises `<think>` :

```python
class ResponseParser:
    """
    Parse la réponse brute du LLM pour extraire le JSON d'action.
    Gère les modèles "Thinking" qui ajoutent des balises <think>...</think>.
    """
    
    def parse(self, raw_response: str) -> ActionResponse:
        """
        Étapes:
        1. Extraire et sauvegarder le contenu <think> (pour debug/logging)
        2. Supprimer les balises <think>...</think>
        3. Nettoyer le texte restant (espaces, retours à la ligne)
        4. Extraire le bloc JSON
        5. Valider le JSON contre le schéma attendu
        6. Retourner un ActionResponse typé
        """
```

**Cas edge à gérer :**
- Réponse sans balises `<think>` (modèle non-thinking).
- JSON malformé → Retry ou réponse par défaut.
- Réponse tronquée (timeout) → Log + réponse de sécurité.
- Multiples blocs JSON → Prendre le dernier.
- JSON avec des champs manquants → Valeurs par défaut.
- Balises `<think>` imbriquées ou malformées.

**Classe `ActionResponse` :**
```python
@dataclass
class ActionResponse:
    action_vocale: str                    # Texte à dire
    niveau_alerte: str                    # normal, attention, alerte, critique
    outils_a_lancer: List[ToolAction]     # Actions à exécuter
    raw_thinking: Optional[str]           # Pensée interne (debug)
    parse_success: bool                   # Parsing réussi?
    parse_errors: List[str]               # Erreurs de parsing
    response_time_ms: float               # Latence LLM
```

### 4.4 — Orchestrateur d'Analyse Asynchrone

Créer l'orchestrateur qui pilote le cycle d'analyse toutes les 5 secondes :

```python
class AnalysisOrchestrator:
    """
    Orchestre le cycle d'analyse asynchrone:
    1. Toutes les 5s, capture un snapshot
    2. Compile le contexte (détections, tracking, audio)
    3. Envoie au LLM de manière NON-BLOQUANTE
    4. Reçoit et dispatch la réponse
    """
    
    async def analysis_loop(self):
        while self.running:
            # 1. Snapshot
            frame = self.camera.get_snapshot()
            
            # 2. Contexte
            prompt = self.prompt_manager.build_analysis_prompt(...)
            
            # 3. Envoi async au LLM (NE BLOQUE PAS le flux vidéo)
            response = await self.llm_client.generate(prompt, frame)
            
            # 4. Parse
            action = self.parser.parse(response)
            
            # 5. Dispatch
            self.event_bus.emit("llm_response", action)
            
            # 6. Attendre le prochain cycle
            await asyncio.sleep(self.analysis_interval)
```

**Points critiques :**
- L'analyse est **asynchrone** — elle ne doit JAMAIS bloquer le flux vidéo.
- Si une analyse est en cours quand le prochain cycle arrive, **skip** (ne pas empiler).
- Stocker le contexte précédent pour la cohérence conversationnelle.
- Mode dégradé si le LLM ne répond pas : continuer la surveillance silencieusement.

### 4.5 — Gestion de l'historique conversationnel

Maintenir une mémoire courte pour la cohérence du dialogue :

```python
class ConversationMemory:
    """
    Mémoire glissante des N dernières interactions.
    Permet au LLM de maintenir un contexte cohérent.
    """
    def __init__(self, max_entries: int = 10):
        self.history: Deque[dict] = deque(maxlen=max_entries)
    
    def add_exchange(self, context: str, response: ActionResponse):
        ...
    
    def get_context_summary(self) -> str:
        """Résumé des interactions récentes pour le prompt."""
        ...
```

---

## 🧪 Tests de validation (Étape 4)

| # | Test | Description | Résultat attendu |
|---|------|-------------|-------------------|
| 1 | Health Check | `llm_client.health_check()` | `True` si Ollama tourne |
| 2 | Generate Text | Prompt simple sans image | Réponse textuelle valide |
| 3 | Generate Multimodal | Prompt + image JPEG | Analyse de l'image |
| 4 | Parse Thinking | Réponse avec `<think>...</think>` | JSON extrait correctement |
| 5 | Parse No Think | Réponse sans balises think | JSON extrait correctement |
| 6 | Parse Malformed | JSON cassé dans la réponse | Erreur gérée proprement |
| 7 | Prompt Build | Contexte avec 3 détections | Prompt structuré correct |
| 8 | Async No Block | Analyse pendant flux vidéo | FPS vidéo stable |
| 9 | Retry Logic | Ollama arrêté puis relancé | Reconnexion auto |
| 10 | Conversation Memory | 5 échanges | Résumé cohérent |

### Tests unitaires obligatoires :
- `tests/test_llm_client.py` — Tests avec mock HTTP.
- `tests/test_response_parser.py` — Tests de parsing (10+ cas edge).
- `tests/test_prompt_manager.py` — Tests de construction de prompts.

---

## 📦 Livrables de l'étape

- [ ] `src/cognitive/llm_client.py` — Client API agnostique
- [ ] `src/cognitive/prompt_manager.py` — Construction de prompts contextuels
- [ ] `src/cognitive/response_parser.py` — Parsing robuste des réponses LLM
- [ ] `config/prompts/sentinel_system.txt` — System prompt optimisé
- [ ] Orchestrateur d'analyse asynchrone
- [ ] Mémoire conversationnelle
- [ ] Demo : `scripts/demo_cognitive.py`
- [ ] Tests unitaires (3 fichiers, 10+ tests)
- [ ] `RESUME_ETAPE_04.md`

---

## ⚠️ Points d'attention

- **Ollama DOIT être installé** et le modèle Gemma 4 téléchargé (`ollama pull gemma4`).
- **Le parsing est le point critique** — un seul bug ici peut casser tout le système.
- **Ne PAS implémenter le TTS/STT** — c'est l'étape 5.
- **Ne PAS implémenter les outils (email, alarme)** — c'est l'étape 5.
- Tester avec des réponses mockées d'abord avant de brancher le vrai LLM.

---

## 🔗 Dépendances pour l'étape suivante

L'étape 5 nécessite :
- ✅ `llm_client.py` capable d'envoyer des requêtes multimodales
- ✅ `response_parser.py` retournant des `ActionResponse` fiables
- ✅ Cycle d'analyse toutes les 5s fonctionnel
- ✅ Événement `llm_response` émis avec les actions parsées
