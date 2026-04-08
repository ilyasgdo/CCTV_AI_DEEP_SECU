# 🧪 ÉTAPE 8 — Tests, Optimisation et Hardening

> **Durée estimée : 3-4 jours**
> **Priorité : HAUTE**
> **Prérequis : ÉTAPE 7 complétée et validée**

---

## 🎯 Objectif

Renforcer la qualité, la stabilité et les performances du système. Cette étape se concentre sur les tests approfondis, l'optimisation des performances, la sécurisation, et la correction des bugs résiduels. Le but est d'atteindre un niveau de qualité **production-ready**.

---

## 📋 Tâches détaillées

### 8.1 — Suite de Tests Complète

#### Tests Unitaires (couverture ≥ 80%)

Compléter les tests unitaires pour atteindre la couverture cible :

```
tests/
├── unit/
│   ├── test_config.py              # Config loading, defaults, validation
│   ├── test_logger.py              # Log format, rotation, levels
│   ├── test_event_bus.py           # Subscribe, emit, unsubscribe
│   ├── test_camera.py              # Frame capture, reconnect, metrics
│   ├── test_detector.py            # Detection accuracy, classes, confidence
│   ├── test_tracker.py             # ID assignment, state transitions
│   ├── test_face_manager.py        # Enrollment, recognition, threshold
│   ├── test_llm_client.py          # HTTP requests, retry, timeout
│   ├── test_prompt_manager.py      # Prompt construction, context
│   ├── test_response_parser.py     # <think> cleaning, JSON extraction
│   ├── test_tts_engine.py          # Speech generation, queue
│   ├── test_stt_engine.py          # Transcription, VAD
│   ├── test_tool_executor.py       # Tool dispatch, error handling
│   ├── test_email_tool.py          # SMTP mock
│   └── test_alarm_tool.py          # Sound playback mock
├── integration/
│   ├── test_perception_pipeline.py # Camera → Detector → Tracker → Face
│   ├── test_cognitive_pipeline.py  # Prompt → LLM → Parser → Actions
│   ├── test_audio_pipeline.py      # STT → LLM → TTS sync
│   ├── test_dashboard_api.py       # Flask routes, WebSocket
│   └── test_full_workflow.py       # Cycle 5s complet
├── performance/
│   ├── test_fps_benchmark.py       # FPS sous charge
│   ├── test_memory_profile.py      # Profil mémoire sur 1h
│   └── test_latency_llm.py         # Latence LLM sous charge
└── conftest.py                     # Fixtures partagées
```

#### Tests d'intégration

Scénarios end-to-end à valider :

| # | Scénario | Durée | Validation |
|---|----------|-------|------------|
| 1 | Surveillance vide (personne) | 5 min | Pas de faux positif, RAM stable |
| 2 | Personne connue entre et sort | 2 min | Reconnaissance + salutation + tracking |
| 3 | Inconnu entre et dialogue | 5 min | Interrogation + suivi + snapshot |
| 4 | Multi-personnes | 5 min | IDs distincts, reconnaissance correcte |
| 5 | Perte de caméra | 2 min | Reconnexion auto, notification |
| 6 | Perte du LLM | 2 min | Mode dégradé activé |
| 7 | Changement de config live | 1 min | Nouveaux paramètres appliqués |
| 8 | Stress test 1h | 60 min | Pas de crash, RAM stable |

### 8.2 — Optimisation des performances

#### CPU — Pipeline de Perception

| Métrique | Cible | Comment mesurer |
|----------|-------|-----------------|
| FPS détection | ≥ 15 | `time.perf_counter()` autour de `detect()` |
| Latence frame | < 50ms | Temps entre capture et affichage |
| RAM usage | < 2 GB | `psutil.Process().memory_info()` |
| CPU usage | < 60% | `psutil.cpu_percent()` |

**Optimisations à appliquer :**
1. **Réduction taille inférence** : 640x640 → 320x320 si FPS insuffisant.
2. **Skip frames** : Détecter 1 frame sur 2 ou 3.
3. **ROI** : Détecter uniquement dans la zone d'intérêt.
4. **Thread pool** : Utiliser `concurrent.futures.ThreadPoolExecutor`.
5. **Batch processing** : Si multi-caméra, grouper les inférences.

#### GPU/LLM — Pipeline Cognitif

| Métrique | Cible | Comment mesurer |
|----------|-------|-----------------|
| Latence LLM | < 3s | Temps de POST à réponse |
| Token/s | ≥ 30 | Mesurer le streaming |
| Taux d'erreur JSON | < 5% | Compteur d'erreurs de parsing |

**Optimisations :**
1. **Context window** : Limiter le contexte à l'essentiel.
2. **Temperature** : 0.2-0.3 pour des réponses plus déterministes.
3. **Max tokens** : Limiter à 512 pour des réponses courtes.
4. **Cache** : Cacher les réponses pour des scènes statiques.

### 8.3 — Sécurisation

#### Authentification Dashboard

```python
# Authentification basique pour le dashboard
from functools import wraps
from flask import request, Response

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response('Accès refusé', 401,
                          {'WWW-Authenticate': 'Basic realm="Sentinel-AI"'})
        return f(*args, **kwargs)
    return decorated
```

#### Checklist sécurité

- [ ] Dashboard protégé par mot de passe.
- [ ] HTTPS recommandé (instructions dans la doc).
- [ ] Pas de secrets dans le code source.
- [ ] Validation des entrées API (injection, XSS).
- [ ] Rate limiting sur les endpoints API.
- [ ] Logs des accès au dashboard.
- [ ] Données biométriques (embeddings) chiffrées au repos.

### 8.4 — Monitoring et Métriques

Implémenter un module de monitoring :

```python
class SystemMonitor:
    """Surveille les métriques système en continu."""
    
    def get_metrics(self) -> dict:
        return {
            "cpu_percent": psutil.cpu_percent(),
            "ram_used_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "fps_current": self.perception.current_fps,
            "fps_average": self.perception.average_fps,
            "llm_latency_ms": self.cognitive.last_latency_ms,
            "llm_error_rate": self.cognitive.error_rate,
            "persons_detected": self.tracker.active_count,
            "uptime_seconds": time.time() - self.start_time,
            "events_total": self.event_count,
            "alerts_total": self.alert_count,
        }
```

### 8.5 — Documentation Technique

Rédiger la documentation technique :

- `DOCS/ARCHITECTURE.md` — Diagramme d'architecture détaillé.
- `DOCS/API_REFERENCE.md` — Documentation de l'API REST.
- `DOCS/DEPLOYMENT.md` — Guide de déploiement.
- `DOCS/TROUBLESHOOTING.md` — Guide de dépannage.
- `DOCS/CONFIGURATION.md` — Référence de configuration complète.

---

## 🧪 Tests de validation (Étape 8)

| # | Test | Résultat attendu |
|---|------|-------------------|
| 1 | `pytest tests/ -v --cov` | Couverture ≥ 80% |
| 2 | Stress test 1h | Pas de crash, RAM stable |
| 3 | FPS benchmark | ≥ 15 FPS sur CPU |
| 4 | LLM latency | < 3s en moyenne |
| 5 | Auth dashboard | 401 sans credentials |
| 6 | API validation | Inputs invalides → erreurs propres |
| 7 | Monitoring | Métriques exposées en temps réel |

---

## 📦 Livrables de l'étape

- [ ] Suite de tests complète (30+ tests)
- [ ] Rapport de couverture ≥ 80%
- [ ] Benchmark de performances documenté
- [ ] Auth dashboard implémentée
- [ ] Module de monitoring
- [ ] Documentation technique (5 fichiers)
- [ ] `RESUME_ETAPE_08.md`

---

## 🔗 Dépendances pour l'étape suivante

L'étape 9 nécessite :
- ✅ Tous les tests passent
- ✅ Performances validées
- ✅ Sécurité vérifiée
- ✅ Documentation complète
