# 🖥️ ÉTAPE 6 — Dashboard Web (Front-End)

> **Durée estimée : 5-6 jours**
> **Priorité : HAUTE**
> **Prérequis : ÉTAPE 5 complétée et validée**

---

## 🎯 Objectif

Créer l'interface web complète de Sentinel-AI : un dashboard moderne et intuitif permettant de visualiser le flux caméra en direct, gérer la whitelist, consulter le journal d'événements, et configurer le système. L'interface utilise **Flask** côté serveur et **HTML/CSS/JS** (vanilla) côté client.

---

## 📋 Tâches détaillées

### 6.1 — Serveur Flask (`src/dashboard/app.py`)

**Architecture du serveur :**

```python
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO  # Pour le temps réel

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
```

**Routes API :**

| Route | Méthode | Description |
|-------|---------|-------------|
| `/` | GET | Dashboard principal (Live View) |
| `/api/stream` | GET | Flux vidéo MJPEG |
| `/api/events` | GET | Journal des événements (paginé) |
| `/api/events/<id>` | GET | Détail d'un événement |
| `/api/whitelist` | GET | Liste des personnes enrôlées |
| `/api/whitelist` | POST | Enrôler une nouvelle personne |
| `/api/whitelist/<id>` | DELETE | Supprimer une personne |
| `/api/whitelist/<id>` | PUT | Modifier une personne |
| `/api/settings` | GET | Configuration actuelle |
| `/api/settings` | PUT | Modifier la configuration |
| `/api/settings/llm/test` | POST | Tester la connexion au LLM |
| `/api/status` | GET | État du système (CPU, RAM, FPS, LLM) |
| `/api/snapshots` | GET | Galerie de snapshots |
| `/api/stats` | GET | Statistiques (personnes/jour, alertes, etc.) |

### 6.2 — Flux Vidéo Temps Réel

**Option A — MJPEG (Simple, fiable) :**
```python
@app.route('/api/stream')
def video_stream():
    def generate():
        while True:
            frame = camera.get_frame()
            annotated = visualizer.draw(frame, tracker.entities)
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
```

**Option B — WebSocket (Avancé, recommandé pour la latence) :**
```python
@socketio.on('request_frame')
def send_frame():
    frame = camera.get_frame()
    annotated = visualizer.draw(frame, tracker.entities)
    _, buffer = cv2.imencode('.jpg', annotated)
    emit('video_frame', {'image': base64.b64encode(buffer).decode()})
```

### 6.3 — Pages du Dashboard

#### 6.3.1 — Live View (Page principale)

**Layout :**
```
┌──────────────────────────────────────────────────────┐
│  🛡️ SENTINEL-AI            [Status: ● En ligne]     │
├──────────────────────────┬───────────────────────────┤
│                          │  📊 Infos temps réel      │
│                          │  ─────────────────────    │
│   [Flux Vidéo Live]      │  FPS: 28                  │
│   (avec bounding boxes   │  Personnes: 2             │
│    et labels)            │  Alertes: 0               │
│                          │  LLM: ● Connecté          │
│                          │  CPU: 45% | RAM: 62%      │
│                          │  ─────────────────────    │
│                          │  🗣️ Dernière action IA:   │
│                          │  "Bonsoir M. Ghandaoui"   │
├──────────────────────────┴───────────────────────────┤
│  📜 Journal des événements (feed en temps réel)      │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 🟢 12:05:30 - Ilyas Ghandaoui reconnu (0.92)   │ │
│  │ 🟡 12:03:15 - Personne inconnue détectée        │ │
│  │ 🔴 12:01:00 - Alerte: mouvement zone interdite  │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

#### 6.3.2 — Gestion Whitelist

**Interface d'enrôlement :**
```
┌──────────────────────────────────────────────────────┐
│  👥 Gestion de la Whitelist                          │
├──────────────────────────────────────────────────────┤
│  [+ Ajouter une personne]                            │
│                                                      │
│  ┌────────┐ Ilyas Ghandaoui    Admin    ✏️ 🗑️      │
│  │ [Photo]│ Enrôlé: 08/04/2026                      │
│  │        │ Dernière vue: il y a 2h                  │
│  └────────┘                                          │
│                                                      │
│  ┌────────┐ Marie Dupont       Employée  ✏️ 🗑️     │
│  │ [Photo]│ Enrôlé: 05/04/2026                      │
│  │        │ Dernière vue: il y a 1j                  │
│  └────────┘                                          │
├──────────────────────────────────────────────────────┤
│  📸 Enrôlement rapide                                │
│  [Capturer 5 photos depuis la caméra]                │
│  ou                                                  │
│  [Uploader des photos]                               │
└──────────────────────────────────────────────────────┘
```

#### 6.3.3 — Journal des événements

- Feed chronologique avec filtres (date, type, personne).
- Chaque événement montre : timestamp, snapshot, type d'événement, action IA.
- Export CSV des événements.

#### 6.3.4 — Page Paramètres

**Sections :**
- **🎥 Caméra** : Source vidéo, résolution, FPS.
- **🧠 LLM** : URL API, modèle, timeout, intervalle d'analyse + bouton "Tester la connexion".
- **🔊 Audio** : Activer/désactiver TTS/STT, choix de la voix.
- **📧 Alertes** : Configuration SMTP, email destinataire.
- **⚙️ Système** : Seuils de détection, niveaux de log.

### 6.4 — Design et CSS

**Palette de couleurs (thème sombre) :**
```css
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #1a1a2e;
    --accent-primary: #00d4ff;    /* Cyan néon */
    --accent-secondary: #7c3aed;  /* Violet */
    --accent-success: #10b981;    /* Vert */
    --accent-warning: #f59e0b;    /* Orange */
    --accent-danger: #ef4444;     /* Rouge */
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --border-color: #2a2a3e;
}
```

**Effets visuels :**
- Glassmorphism sur les cartes.
- Animations de pulse sur les alertes.
- Transitions fluides entre les pages.
- Police : Inter ou Roboto Mono (Google Fonts).
- Responsive design (desktop + tablette).

### 6.5 — WebSocket pour le temps réel

Utiliser Socket.IO pour les mises à jour en temps réel :

```javascript
const socket = io();

// Recevoir les événements en temps réel
socket.on('new_event', (data) => {
    addEventToFeed(data);
});

socket.on('person_detected', (data) => {
    updatePersonCount(data.count);
});

socket.on('alert', (data) => {
    showAlertNotification(data);
});

socket.on('system_status', (data) => {
    updateSystemMetrics(data);
});
```

---

## 🧪 Tests de validation (Étape 6)

| # | Test | Description | Résultat attendu |
|---|------|-------------|-------------------|
| 1 | Serveur Start | `python -m src.dashboard.app` | Dashboard accessible sur :5000 |
| 2 | Live View | Ouvrir la page principale | Flux vidéo en direct visible |
| 3 | Events API | `GET /api/events` | JSON des événements |
| 4 | Whitelist CRUD | Ajouter/Modifier/Supprimer | Opérations réussies |
| 5 | Settings Save | Modifier URL LLM | Config mise à jour |
| 6 | LLM Test | Bouton "Tester connexion" | Feedback visuel (✅ ou ❌) |
| 7 | Real-time Events | Créer une alerte | Apparaît dans le feed live |
| 8 | Responsive | Redimensionner le navigateur | Layout s'adapte |
| 9 | Enrollment UI | Enrôler via le dashboard | Photos capturées, personne ajoutée |
| 10 | Performance | Dashboard + flux vidéo | Pas de lag visible |

---

## 📦 Livrables de l'étape

- [ ] `src/dashboard/app.py` — Serveur Flask complet
- [ ] `src/dashboard/templates/` — Templates HTML (5+ pages)
- [ ] `src/dashboard/static/css/` — Styles (dark theme, responsive)
- [ ] `src/dashboard/static/js/` — Scripts (WebSocket, API calls, UI)
- [ ] `src/dashboard/static/assets/` — Logo, icônes, sons
- [ ] API REST complète (8+ endpoints)
- [ ] WebSocket temps réel
- [ ] Tests d'intégration
- [ ] `RESUME_ETAPE_06.md`

---

## ⚠️ Points d'attention

- **MJPEG vs WebSocket** : Commencer par MJPEG (plus simple), migrer vers WebSocket si latence trop élevée.
- **CORS** : Configurer pour permettre le développement local.
- **Sécurité** : Ajouter une authentification basique (au minimum un mot de passe) — le dashboard ne doit pas être accessible sans auth.
- **Mobile** : Le design doit être utilisable sur tablette minimum.
- **Assets** : Créer/générer un logo pour Sentinel-AI.

---

## 🔗 Dépendances pour l'étape suivante

L'étape 7 nécessite :
- ✅ Dashboard fonctionnel affichant le live view
- ✅ API REST complète pour toutes les fonctionnalités
- ✅ WebSocket pour les events temps réel
- ✅ Interface de paramètres pour changer l'URL du LLM
