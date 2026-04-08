# 👁️ ÉTAPE 2 — Pipeline de Perception Visuelle (CPU)

> **Durée estimée : 3-4 jours**
> **Priorité : CRITIQUE**
> **Prérequis : ÉTAPE 1 complétée et validée**

---

## 🎯 Objectif

Implémenter le cœur du système de perception : capture vidéo multi-source, détection d'objets/personnes par YOLO, et suivi (tracking) des entités détectées. Ce pipeline tourne **exclusivement sur CPU** et doit maintenir **≥15 FPS**.

---

## 📋 Tâches détaillées

### 2.1 — Module Caméra (`src/core/camera.py`)

Créer un gestionnaire de caméra polyvalent qui supporte :

| Source | Exemple | Méthode |
|--------|---------|---------|
| Webcam locale | `0`, `1` | `cv2.VideoCapture(0)` |
| Flux RTSP | `rtsp://192.168.1.10:554/stream` | `cv2.VideoCapture(url)` |
| Flux HTTP/IP | `http://192.168.1.10:4747/video` | `cv2.VideoCapture(url)` |
| DroidCam | `http://phone_ip:4747/video` | Via HTTP |
| Fichier vidéo | `./test_video.mp4` | `cv2.VideoCapture(path)` |

**Fonctionnalités obligatoires :**
- Capture dans un **thread dédié** (ne jamais bloquer le thread principal).
- Buffer circulaire de dernières frames (taille configurable, défaut: 30).
- Auto-reconnexion en cas de perte du flux (retry avec backoff exponentiel).
- Métriques en temps réel : FPS réel, latence, frames perdues.
- Méthode `get_frame()` thread-safe qui retourne la dernière frame disponible.
- Méthode `get_snapshot()` qui retourne une frame encodée en JPEG pour l'envoi au LLM.

**Événements émis :**
- `camera_connected` : Quand la caméra est prête.
- `camera_disconnected` : Quand le flux est perdu.
- `camera_error` : Erreur critique.

### 2.2 — Module Détection (`src/core/detector.py`)

Implémenter la détection d'objets et de personnes avec YOLO :

**Architecture :**
```python
class ObjectDetector:
    """
    Détecteur d'objets multi-modèle basé sur Ultralytics YOLO.
    Supporte détection classique et pose estimation.
    """
    def __init__(self, config: Config):
        # Charger le modèle UNE SEULE FOIS
        self.model = YOLO(config.detection.model_path)
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Détecte les objets dans une frame."""
        ...
    
    def detect_persons(self, frame: np.ndarray) -> List[PersonDetection]:
        """Détecte spécifiquement les personnes avec pose."""
        ...
```

**Classe `Detection` (dataclass) :**
```python
@dataclass
class Detection:
    class_id: int          # ID de classe YOLO
    class_name: str        # Nom lisible (person, car, bag, etc.)
    confidence: float      # Score de confiance [0-1]
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]          # Centre du bbox
    frame_id: int          # ID de la frame source
    timestamp: float       # Timestamp de détection
```

**Modèles à supporter :**
- `yolo26n.pt` — Détection rapide (nano) → CPU optimal.
- `yolo26n-pose.pt` — Détection + pose estimation pour analyse gestuelle.
- `yolo11m-pose.pt` — Pose medium (GPU recommandé, fallback CPU).

**Optimisations obligatoires :**
- Inférence à taille réduite (640x640 par défaut, configurable).
- Device auto-detection : GPU si disponible, sinon CPU.
- Skip frames configurable (ex: détecter 1 frame sur 2 si FPS trop bas).

**Événements émis :**
- `person_detected` : Nouvelle personne dans le champ.
- `object_detected` : Objet notable détecté (arme, sac, outil, etc.).
- `scene_empty` : Aucune personne détectée.

### 2.3 — Module Tracking (`src/core/tracker.py`)

Implémenter un tracker basé sur **distance euclidienne + IoU** pour suivre les entités à travers les frames :

**Fonctionnalités :**
- Assignation d'un **ID unique** à chaque personne/objet suivi.
- Gestion du cycle de vie : `APPEARED` → `TRACKED` → `LOST` → `DISAPPEARED`.
- Seuil de disparition configurable (ex: 30 frames sans détection = disparu).
- Comptage des personnes : entrées/sorties, temps de présence dans la zone.

**Classe `TrackedEntity` :**
```python
@dataclass
class TrackedEntity:
    track_id: int
    detections: List[Detection]  # Historique
    status: str                   # APPEARED, TRACKED, LOST, DISAPPEARED
    first_seen: float
    last_seen: float
    total_frames: int
    face_id: Optional[str]       # Lié à la whitelist (étape 3)
```

**Événements émis :**
- `entity_appeared` : Nouvelle entité dans la scène.
- `entity_disappeared` : Entité quittée la scène.
- `entity_lingering` : Entité présente depuis > N secondes (suspecte).

### 2.4 — Visualisation des détections

Créer un module d'overlay pour dessiner sur les frames :

- **Bounding boxes** colorées par classe (vert=personne connue, rouge=inconnue, bleu=objet).
- **Labels** avec nom de classe + confidence.
- **Skeleton/Pose** si modèle pose utilisé.
- **Zone de surveillance** configurable (polygone).
- **HUD info** : FPS, nombre de personnes, timestamp.

### 2.5 — Pipeline intégré

Créer la boucle principale de perception dans un script de test :

```python
# Pseudo-code du pipeline
while running:
    frame = camera.get_frame()
    detections = detector.detect(frame)
    tracked = tracker.update(detections)
    annotated = visualizer.draw(frame, tracked)
    cv2.imshow("Sentinel-AI", annotated)
```

---

## 🧪 Tests de validation (Étape 2)

| # | Test | Description | Résultat attendu |
|---|------|-------------|-------------------|
| 1 | Camera Webcam | Ouvrir webcam ID 0 | Flux vidéo affiché |
| 2 | Camera File | Lire `test_video.mp4` | Vidéo lue frame par frame |
| 3 | Camera Reconnect | Débrancher/rebrancher webcam | Reconnexion auto |
| 4 | Detection Basic | YOLO sur une image test | Détections avec bbox |
| 5 | Detection FPS | Mesurer FPS sur 100 frames | ≥ 15 FPS (CPU) |
| 6 | Tracking IDs | 2 personnes marchant | IDs stables |
| 7 | Tracking Lost | Personne quitte le champ | Status DISAPPEARED |
| 8 | Overlay | Frames annotées | Bboxes + labels visibles |
| 9 | Event emission | Vérifier events émis | Events reçus correctement |

### Tests unitaires obligatoires :
- `tests/test_camera.py` — Test avec mock video.
- `tests/test_detector.py` — Test détection sur images statiques.
- `tests/test_tracker.py` — Test assignation/perte IDs.

---

## 📦 Livrables de l'étape

- [ ] `src/core/camera.py` — Capture multi-source threadée
- [ ] `src/core/detector.py` — Détection YOLO multi-modèle
- [ ] `src/core/tracker.py` — Suivi par ID unique
- [ ] Visualisation avec overlay (bboxes, labels, FPS)
- [ ] Demo script : `scripts/demo_perception.py`
- [ ] Tests unitaires (3 fichiers minimum)
- [ ] `RESUME_ETAPE_02.md`

---

## ⚠️ Points d'attention

- **Thread safety** : La caméra lit dans un thread, le détecteur dans un autre. Utiliser des `Lock` ou `Queue`.
- **Mémoire** : Ne PAS stocker toutes les frames. Buffer circulaire uniquement.
- **GPU** : Si GPU disponible, l'utiliser pour YOLO. Sinon, le code doit fonctionner en CPU pur.
- **Ne PAS implémenter la reconnaissance faciale** — c'est l'étape 3.

---

## 🔗 Dépendances pour l'étape suivante

L'étape 3 nécessite :
- ✅ `camera.py` stable et threadé
- ✅ `detector.py` retournant des `Detection` propres
- ✅ `tracker.py` avec IDs stables
- ✅ Pipeline de test fonctionnel avec overlay
