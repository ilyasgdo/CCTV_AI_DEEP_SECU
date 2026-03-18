"""
CCTV AI DEEP SECU — Configuration Centralisée
Toutes les constantes et paramètres du système sont ici.
"""
import os
import platform
from pathlib import Path

# === CHEMINS ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
WHITELIST_DIR = DATA_DIR / "whitelist_photos"
DB_PATH = PROJECT_ROOT / "src" / "database" / "cctv_records.db"
STGCN_WEIGHTS = PROJECT_ROOT / "src" / "models" / "stgcn" / "weights"
YOLO_WEIGHTS = PROJECT_ROOT / "src" / "models" / "yolo" / "weights"

# === DÉTECTION AUTOMATIQUE DU GPU ===
def _detect_device():
    """Détecte le meilleur device : CUDA (NVIDIA) > MPS (Apple M) > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0  # NVIDIA GPU
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
    except ImportError:
        pass
    return "cpu"

DEVICE = _detect_device()

# === YOLO-POSE ===
YOLO_MODEL = "yolo11m-pose.pt"  # YOLO11 Medium Pose — meilleur que v8 (+2% mAP)
YOLO_CONFIDENCE = 0.5            # Seuil de confiance minimum
YOLO_DEVICE = DEVICE             # Auto-détecté : 0 (CUDA), "mps" (Apple M), "cpu"

# === BYTETRACK (Suivi) ===
TRACKER_TYPE = "bytetrack"        # Tracker intégré à Ultralytics

# === INSIGHTFACE ===
FACE_RECOGNITION_THRESHOLD = 0.45  # Seuil de similarité cosinus
FACE_RECOGNITION_INTERVAL = 15     # Frames entre chaque scan pour un INCONNU (~3s à 5fps)
FACE_CONFIDENCE_LOCK = 0.60        # Au-dessus, on arrête de scanner

# === ST-GCN ===
STGCN_BUFFER_SIZE = 30            # Nombre de frames dans le buffer temporel
STGCN_INFERENCE_INTERVAL = 5     # Inférence toutes les N frames
STGCN_NUM_KEYPOINTS = 17          # Keypoints COCO (sortie YOLOv8-Pose)
STGCN_IN_CHANNELS = 3             # X, Y, Confidence

# === ACTIONS RECONNUES ===
ACTION_LABELS = [
    "marcher",
    "courir",
    "s'asseoir",
    "se_lever",
    "chute",
    "donner_un_coup",
    "immobile",
    "se_pencher",
    "mains_en_l_air",
]

# === ALERTES ===
ALERT_ACTIONS = ["chute", "donner_un_coup", "mains_en_l_air", "courir"]
LOITERING_TIMEOUT = 300            # Secondes avant alerte de maraudage (5 min)
PERSON_LOST_TIMEOUT = 300          # Secondes avant de considérer la personne partie

# === PERFORMANCE ===
TARGET_FPS = 30
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# === AI GUARD (Agent de Sécurité IA) ===
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
GUARD_STT_MODEL = "base"           # Whisper model: tiny, base, small, medium
GUARD_LISTEN_DURATION = 5          # Secondes d'écoute micro
GUARD_TRIGGER_INTERVAL = 15       # Secondes min entre interventions

# === VISION MODEL (Analyse de frames) ===
VISION_MODEL = "moondream"          # Modèle vision Ollama (léger, 1.8B)
VISION_ANALYSIS_INTERVAL = 10      # Secondes entre chaque analyse de frame
VISION_ENABLED = True              # Activer/désactiver l'analyse de frames
