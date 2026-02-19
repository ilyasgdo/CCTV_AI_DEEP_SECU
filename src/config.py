"""
CCTV AI DEEP SECU — Configuration Centralisée
Toutes les constantes et paramètres du système sont ici.
"""
import os
from pathlib import Path

# === CHEMINS ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
WHITELIST_DIR = DATA_DIR / "whitelist_photos"
DB_PATH = PROJECT_ROOT / "src" / "database" / "cctv_records.db"
STGCN_WEIGHTS = PROJECT_ROOT / "src" / "models" / "stgcn" / "weights"
YOLO_WEIGHTS = PROJECT_ROOT / "src" / "models" / "yolo" / "weights"

# === YOLO-POSE ===
YOLO_MODEL = "yolov8m-pose.pt"  # Medium. Changer en "yolov8l-pose.pt" pour Large
YOLO_CONFIDENCE = 0.5            # Seuil de confiance minimum
YOLO_DEVICE = 0                  # 0 = premier GPU, "cpu" pour CPU

# === BYTETRACK (Suivi) ===
TRACKER_TYPE = "bytetrack"        # Tracker intégré à Ultralytics

# === INSIGHTFACE ===
FACE_RECOGNITION_THRESHOLD = 0.55  # Seuil de similarité cosinus
FACE_RECOGNITION_INTERVAL = 60     # Frames entre chaque scan pour un INCONNU (~2s à 30fps)
FACE_CONFIDENCE_LOCK = 0.95        # Au-dessus, on arrête de scanner

# === ST-GCN ===
STGCN_BUFFER_SIZE = 30            # Nombre de frames dans le buffer temporel
STGCN_INFERENCE_INTERVAL = 5     # Inférence toutes les N frames
STGCN_NUM_KEYPOINTS = 17          # Keypoints COCO (sortie YOLOv8-Pose)
STGCN_IN_CHANNELS = 2             # X, Y (ou 3 si on ajoute la confiance)

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
]

# === ALERTES ===
ALERT_ACTIONS = ["chute", "donner_un_coup"]  # Actions déclenchant une alerte
LOITERING_TIMEOUT = 300            # Secondes avant alerte de maraudage (5 min)
PERSON_LOST_TIMEOUT = 300          # Secondes avant de considérer la personne partie

# === PERFORMANCE ===
TARGET_FPS = 30
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
