"""
Détecteur d'objets portés/utilisés par chaque personne.

Utilise YOLOv8n (nano) pour détecter les 80 classes COCO
à l'intérieur de la boîte englobante de chaque personne.

Objets détectés pertinents pour la vidéosurveillance :
  📱 telephone, 💼 sac_a_main, 🎒 sac_a_dos, 🧳 valise
  🔪 couteau, ✂️ ciseaux, 💻 ordinateur, 📕 livre
  🍼 bouteille, ☂️ parapluie, 🥤 tasse, etc.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

# Mapping des classes COCO pertinentes → labels français + emojis
OBJECT_LABELS = {
    # Objets portables / personnels
    "cell phone":   "📱 telephone",
    "handbag":      "👜 sac_a_main",
    "backpack":     "🎒 sac_a_dos",
    "suitcase":     "🧳 valise",
    "umbrella":     "☂️ parapluie",
    "tie":          "👔 cravate",

    # Objets dangereux
    "knife":        "🔪 couteau",
    "scissors":     "✂️ ciseaux",
    "baseball bat": "🏏 batte",

    # Électronique
    "laptop":       "💻 ordinateur",
    "remote":       "🎮 telecommande",
    "keyboard":     "⌨️ clavier",
    "mouse":        "🖱️ souris",

    # Nourriture / boissons
    "bottle":       "🍼 bouteille",
    "cup":          "🥤 tasse",
    "wine glass":   "🍷 verre",
    "fork":         "🍴 fourchette",
    "spoon":        "🥄 cuillere",
    "bowl":         "🥣 bol",
    "banana":       "🍌 banane",
    "apple":        "🍎 pomme",
    "sandwich":     "🥪 sandwich",
    "pizza":        "🍕 pizza",
    "donut":        "🍩 donut",
    "cake":         "🎂 gateau",

    # Activités
    "sports ball":  "⚽ ballon",
    "tennis racket":"🎾 raquette",
    "skateboard":   "🛹 skateboard",
    "surfboard":    "🏄 surf",
    "frisbee":      "🥏 frisbee",
    "skis":         "🎿 skis",
    "snowboard":    "🏂 snowboard",

    # Accessoires
    "book":         "📕 livre",
    "clock":        "🕐 horloge",
    "vase":         "🏺 vase",
    "toothbrush":   "🪥 brosse_a_dent",
    "hair drier":   "💇 seche_cheveux",
}

# Objets qui déclenchent une alerte de sécurité
DANGEROUS_OBJECTS = {"knife", "scissors", "baseball bat"}


class ObjectDetector:
    """
    Détecte les objets portés/utilisés par chaque personne.
    
    Utilise YOLOv8n sur les crops des personnes détectées.
    Tourne à fréquence réduite pour économiser le GPU.
    """

    def __init__(self, model_name: str = "yolov8n.pt", device: int = 0,
                 confidence: float = 0.35, detect_interval: int = 3):
        """
        Args:
            model_name: Modèle YOLO pour la détection d'objets
            device: GPU device (0) ou "cpu"
            confidence: Seuil de confiance minimum
            detect_interval: Fréquence de détection (toutes les N frames)
        """
        print(f"[OBJECTS] Chargement du modèle {model_name}...")
        self.model = YOLO(model_name)
        self.device = device
        self.confidence = confidence
        self.detect_interval = detect_interval

        # Cache des dernières détections par personne
        self._cache: Dict[int, List[dict]] = {}
        # Objets cumulés par personne (pour le suivi)
        self._person_objects: Dict[int, Dict[str, float]] = {}  # {tid: {obj: last_seen_time}}

        print(f"[OBJECTS] Modèle chargé (80 classes COCO, seuil={confidence})")

    def detect_for_persons(self, frame: np.ndarray, detections: list,
                           frame_count: int) -> Dict[int, List[dict]]:
        """
        Détecte les objets dans la boîte de chaque personne.

        Args:
            frame: Frame complète BGR
            detections: Liste de PersonDetection
            frame_count: Numéro de frame

        Returns:
            Dict {track_id: [{name, label, confidence, bbox_in_crop}, ...]}
        """
        # Cadence réduite
        if frame_count % self.detect_interval != 0:
            return self._cache

        results = {}

        for det in detections:
            tid = det.track_id
            x1, y1, x2, y2 = det.bbox.astype(int)

            # Étendre légèrement la boîte (10%)
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            # Crop de la personne
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.shape[0] < 30 or crop.shape[1] < 30:
                continue

            # Détection d'objets dans le crop
            yolo_results = self.model(
                crop,
                conf=self.confidence,
                device=self.device,
                verbose=False,
                classes=None  # Toutes les classes
            )

            person_objects = []
            if yolo_results and len(yolo_results) > 0:
                r = yolo_results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    for i in range(len(r.boxes)):
                        cls_id = int(r.boxes.cls[i])
                        cls_name = r.names[cls_id]
                        conf = float(r.boxes.conf[i])

                        # Filtrer : garder seulement les objets pertinents
                        if cls_name in OBJECT_LABELS:
                            obj_bbox = r.boxes.xyxy[i].cpu().numpy()
                            person_objects.append({
                                "name": cls_name,
                                "label": OBJECT_LABELS[cls_name],
                                "confidence": conf,
                                "bbox": obj_bbox,
                                "dangerous": cls_name in DANGEROUS_OBJECTS,
                            })

                            # Mettre à jour le suivi temporel
                            if tid not in self._person_objects:
                                self._person_objects[tid] = {}
                            self._person_objects[tid][cls_name] = time.time()

            results[tid] = person_objects

        self._cache = results
        return results

    def get_object_labels(self, track_id: int) -> List[str]:
        """Retourne les labels des objets détectés pour une personne."""
        if track_id not in self._cache:
            return []
        return [obj["label"] for obj in self._cache[track_id]]

    def get_dangerous_objects(self, track_id: int) -> List[str]:
        """Retourne les objets dangereux détectés."""
        if track_id not in self._cache:
            return []
        return [obj["label"] for obj in self._cache[track_id] if obj["dangerous"]]

    def get_all_dangerous(self) -> Dict[int, List[str]]:
        """Retourne tous les objets dangereux par personne."""
        result = {}
        for tid, objects in self._cache.items():
            dangerous = [obj["label"] for obj in objects if obj["dangerous"]]
            if dangerous:
                result[tid] = dangerous
        return result

    def get_object_history(self, track_id: int) -> Dict[str, float]:
        """Retourne l'historique des objets avec la dernière détection."""
        return self._person_objects.get(track_id, {})

    def cleanup_lost_ids(self, active_ids: set):
        """Nettoie les IDs perdus."""
        lost = set(self._cache.keys()) - active_ids
        for tid in lost:
            if tid in self._cache:
                del self._cache[tid]

        # Garder l'historique plus longtemps (60s)
        now = time.time()
        for tid in list(self._person_objects.keys()):
            if tid not in active_ids:
                old = {k: v for k, v in self._person_objects[tid].items()
                       if now - v > 60}
                if len(old) == len(self._person_objects[tid]):
                    del self._person_objects[tid]

    def get_stats(self) -> dict:
        return {
            "cached_persons": len(self._cache),
            "total_tracked": len(self._person_objects),
            "detect_interval": self.detect_interval,
        }
