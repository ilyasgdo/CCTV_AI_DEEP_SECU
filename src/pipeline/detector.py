"""
Module de détection, suivi et extraction squelettique.
Utilise YOLOv8-Pose avec ByteTrack pour un pipeline unifié.
"""
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    YOLO_MODEL, YOLO_CONFIDENCE, YOLO_DEVICE, TRACKER_TYPE
)


@dataclass
class PersonDetection:
    """Représente une personne détectée dans une frame."""
    track_id: int                          # ID de suivi ByteTrack
    bbox: np.ndarray                       # [x1, y1, x2, y2]
    confidence: float                      # Confiance de détection
    keypoints: np.ndarray                  # (17, 3) → [x, y, conf]
    keypoints_xy: np.ndarray               # (17, 2) → [x, y] uniquement
    name: str = "INCONNU"                  # Sera rempli par InsightFace (Étape 2)
    action: str = "N/A"                    # Sera rempli par ST-GCN (Étape 4)

    @property
    def head_bbox(self) -> np.ndarray:
        """Retourne la boîte englobante du tiers supérieur (pour InsightFace)."""
        x1, y1, x2, y2 = self.bbox
        head_height = (y2 - y1) / 3
        return np.array([x1, y1, x2, y1 + head_height])

    @property
    def center(self) -> tuple:
        """Retourne le centre de la boîte englobante."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class PoseDetector:
    """
    Détecteur unifié YOLOv8-Pose + ByteTrack.
    Sortie : liste de PersonDetection par frame.
    """

    def __init__(self, model_path: str = YOLO_MODEL, device: int = YOLO_DEVICE):
        """
        Initialise le détecteur.

        Args:
            model_path: Chemin vers le modèle YOLOv8-Pose
            device: Index du GPU (0) ou "cpu"
        """
        print(f"[DETECTOR] Chargement du modèle {model_path}...")
        self.model = YOLO(model_path)
        self.device = device
        self.frame_count = 0
        print(f"[DETECTOR] Modèle chargé avec succès sur device={device}")

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Détecte les personnes, suit leur ID, et extrait les squelettes.

        Args:
            frame: Image BGR (numpy array) depuis OpenCV

        Returns:
            Liste de PersonDetection pour chaque personne détectée
        """
        self.frame_count += 1
        detections: List[PersonDetection] = []

        # Inférence YOLOv8-Pose avec suivi ByteTrack
        results = self.model.track(
            source=frame,
            persist=True,          # Maintenir le suivi entre les frames
            tracker=f"{TRACKER_TYPE}.yaml",
            conf=YOLO_CONFIDENCE,
            device=self.device,
            verbose=False          # Pas de log à chaque frame
        )

        if results is None or len(results) == 0:
            return detections

        result = results[0]

        # Vérifier que des personnes ont été détectées
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # Vérifier que le suivi est actif (IDs disponibles)
        if result.boxes.id is None:
            return detections

        # Extraire les données
        boxes = result.boxes.xyxy.cpu().numpy()                    # (N, 4)
        confs = result.boxes.conf.cpu().numpy()                    # (N,)
        track_ids = result.boxes.id.cpu().numpy().astype(int)      # (N,)

        # Keypoints
        if result.keypoints is not None:
            kpts_data = result.keypoints.data.cpu().numpy()        # (N, 17, 3)
            kpts_xy = result.keypoints.xy.cpu().numpy()            # (N, 17, 2)
        else:
            return detections

        # Créer les objets PersonDetection
        for i in range(len(boxes)):
            detection = PersonDetection(
                track_id=int(track_ids[i]),
                bbox=boxes[i],
                confidence=float(confs[i]),
                keypoints=kpts_data[i],
                keypoints_xy=kpts_xy[i]
            )
            detections.append(detection)

        return detections

    def get_stats(self) -> dict:
        """Retourne les statistiques du détecteur."""
        return {
            "frames_processed": self.frame_count,
            "model": str(self.model.model_name),
            "device": self.device
        }
