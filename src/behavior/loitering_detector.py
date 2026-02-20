"""
Détecteur de maraudage (loitering).
Vérifie si une personne reste trop longtemps dans une zone définie
par un polygone.

Le ST-GCN ne gère pas le temps → cette règle spatiale le complète.
"""
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import LOITERING_TIMEOUT


class LoiteringDetector:
    """
    Détecte le maraudage basé sur le temps passé dans une zone.

    Algorithme :
    1. Définir un (ou plusieurs) polygone(s) de surveillance
    2. Pour chaque personne, vérifier si son centre est dans le polygone
    3. Si elle y reste plus de LOITERING_TIMEOUT → alerte MARAUDAGE
    """

    def __init__(self, timeout: float = LOITERING_TIMEOUT):
        """
        Args:
            timeout: Temps en secondes avant alerte de maraudage
        """
        self.timeout = timeout
        self.zones: List[np.ndarray] = []  # Liste de polygones

        # Suivi du temps par personne
        # {track_id: {"zone_idx": zone_index, "enter_time": float}}
        self._tracking: Dict[int, Dict] = {}

    def add_zone(self, polygon: list):
        """
        Ajoute une zone de surveillance.

        Args:
            polygon: Liste de points [(x1,y1), (x2,y2), ...]
                     définissant le polygone
        """
        poly = np.array(polygon, dtype=np.int32)
        self.zones.append(poly)
        print(f"[LOITERING] Zone ajoutée ({len(poly)} points)")

    def set_default_zones(self, frame_width: int, frame_height: int):
        """
        Crée une zone par défaut couvrant le centre de l'image.
        Utile pour les tests. En production, définir les zones manuellement.
        """
        # Zone centrale (60% de l'image)
        margin_x = int(frame_width * 0.2)
        margin_y = int(frame_height * 0.2)
        default_zone = [
            (margin_x, margin_y),
            (frame_width - margin_x, margin_y),
            (frame_width - margin_x, frame_height - margin_y),
            (margin_x, frame_height - margin_y),
        ]
        self.add_zone(default_zone)

    def is_in_zone(self, point: tuple, zone_idx: int = 0) -> bool:
        """Vérifie si un point est dans la zone spécifiée."""
        if zone_idx >= len(self.zones):
            return False
        result = cv2.pointPolygonTest(
            self.zones[zone_idx], (float(point[0]), float(point[1])), False
        )
        return result >= 0

    def update(self, track_id: int, center: tuple) -> Optional[Tuple[str, float]]:
        """
        Met à jour le suivi de maraudage pour une personne.

        Args:
            track_id: ID de suivi
            center: (x, y) centre de la personne

        Returns:
            ("MARAUDAGE", durée_en_secondes) si alerte, ou None
        """
        now = time.time()

        # Vérifier chaque zone
        in_any_zone = False
        for zone_idx, zone in enumerate(self.zones):
            if self.is_in_zone(center, zone_idx):
                in_any_zone = True

                if track_id not in self._tracking:
                    self._tracking[track_id] = {
                        "zone_idx": zone_idx,
                        "enter_time": now
                    }
                else:
                    duration = now - self._tracking[track_id]["enter_time"]
                    if duration >= self.timeout:
                        return ("MARAUDAGE", duration)
                break

        # Si pas dans une zone, réinitialiser le compteur
        if not in_any_zone and track_id in self._tracking:
            del self._tracking[track_id]

        return None

    def cleanup_lost_ids(self, active_ids: set):
        """Nettoie les IDs perdus."""
        lost = set(self._tracking.keys()) - active_ids
        for tid in lost:
            del self._tracking[tid]

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Dessine les zones de surveillance sur la frame."""
        overlay = frame.copy()
        for zone in self.zones:
            cv2.polylines(overlay, [zone], True, (0, 200, 255), 2)
            # Zone semi-transparente
            cv2.fillPoly(overlay, [zone], (0, 200, 255))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        return frame

    def get_stats(self) -> dict:
        """Statistiques du détecteur."""
        return {
            "num_zones": len(self.zones),
            "tracked_persons": len(self._tracking),
            "durations": {
                tid: f"{time.time() - info['enter_time']:.0f}s"
                for tid, info in self._tracking.items()
            }
        }
