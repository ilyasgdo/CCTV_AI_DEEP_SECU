"""
Module de comparaison des visages.
Compare un visage détecté aux embeddings de la liste blanche.
Intègre la stratégie "paresseuse" pour économiser le GPU.

Compatible avec InsightFace v0.2.x via encodeur SCRFD + ArcFace direct.
"""
import numpy as np
import cv2
import time
from typing import Optional, Dict, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    FACE_RECOGNITION_THRESHOLD,
    FACE_RECOGNITION_INTERVAL,
    FACE_CONFIDENCE_LOCK
)
from src.face_recognition.encoder import FaceEncoder


class FaceMatcher:
    """
    Gère la reconnaissance faciale avec stratégie paresseuse.

    Stratégie :
    1. Ne scanner que les INCONNUS
    2. Scanner 1 fois toutes les FACE_RECOGNITION_INTERVAL frames
    3. Arrêter de scanner une fois identifié avec >= FACE_CONFIDENCE_LOCK
    """

    def __init__(self):
        """Initialise le matcher."""
        self.encoder = FaceEncoder()
        self.whitelist: Dict[str, np.ndarray] = {}
        self._identified: Dict[int, Tuple[str, float]] = {}  # {track_id: (nom, score)}
        self._last_scan: Dict[int, int] = {}  # {track_id: dernier frame scanné}

    def load_whitelist(self):
        """Charge la liste blanche depuis les fichiers .npy."""
        self.whitelist = self.encoder.load_whitelist()
        if not self.whitelist:
            print("⚠ [MATCHER] Liste blanche vide ! Lancer d'abord l'encodage.")

    def should_scan(self, track_id: int, frame_count: int) -> bool:
        """
        Détermine si on doit scanner le visage de cette personne.

        Règles :
        - Si déjà identifié avec haute confiance → NON
        - Si scanné trop récemment → NON
        - Sinon → OUI
        """
        # Déjà identifié avec certitude ?
        if track_id in self._identified:
            _, score = self._identified[track_id]
            if score >= FACE_CONFIDENCE_LOCK:
                return False

        # Scanné trop récemment ?
        if track_id in self._last_scan:
            if (frame_count - self._last_scan[track_id]) < FACE_RECOGNITION_INTERVAL:
                return False

        return True

    def identify(self, face_crop: np.ndarray, track_id: int,
                 frame_count: int) -> Tuple[str, float]:
        """
        Identifie un visage par comparaison avec la liste blanche.

        Args:
            face_crop: Image recadrée du visage (BGR)
            track_id: ID de suivi de la personne
            frame_count: Numéro de frame actuel

        Returns:
            (nom, score) : ("Thomas", 0.87) ou ("INCONNU", 0.0)
        """
        # Vérifier si on doit scanner
        if not self.should_scan(track_id, frame_count):
            if track_id in self._identified:
                return self._identified[track_id]
            return ("INCONNU", 0.0)

        # Marquer le scan
        self._last_scan[track_id] = frame_count

        # Détecter et encoder le visage dans le crop
        results = self.encoder.detect_and_encode(face_crop)
        if len(results) == 0:
            if track_id in self._identified:
                return self._identified[track_id]
            return ("INCONNU", 0.0)

        # Prendre le premier visage encodé
        _, query_emb = results[0]

        # Comparer avec la whitelist
        best_name = "INCONNU"
        best_score = 0.0

        for name, ref_emb in self.whitelist.items():
            # Similarité cosinus
            score = float(np.dot(query_emb, ref_emb))
            if score > best_score:
                best_score = score
                best_name = name

        # Seuil de reconnaissance
        if best_score < FACE_RECOGNITION_THRESHOLD:
            result = ("INCONNU", best_score)
        else:
            result = (best_name, best_score)
            self._identified[track_id] = result
            print(f"  🔍 ID:{track_id} identifié comme {best_name} "
                  f"(score: {best_score:.3f})")

        return result

    def get_name(self, track_id: int) -> str:
        """Retourne le nom connu d'un track_id, ou 'INCONNU'."""
        if track_id in self._identified:
            return self._identified[track_id][0]
        return "INCONNU"

    def cleanup_lost_ids(self, active_ids: set):
        """Nettoie les IDs qui ne sont plus suivis."""
        lost_ids = set(self._identified.keys()) - active_ids
        for lost_id in lost_ids:
            del self._identified[lost_id]
            if lost_id in self._last_scan:
                del self._last_scan[lost_id]

    def get_stats(self) -> dict:
        """Retourne les statistiques du matcher."""
        return {
            "whitelist_size": len(self.whitelist),
            "identified_count": len(self._identified),
            "identified_persons": {
                tid: (name, f"{score:.3f}")
                for tid, (name, score) in self._identified.items()
            }
        }
