"""
Thread d'analyse : ST-GCN + InsightFace + SQLite.
Tourne à fréquence réduite pour économiser le GPU.

- ST-GCN : 1 inférence toutes les 5 frames
- InsightFace : 1 scan toutes les 2 secondes pour les INCONNUS
"""
import threading
import time
import numpy as np
from queue import Queue
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import STGCN_INFERENCE_INTERVAL, FACE_RECOGNITION_INTERVAL
from src.behavior.action_classifier import ActionClassifier
from src.behavior.loitering_detector import LoiteringDetector
from src.face_recognition.matcher import FaceMatcher
from src.database.db_manager import DatabaseManager


class Analyzer:
    """
    Thread d'analyse combinant ST-GCN, InsightFace et la base de données.

    Fonctionne en mode asynchrone : reçoit les détections via une queue,
    et met à jour les résultats dans des dictionnaires thread-safe.
    """

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        """
        Initialise tous les sous-modules d'analyse.
        """
        print("[ANALYZER] Initialisation des modules...")

        # ST-GCN
        self.classifier = ActionClassifier(device="cuda")

        # Maraudage
        self.loitering = LoiteringDetector()
        self.loitering.set_default_zones(frame_width, frame_height)

        # InsightFace
        self.face_matcher = FaceMatcher()
        self.face_matcher.load_whitelist()

        # Base de données
        self.db = DatabaseManager()

        # Thread
        self.running = False
        self.thread = None
        self._lock = threading.Lock()

        # Résultats partagés (thread-safe via _lock)
        self.results: Dict[int, dict] = {}  # {track_id: {name, action, alerts}}

        print("[ANALYZER] Tous les modules initialisés")

    def start(self):
        """Démarre l'analyseur."""
        self.running = True
        print("[ANALYZER] Prêt")
        return self

    def process(self, detections: list, frame: np.ndarray, frame_count: int):
        """
        Traite les détections (appelé depuis le thread principal).

        Args:
            detections: Liste de PersonDetection
            frame: Frame courante (pour le crop du visage)
            frame_count: Numéro de frame
        """
        active_ids = {d.track_id for d in detections}

        for det in detections:
            # --- Mise à jour ST-GCN buffer ---
            self.classifier.update(det.track_id, det.keypoints_xy)

            # --- Mise à jour BDD ---
            name = self.face_matcher.get_name(det.track_id)
            self.db.update_presence(det.track_id, name)

            # --- Reconnaissance faciale (paresseuse) ---
            if self.face_matcher.should_scan(det.track_id, frame_count):
                head = det.head_bbox.astype(int)
                x1 = max(0, head[0])
                y1 = max(0, head[1])
                x2 = min(frame.shape[1], head[2])
                y2 = min(frame.shape[0], head[3])

                if x2 > x1 + 20 and y2 > y1 + 20:  # Min 20px
                    face_crop = frame[y1:y2, x1:x2]
                    fname, fscore = self.face_matcher.identify(
                        face_crop, det.track_id, frame_count
                    )
                    if fname != "INCONNU":
                        self.db.update_name(det.track_id, fname)

        # --- Inférence ST-GCN (cadence réduite) ---
        predictions = self.classifier.classify(frame_count)

        # --- Vérifier maraudage ---
        loitering_alerts = {}
        for det in detections:
            result = self.loitering.update(det.track_id, det.center)
            if result:
                loitering_alerts[det.track_id] = result

        # --- Vérifier alertes ST-GCN ---
        stgcn_alerts = self.classifier.check_alerts()

        # --- Compiler les résultats ---
        with self._lock:
            for det in detections:
                tid = det.track_id
                self.results[tid] = {
                    "name": self.face_matcher.get_name(tid),
                    "action": self.classifier.get_action(tid),
                    "prediction": self.classifier.get_prediction(tid),
                    "loitering": loitering_alerts.get(tid, None),
                }

        # --- Logger les alertes en BDD ---
        for tid, action, conf in stgcn_alerts:
            name = self.face_matcher.get_name(tid)
            self.db.log_alert(tid, action, conf, name, frame_count)

        for tid, (alert_type, duration) in loitering_alerts.items():
            name = self.face_matcher.get_name(tid)
            self.db.log_alert(tid, alert_type, duration, name, frame_count)

        # --- Nettoyage ---
        self.classifier.cleanup_lost_ids(active_ids)
        self.face_matcher.cleanup_lost_ids(active_ids)
        self.loitering.cleanup_lost_ids(active_ids)

        # Vérifier les sorties (toutes les 30 frames)
        if frame_count % 30 == 0:
            self.db.check_exits()

    def get_results(self) -> Dict[int, dict]:
        """Retourne les résultats d'analyse (thread-safe)."""
        with self._lock:
            return self.results.copy()

    def apply_to_detections(self, detections: list):
        """Applique les résultats d'analyse aux détections."""
        results = self.get_results()
        for det in detections:
            if det.track_id in results:
                r = results[det.track_id]
                det.name = r["name"]
                det.action = r["action"]
                if r["loitering"]:
                    det.action = f"MARAUDAGE ({r['loitering'][1]:.0f}s)"

    def stop(self):
        """Arrête l'analyseur."""
        self.running = False
        self.db.close()
        print("[ANALYZER] Arrêté")

    def get_stats(self) -> dict:
        """Statistiques complètes."""
        return {
            "classifier": self.classifier.get_stats(),
            "face_matcher": self.face_matcher.get_stats(),
            "loitering": self.loitering.get_stats(),
            "database": self.db.get_stats(),
        }
