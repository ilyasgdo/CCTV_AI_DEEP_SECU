"""
Thread d'analyse : Actions + InsightFace + Objets + SQLite + Stats.
Tourne à fréquence réduite pour économiser le GPU.

- Actions : analyse géométrique toutes les 5 frames
- InsightFace : 1 scan toutes les 2 secondes pour les INCONNUS
- Objets : YOLOv8n toutes les 3 frames sur les crops de personnes
- Stats : temps de présence, actions avec durées, objets détectés
"""
import threading
import time
import numpy as np
from queue import Queue
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import STGCN_INFERENCE_INTERVAL, FACE_RECOGNITION_INTERVAL, DEVICE
from src.behavior.action_classifier import ActionClassifier
from src.behavior.loitering_detector import LoiteringDetector
from src.face_recognition.matcher import FaceMatcher
from src.database.db_manager import DatabaseManager
from src.pipeline.object_detector import ObjectDetector


class PersonTracker:
    """
    Suivi des statistiques par personne en temps réel.
    
    Accumule :
    - Temps de présence total
    - Historique des actions avec durées
    - Détection d'objets basée sur la pose (téléphone, sac, etc.)
    """

    def __init__(self):
        self._persons: Dict[int, dict] = {}

    def _get_or_create(self, track_id: int):
        """Crée ou récupère l'entrée pour une personne."""
        if track_id not in self._persons:
            self._persons[track_id] = {
                "first_seen": time.time(),
                "last_seen": time.time(),
                "name": "INCONNU",
                "current_action": "N/A",
                "action_start": time.time(),
                "action_durations": {},    # {"marcher": 45.2, "immobile": 12.0}
                "pose_objects": [],        # ["telephone", "sac"] — pose-based
                "detected_objects": [],    # ["📱 telephone", "🎒 sac_a_dos"] — YOLO
                "object_durations": {},    # {"telephone": 120.5} — durées cumulées
                "_prev_action": "N/A",
            }
        return self._persons[track_id]

    def update(self, track_id: int, name: str, action: str,
               keypoints: np.ndarray, detected_objects: List[str] = None):
        """
        Met à jour les stats d'une personne.
        
        Args:
            track_id: ID de suivi
            name: Nom identifié
            action: Action courante
            keypoints: Keypoints (17, 3) pour la détection d'objets par pose
            detected_objects: Objets détectés par YOLOv8n
        """
        p = self._get_or_create(track_id)
        now = time.time()
        p["last_seen"] = now
        p["name"] = name

        # --- Suivi des durées d'action ---
        prev_action = p["_prev_action"]
        if action != prev_action:
            if prev_action != "N/A":
                elapsed = now - p["action_start"]
                if prev_action not in p["action_durations"]:
                    p["action_durations"][prev_action] = 0
                p["action_durations"][prev_action] += elapsed
            p["action_start"] = now
            p["_prev_action"] = action
        
        p["current_action"] = action

        # --- Détection d'objets par la pose ---
        p["pose_objects"] = self._detect_objects_by_pose(keypoints)

        # --- Objets détectés par YOLO ---
        if detected_objects:
            p["detected_objects"] = detected_objects
            # Cumuler les durées d'objets
            for obj in detected_objects:
                if obj not in p["object_durations"]:
                    p["object_durations"][obj] = 0
                # Ajouter le temps depuis le dernier update (~1/FPS)
                p["object_durations"][obj] += 0.1  # ~10 updates/sec approx

    def _detect_objects_by_pose(self, kpts: np.ndarray) -> List[str]:
        """
        Détecte des objets et postures basés sur les keypoints (17, 3).
        
        kpts format : (17, 3) avec [x, y, confidence]
        
        Détections :
        - 📱 telephone : main près du visage OU main devant le torse (position texting)
        - ✋ main levee : poignet au-dessus de l'épaule
        - 🤝 bras croises : deux poignets proches au centre du torse
        """
        objects = []

        NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
        L_SHOULDER, R_SHOULDER = 5, 6
        L_ELBOW, R_ELBOW = 7, 8
        L_WRIST, R_WRIST = 9, 10
        L_HIP, R_HIP = 11, 12

        def conf(idx):
            return kpts[idx][2] if idx < len(kpts) else 0

        def pt(idx):
            return kpts[idx][:2]

        def dist(a, b):
            if conf(a) < 0.3 or conf(b) < 0.3:
                return float('inf')
            return np.linalg.norm(pt(a) - pt(b))

        # Référence de taille : distance entre les épaules
        sd = dist(L_SHOULDER, R_SHOULDER)
        if sd == float('inf') or sd < 10:
            return objects

        # ═══════════════════════════════════════════════
        # 📱 TELEPHONE — Deux modes de détection
        # ═══════════════════════════════════════════════

        phone_detected = False

        # Mode 1 : Appel téléphonique (main près de l'oreille)
        for wrist, ear in [(L_WRIST, L_EAR), (R_WRIST, R_EAR)]:
            d = dist(wrist, ear)
            if d < sd * 1.0:
                phone_detected = True
                break

        # Mode 2 : Texting/browsing (main devant le visage/poitrine)
        # Le poignet est entre les épaules et les hanches, et le coude est plié
        if not phone_detected:
            for wrist, elbow, shoulder in [(L_WRIST, L_ELBOW, L_SHOULDER),
                                           (R_WRIST, R_ELBOW, R_SHOULDER)]:
                if conf(wrist) < 0.3 or conf(elbow) < 0.3 or conf(shoulder) < 0.3:
                    continue

                # Vérifier que le poignet est devant le torse (entre épaules et hanches en Y)
                shoulder_y = kpts[shoulder][1]
                hip_y = kpts[L_HIP][1] if conf(L_HIP) > 0.3 else shoulder_y + sd * 2
                wrist_y = kpts[wrist][1]

                # Le poignet doit être entre les épaules et les hanches
                if shoulder_y - sd * 0.3 < wrist_y < hip_y + sd * 0.3:
                    # Vérifier que le coude est plié (angle fermé)
                    upper_arm = dist(shoulder, elbow)
                    forearm = dist(elbow, wrist)
                    full_arm = dist(shoulder, wrist)

                    if upper_arm != float('inf') and forearm != float('inf') and full_arm != float('inf'):
                        # Bras plié si la distance directe < somme des segments
                        if upper_arm + forearm > 0:
                            bend_ratio = full_arm / (upper_arm + forearm)
                            # Coude plié = ratio < 0.85 (bras complètement tendu ≈ 1.0)
                            if bend_ratio < 0.85:
                                # Vérifier que le poignet est centré (pas sur le côté)
                                torso_center_x = (kpts[L_SHOULDER][0] + kpts[R_SHOULDER][0]) / 2
                                wrist_x = kpts[wrist][0]
                                # Poignet pas trop loin du centre du torse
                                if abs(wrist_x - torso_center_x) < sd * 1.5:
                                    phone_detected = True
                                    break

        # Mode 3 : Main proche du nez (regarder le téléphone de près)
        if not phone_detected:
            for wrist in [L_WRIST, R_WRIST]:
                d = dist(wrist, NOSE)
                if d < sd * 1.2:
                    phone_detected = True
                    break

        if phone_detected:
            objects.append("📱 telephone")

        # ═══════════════════════════════════════════════
        # ✋ MAIN LEVÉE
        # ═══════════════════════════════════════════════
        for wrist, shoulder in [(L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)]:
            if conf(wrist) > 0.3 and conf(shoulder) > 0.3:
                if kpts[wrist][1] < kpts[shoulder][1] - sd * 0.4:
                    if "✋ main levee" not in objects:
                        objects.append("✋ main levee")

        # ═══════════════════════════════════════════════
        # 🤝 BRAS CROISÉS
        # ═══════════════════════════════════════════════
        if conf(L_WRIST) > 0.3 and conf(R_WRIST) > 0.3:
            wrist_d = dist(L_WRIST, R_WRIST)
            if wrist_d < sd * 0.6:
                torso_y = (kpts[L_SHOULDER][1] + kpts[R_SHOULDER][1]) / 2
                hip_y = (kpts[L_HIP][1] + kpts[R_HIP][1]) / 2 if conf(L_HIP) > 0.3 and conf(R_HIP) > 0.3 else torso_y + sd * 1.5
                avg_wrist_y = (kpts[L_WRIST][1] + kpts[R_WRIST][1]) / 2
                if torso_y - sd * 0.2 < avg_wrist_y < hip_y + sd * 0.2:
                    objects.append("🤝 bras croises")

        return objects

    def get_stats(self, track_id: int) -> dict:
        """Retourne les stats complètes d'une personne."""
        if track_id not in self._persons:
            return {}
        
        p = self._persons[track_id]
        now = time.time()
        presence = now - p["first_seen"]

        # Ajouter la durée de l'action en cours
        action_durations = dict(p["action_durations"])
        current_action = p["current_action"]
        if current_action != "N/A":
            current_dur = now - p["action_start"]
            if current_action not in action_durations:
                action_durations[current_action] = 0
            action_durations[current_action] += current_dur

        # Top action
        top_action = None
        if action_durations:
            top = max(action_durations.items(), key=lambda x: x[1])
            top_action = (top[0], top[1])

        # Combiner objets pose + YOLO (sans doublons)
        all_objects = list(p.get("detected_objects", []))
        for po in p.get("pose_objects", []):
            if po not in all_objects:
                all_objects.append(po)

        return {
            "name": p["name"],
            "presence_time": presence,
            "current_action": current_action,
            "action_durations": action_durations,
            "top_action": top_action,
            "pose_objects": all_objects,
            "object_durations": p.get("object_durations", {}),
        }

    def get_all_stats(self) -> Dict[int, dict]:
        """Retourne les stats de toutes les personnes actives."""
        return {tid: self.get_stats(tid) for tid in self._persons}

    def cleanup(self, active_ids: set):
        """Ferme les stats des personnes qui ont quitté (garde l'historique)."""
        # On ne supprime PAS les personnes sorties, on les garde pour le panneau
        pass

    def remove_old(self, timeout: float = 30.0):
        """Supprime les personnes pas vues depuis timeout secondes."""
        now = time.time()
        to_remove = [tid for tid, p in self._persons.items()
                     if now - p["last_seen"] > timeout]
        for tid in to_remove:
            del self._persons[tid]


class Analyzer:
    """
    Analyseur combinant ST-GCN, InsightFace, base de données et suivi de stats.
    """

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        """Initialise tous les sous-modules d'analyse."""
        print("[ANALYZER] Initialisation des modules...")

        # Actions (analyse géométrique)
        self.classifier = ActionClassifier(device=str(DEVICE))

        # Maraudage
        self.loitering = LoiteringDetector()
        self.loitering.set_default_zones(frame_width, frame_height)

        # InsightFace
        self.face_matcher = FaceMatcher()
        self.face_matcher.load_whitelist()

        # Détection d'objets (YOLOv8n)
        self.object_detector = ObjectDetector(
            model_name="yolo26n.pt",
            device=DEVICE,
            confidence=0.35,
            detect_interval=3
        )

        # Base de données
        self.db = DatabaseManager()

        # Suivi des stats par personne
        self.person_tracker = PersonTracker()

        # Thread
        self.running = False
        self.thread = None
        self._lock = threading.Lock()

        # Résultats partagés (thread-safe via _lock)
        self.results: Dict[int, dict] = {}

        print("[ANALYZER] Tous les modules initialisés")

    def start(self):
        """Démarre l'analyseur."""
        self.running = True
        print("[ANALYZER] Prêt")
        return self

    def process(self, detections: list, frame: np.ndarray, frame_count: int):
        """
        Traite les détections (appelé depuis le thread principal).
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

                if x2 > x1 + 20 and y2 > y1 + 20:
                    face_crop = frame[y1:y2, x1:x2]
                    fname, fscore = self.face_matcher.identify(
                        face_crop, det.track_id, frame_count
                    )
                    if fname != "INCONNU":
                        self.db.update_name(det.track_id, fname)

        # --- Inférence actions (cadence réduite) ---
        predictions = self.classifier.classify(frame_count)

        # --- Détection d'objets (YOLOv8n, cadence réduite) ---
        obj_results = self.object_detector.detect_for_persons(
            frame, detections, frame_count
        )

        # --- Vérifier maraudage ---
        loitering_alerts = {}
        for det in detections:
            result = self.loitering.update(det.track_id, det.center)
            if result:
                loitering_alerts[det.track_id] = result

        # --- Vérifier alertes actions ---
        action_alerts = self.classifier.check_alerts()

        # --- Mettre à jour les stats par personne ---
        for det in detections:
            tid = det.track_id
            name = self.face_matcher.get_name(tid)
            action = self.classifier.get_action(tid)
            obj_labels = self.object_detector.get_object_labels(tid)
            self.person_tracker.update(tid, name, action, det.keypoints,
                                       detected_objects=obj_labels)

        # --- Alertes objets dangereux ---
        dangerous = self.object_detector.get_all_dangerous()
        for tid, objs in dangerous.items():
            name = self.face_matcher.get_name(tid)
            for obj in objs:
                self.db.log_alert(tid, f"OBJET_DANGEREUX: {obj}", 0.9,
                                  name, frame_count)
                print(f"  🚨 OBJET DANGEREUX : {name} (ID:{tid}) porte {obj}")

        # --- Compiler les résultats ---
        with self._lock:
            for det in detections:
                tid = det.track_id
                self.results[tid] = {
                    "name": self.face_matcher.get_name(tid),
                    "action": self.classifier.get_action(tid),
                    "prediction": self.classifier.get_prediction(tid),
                    "loitering": loitering_alerts.get(tid, None),
                    "objects": self.object_detector.get_object_labels(tid),
                }

        # --- Logger les alertes en BDD ---
        for tid, action, conf in action_alerts:
            name = self.face_matcher.get_name(tid)
            self.db.log_alert(tid, action, conf, name, frame_count)

        for tid, (alert_type, duration) in loitering_alerts.items():
            name = self.face_matcher.get_name(tid)
            self.db.log_alert(tid, alert_type, duration, name, frame_count)

        # --- Nettoyage ---
        self.classifier.cleanup_lost_ids(active_ids)
        self.face_matcher.cleanup_lost_ids(active_ids)
        self.loitering.cleanup_lost_ids(active_ids)
        self.object_detector.cleanup_lost_ids(active_ids)

        # Supprimer les personnes pas vues depuis 30s du tracker
        self.person_tracker.remove_old(timeout=30.0)

        # Vérifier les sorties en BDD (toutes les 30 frames)
        if frame_count % 30 == 0:
            self.db.check_exits()

    def get_results(self) -> Dict[int, dict]:
        """Retourne les résultats d'analyse (thread-safe)."""
        with self._lock:
            return self.results.copy()

    def get_person_stats(self) -> Dict[int, dict]:
        """Retourne les statistiques par personne pour l'affichage."""
        return self.person_tracker.get_all_stats()

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
            "objects": self.object_detector.get_stats(),
            "database": self.db.get_stats(),
        }
