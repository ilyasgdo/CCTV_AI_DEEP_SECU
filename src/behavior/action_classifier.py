"""
Classificateur d'actions basé sur l'analyse géométrique des poses.

Utilise des règles géométriques sur les 17 keypoints COCO pour détecter
les actions EN TEMPS RÉEL, sans nécessiter un modèle entraîné.

Actions détectées :
  - immobile : peu de mouvement
  - marcher : mouvement régulier des jambes
  - courir : mouvement rapide, enjambées larges
  - s'asseoir : hanches proches des genoux, genoux pliés
  - chute ⚠ : chute rapide du centre de gravité
  - donner_un_coup ⚠ : bras tendu rapidement
  - se_pencher : incliné vers l'avant
  - mains_en_l_air : les deux mains au-dessus de la tête
"""
import numpy as np
import time
from collections import deque
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    STGCN_INFERENCE_INTERVAL, ALERT_ACTIONS,
    ACTION_LABELS, STGCN_BUFFER_SIZE
)

# Indices COCO 17 keypoints
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


class PoseAnalyzer:
    """
    Analyse une séquence de poses pour détecter les actions.
    Utilise un buffer de positions historiques par personne.
    """

    def __init__(self, buffer_size: int = 15):
        self.buffer_size = buffer_size
        # {track_id: deque of (keypoints_xy, timestamp)}
        self._history: Dict[int, deque] = {}
        # {track_id: prev body height for fall detection}
        self._prev_heights: Dict[int, deque] = {}

    def update(self, track_id: int, kpts: np.ndarray):
        """
        Ajoute un frame de keypoints au buffer.
        kpts: (17, 2) array de coordonnées [x, y]
        """
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.buffer_size)
            self._prev_heights[track_id] = deque(maxlen=self.buffer_size)

        self._history[track_id].append((kpts.copy(), time.time()))

        # Calculer la hauteur du corps (position Y du nez ou de la moyenne épaules)
        if kpts[NOSE][0] > 0 and kpts[NOSE][1] > 0:
            h = kpts[NOSE][1]
        elif kpts[L_SHOULDER][0] > 0:
            h = kpts[L_SHOULDER][1]
        else:
            h = 0
        self._prev_heights[track_id].append(h)

    def _dist(self, kpts, a, b):
        """Distance euclidienne entre deux keypoints."""
        if kpts[a][0] == 0 or kpts[b][0] == 0:
            return float('inf')
        return np.sqrt((kpts[a][0] - kpts[b][0])**2 + (kpts[a][1] - kpts[b][1])**2)

    def _body_height(self, kpts):
        """Taille estimée du corps (distance nez-cheville)."""
        top = kpts[NOSE] if kpts[NOSE][0] > 0 else kpts[L_SHOULDER]
        bottom_l = kpts[L_ANKLE] if kpts[L_ANKLE][0] > 0 else kpts[L_KNEE]
        bottom_r = kpts[R_ANKLE] if kpts[R_ANKLE][0] > 0 else kpts[R_KNEE]

        if top[0] == 0:
            return 0

        bl = np.sqrt((top[0] - bottom_l[0])**2 + (top[1] - bottom_l[1])**2) if bottom_l[0] > 0 else 0
        br = np.sqrt((top[0] - bottom_r[0])**2 + (top[1] - bottom_r[1])**2) if bottom_r[0] > 0 else 0

        return max(bl, br)

    def _shoulder_dist(self, kpts):
        """Distance entre les épaules (référence de taille)."""
        d = self._dist(kpts, L_SHOULDER, R_SHOULDER)
        return d if d != float('inf') else 100

    def _movement_speed(self, track_id: int) -> float:
        """Calcule la vitesse de déplacement du centre de gravité (pixels/frame)."""
        hist = self._history.get(track_id)
        if not hist or len(hist) < 3:
            return 0.0

        positions = []
        for kpts, _ in list(hist)[-5:]:
            # Centre de gravité = moyenne des hanches
            cx = (kpts[L_HIP][0] + kpts[R_HIP][0]) / 2
            cy = (kpts[L_HIP][1] + kpts[R_HIP][1]) / 2
            if cx > 0 and cy > 0:
                positions.append((cx, cy))

        if len(positions) < 2:
            return 0.0

        # Déplacement total
        total = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total += np.sqrt(dx**2 + dy**2)

        return total / len(positions)

    def _leg_spread(self, kpts) -> float:
        """Écartement des jambes (pour détecter la marche/course)."""
        return self._dist(kpts, L_ANKLE, R_ANKLE)

    def analyze(self, track_id: int) -> Tuple[str, float, List[str]]:
        """
        Analyse la pose et retourne (action, confiance, alertes).

        Returns:
            (action_name, confidence, [alert_messages])
        """
        hist = self._history.get(track_id)
        if not hist or len(hist) < 2:
            return ("N/A", 0.0, [])

        kpts = hist[-1][0]  # Derniers keypoints
        alerts = []
        sd = self._shoulder_dist(kpts)
        speed = self._movement_speed(track_id)

        # === CHUTE (ALERTE) ===
        if self._detect_fall(track_id, kpts, sd):
            alerts.append("⚠️ CHUTE DÉTECTÉE")
            return ("chute", 0.95, alerts)

        # === COUP DE POING (ALERTE) ===
        if self._detect_punch(track_id, kpts, sd):
            alerts.append("⚠️ COUP DÉTECTÉ")
            return ("donner_un_coup", 0.85, alerts)

        # === MAINS EN L'AIR (SUSPECT) ===
        if self._detect_hands_up(kpts, sd):
            alerts.append("⚠️ MAINS EN L'AIR")
            return ("mains_en_l_air", 0.90, alerts)

        # === COURIR ===
        if speed > sd * 0.25 and self._leg_spread(kpts) > sd * 1.2:
            return ("courir", 0.80, [])

        # === SE PENCHER ===
        if self._detect_bending(kpts, sd):
            return ("se_pencher", 0.75, [])

        # === S'ASSEOIR ===
        if self._detect_sitting(kpts, sd):
            return ("s'asseoir", 0.80, [])

        # === MARCHER ===
        if speed > sd * 0.08:
            return ("marcher", 0.70, [])

        # === IMMOBILE ===
        return ("immobile", 0.85, [])

    def _detect_fall(self, track_id, kpts, sd) -> bool:
        """
        Détecte une chute :
        - Chute rapide du haut du corps (>30% en quelques frames)
        - OU corps horizontal (épaules au même niveau que les hanches Y)
        """
        heights = self._prev_heights.get(track_id)
        if not heights or len(heights) < 5:
            return False

        recent = list(heights)
        # Chute rapide : la tête descend de plus de 30% du corps
        if len(recent) >= 5:
            h_before = np.mean(recent[-5:-3]) if recent[-5] > 0 and recent[-4] > 0 else 0
            h_now = np.mean(recent[-2:]) if recent[-1] > 0 and recent[-2] > 0 else 0

            if h_before > 0 and h_now > 0:
                # En coordonnées image, Y augmente vers le bas
                drop = h_now - h_before
                body_h = self._body_height(kpts)
                if body_h > 0 and drop > body_h * 0.3:
                    return True

        # Corps horizontal : épaules et hanches au même Y
        if (kpts[L_SHOULDER][0] > 0 and kpts[L_HIP][0] > 0 and
            kpts[R_SHOULDER][0] > 0 and kpts[R_HIP][0] > 0):
            shoulder_y = (kpts[L_SHOULDER][1] + kpts[R_SHOULDER][1]) / 2
            hip_y = (kpts[L_HIP][1] + kpts[R_HIP][1]) / 2
            # Si les épaules sont presque au même Y que les hanches → horizontal
            if abs(shoulder_y - hip_y) < sd * 0.4:
                # Vérifier que c'est anormal (le torse devrait être vertical)
                shoulder_x_diff = abs(kpts[L_SHOULDER][0] - kpts[L_HIP][0])
                if shoulder_x_diff > sd * 0.8:
                    return True

        return False

    def _detect_punch(self, track_id, kpts, sd) -> bool:
        """
        Détecte un coup de poing :
        - Bras complètement tendu (coude aligné épaule-poignet)
        - Mouvement rapide du poignet
        """
        hist = self._history.get(track_id)
        if not hist or len(hist) < 3:
            return False

        # Vérifier bras tendu
        for shoulder, elbow, wrist in [(L_SHOULDER, L_ELBOW, L_WRIST),
                                        (R_SHOULDER, R_ELBOW, R_WRIST)]:
            arm_len = self._dist(kpts, shoulder, wrist)
            upper = self._dist(kpts, shoulder, elbow)
            lower = self._dist(kpts, elbow, wrist)

            if arm_len == float('inf') or upper == float('inf'):
                continue

            # Bras tendu = distance épaule-poignet ≈ épaule-coude + coude-poignet
            if upper + lower > 0 and arm_len / (upper + lower) > 0.92:
                # Vérifier la vitesse du poignet
                prev_kpts = hist[-3][0]
                if prev_kpts[wrist][0] > 0 and kpts[wrist][0] > 0:
                    dx = kpts[wrist][0] - prev_kpts[wrist][0]
                    dy = kpts[wrist][1] - prev_kpts[wrist][1]
                    wrist_speed = np.sqrt(dx**2 + dy**2)
                    if wrist_speed > sd * 0.8:
                        return True

        return False

    def _detect_hands_up(self, kpts, sd) -> bool:
        """Détecte les deux mains en l'air (au-dessus de la tête)."""
        if (kpts[L_WRIST][0] > 0 and kpts[R_WRIST][0] > 0 and
            kpts[NOSE][0] > 0):
            # Les deux poignets au-dessus du nez
            if (kpts[L_WRIST][1] < kpts[NOSE][1] - sd * 0.3 and
                kpts[R_WRIST][1] < kpts[NOSE][1] - sd * 0.3):
                return True
        return False

    def _detect_sitting(self, kpts, sd) -> bool:
        """Détecte la position assise : genoux pliés, hanches basses."""
        if (kpts[L_HIP][0] == 0 or kpts[L_KNEE][0] == 0):
            return False

        # Angle du genou (hanche-genou-cheville)
        hip_y = (kpts[L_HIP][1] + kpts[R_HIP][1]) / 2 if kpts[R_HIP][0] > 0 else kpts[L_HIP][1]
        knee_y = (kpts[L_KNEE][1] + kpts[R_KNEE][1]) / 2 if kpts[R_KNEE][0] > 0 else kpts[L_KNEE][1]

        # Assis = hanches au même niveau que les genoux
        if abs(hip_y - knee_y) < sd * 0.5:
            return True

        return False

    def _detect_bending(self, kpts, sd) -> bool:
        """Détecte que la personne se penche (torse incliné vers l'avant)."""
        if kpts[NOSE][0] == 0 or kpts[L_HIP][0] == 0:
            return False

        shoulder_mid_x = (kpts[L_SHOULDER][0] + kpts[R_SHOULDER][0]) / 2 if kpts[R_SHOULDER][0] > 0 else kpts[L_SHOULDER][0]
        hip_mid_x = (kpts[L_HIP][0] + kpts[R_HIP][0]) / 2 if kpts[R_HIP][0] > 0 else kpts[L_HIP][0]

        shoulder_mid_y = (kpts[L_SHOULDER][1] + kpts[R_SHOULDER][1]) / 2 if kpts[R_SHOULDER][0] > 0 else kpts[L_SHOULDER][1]
        hip_mid_y = (kpts[L_HIP][1] + kpts[R_HIP][1]) / 2 if kpts[R_HIP][0] > 0 else kpts[L_HIP][1]

        # Penché = torse horizontal (shoulder Y proche de hip Y mais X décalé)
        vert_dist = abs(hip_mid_y - shoulder_mid_y)
        horiz_dist = abs(hip_mid_x - shoulder_mid_x)

        if vert_dist < sd * 0.8 and horiz_dist > sd * 0.5:
            return True

        return False

    def cleanup(self, active_ids: set):
        """Nettoie les buffers des IDs perdus."""
        lost = set(self._history.keys()) - active_ids
        for tid in lost:
            del self._history[tid]
            if tid in self._prev_heights:
                del self._prev_heights[tid]


class ActionClassifier:
    """
    Classificateur d'actions principal.
    Utilise PoseAnalyzer (règles géométriques) pour une détection fonctionnelle.
    """

    def __init__(self, weights_path: str = None, device: str = "cuda"):
        self.pose_analyzer = PoseAnalyzer(buffer_size=STGCN_BUFFER_SIZE)

        # Cache des dernières prédictions
        self._last_predictions: Dict[int, Dict[str, float]] = {}
        self._last_actions: Dict[int, str] = {}
        self._last_alerts: List[Tuple] = []
        self._frame_count = 0

        print("[ACTION] Classificateur initialisé (analyse géométrique des poses)")

    def update(self, track_id: int, keypoints_xy: np.ndarray):
        """Met à jour le buffer avec les nouvelles keypoints."""
        self.pose_analyzer.update(track_id, keypoints_xy)

    def classify(self, frame_count: int) -> Dict[int, Dict[str, float]]:
        """
        Lance l'analyse pour toutes les personnes.
        """
        self._frame_count = frame_count
        self._last_alerts = []

        if frame_count % STGCN_INFERENCE_INTERVAL != 0:
            return self._last_predictions

        for tid in list(self.pose_analyzer._history.keys()):
            action, confidence, alerts = self.pose_analyzer.analyze(tid)

            # Construire les probabilités
            prediction = {label: 0.05 for label in ACTION_LABELS}
            if action in prediction:
                prediction[action] = confidence
            else:
                # Action non dans la liste standard → on la map
                prediction["immobile"] = confidence

            self._last_predictions[tid] = prediction
            self._last_actions[tid] = action

            # Stocker les alertes
            for alert_msg in alerts:
                self._last_alerts.append((tid, action, confidence))

        return self._last_predictions

    def get_action(self, track_id: int) -> str:
        """Retourne la dernière action détectée."""
        return self._last_actions.get(track_id, "N/A")

    def get_prediction(self, track_id: int) -> Optional[Dict[str, float]]:
        """Retourne les probabilités complètes."""
        return self._last_predictions.get(track_id, None)

    def check_alerts(self) -> list:
        """Retourne les alertes détectées."""
        return self._last_alerts

    def cleanup_lost_ids(self, active_ids: set):
        """Nettoie les buffers des IDs perdus."""
        self.pose_analyzer.cleanup(active_ids)
        lost = set(self._last_predictions.keys()) - active_ids
        for tid in lost:
            del self._last_predictions[tid]
            if tid in self._last_actions:
                del self._last_actions[tid]

    def get_stats(self) -> dict:
        return {
            "active_tracks": len(self.pose_analyzer._history),
            "current_actions": self._last_actions.copy(),
            "pending_alerts": len(self._last_alerts),
        }
