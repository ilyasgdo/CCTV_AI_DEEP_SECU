"""
Module d'affichage : dessine les boîtes, squelettes et informations sur la vidéo.
"""
import cv2
import numpy as np
from typing import List

# Connexions du squelette COCO pour dessiner les os
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),       # Nez → Yeux
    (1, 3), (2, 4),       # Yeux → Oreilles
    (5, 6),                # Épaule gauche → Épaule droite
    (5, 7), (7, 9),       # Épaule G → Coude G → Poignet G
    (6, 8), (8, 10),      # Épaule D → Coude D → Poignet D
    (5, 11), (6, 12),     # Épaules → Hanches
    (11, 12),              # Hanche G → Hanche D
    (11, 13), (13, 15),   # Hanche G → Genou G → Cheville G
    (12, 14), (14, 16),   # Hanche D → Genou D → Cheville D
]

# Couleurs pour différents IDs (palette de 20 couleurs)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (128, 128, 255), (255, 128, 128), (128, 255, 128), (255, 128, 255),
    (128, 255, 255), (255, 255, 128), (192, 0, 0), (0, 192, 0),
]


def get_color(track_id: int) -> tuple:
    """Retourne une couleur unique basée sur l'ID de suivi."""
    return COLORS[track_id % len(COLORS)]


def draw_detections(frame: np.ndarray, detections: list,
                    draw_skeleton: bool = True,
                    draw_bbox: bool = True,
                    draw_label: bool = True) -> np.ndarray:
    """
    Dessine les détections sur la frame.

    Args:
        frame: Image BGR
        detections: Liste de PersonDetection
        draw_skeleton: Dessiner le squelette
        draw_bbox: Dessiner la boîte englobante
        draw_label: Dessiner le label (ID + nom + action)

    Returns:
        Frame annotée
    """
    annotated = frame.copy()

    for det in detections:
        color = get_color(det.track_id)

        # --- Boîte englobante ---
        if draw_bbox:
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # --- Label ---
        if draw_label:
            x1, y1 = det.bbox[:2].astype(int)
            label = f"ID:{det.track_id} | {det.name}"
            if det.action != "N/A":
                label += f" | {det.action}"

            # Fond du texte
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Squelette ---
        if draw_skeleton:
            kpts = det.keypoints  # (17, 3)

            # Dessiner les points
            for j in range(17):
                x, y, conf = kpts[j]
                if conf > 0.5:  # Seuil de confiance du keypoint
                    cv2.circle(annotated, (int(x), int(y)), 4, color, -1)

            # Dessiner les connexions
            for (a, b) in SKELETON_CONNECTIONS:
                xa, ya, ca = kpts[a]
                xb, yb, cb = kpts[b]
                if ca > 0.5 and cb > 0.5:
                    cv2.line(annotated, (int(xa), int(ya)), (int(xb), int(yb)),
                             color, 2)

    return annotated


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Affiche le FPS en haut à gauche."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def draw_alert(frame: np.ndarray, message: str,
               position: tuple = None) -> np.ndarray:
    """Affiche une alerte rouge."""
    h, w = frame.shape[:2]
    if position is None:
        position = (w // 2 - 200, 50)

    # Fond rouge semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (position[0] - 10, position[1] - 30),
                  (position[0] + 400, position[1] + 10), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"! {message}", position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame
