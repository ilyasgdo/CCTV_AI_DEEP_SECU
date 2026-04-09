"""
Module de visualisation / overlay pour Sentinel-AI.

Dessine les bounding boxes, labels, squelettes, et informations
HUD sur les frames vidéo annotées.

Usage:
    from src.core.visualizer import Visualizer

    visualizer = Visualizer()
    annotated = visualizer.draw(frame, tracked_entities)
"""

import time
from typing import Optional

import cv2
import numpy as np

from src.core.detector import Detection, PersonDetection
from src.core.tracker import TrackedEntity, EntityStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Constantes de couleurs (BGR)
# ──────────────────────────────────────────────

COLOR_KNOWN_PERSON = (0, 200, 0)        # Vert
COLOR_UNKNOWN_PERSON = (0, 0, 220)      # Rouge
COLOR_LOST_ENTITY = (0, 140, 255)       # Orange
COLOR_OBJECT = (200, 150, 0)            # Bleu clair
COLOR_BBOX_DEFAULT = (255, 200, 0)      # Cyan
COLOR_SKELETON = (0, 255, 255)          # Jaune
COLOR_TEXT_BG = (0, 0, 0)              # Noir
COLOR_HUD_BG = (20, 20, 30)           # Gris très sombre
COLOR_HUD_TEXT = (200, 220, 240)       # Blanc chaud
COLOR_HUD_ACCENT = (0, 200, 255)      # Or

# Connexions du squelette COCO (17 keypoints)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # Tête
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Bras
    (5, 11), (6, 12),                          # Torse
    (11, 12), (11, 13), (13, 15),              # Jambe gauche
    (12, 14), (14, 16),                        # Jambe droite
]


class Visualizer:
    """
    Module de visualisation pour dessiner les détections et
    informations sur les frames vidéo.

    Dessine :
    - Bounding boxes colorées par statut (connu/inconnu/objet).
    - Labels avec classe + confiance + ID de tracking.
    - Squelettes si pose estimation disponible.
    - HUD avec FPS, nombre de personnes, timestamp.

    Args:
        show_skeleton: Afficher les squelettes (si dispo).
        show_hud: Afficher les informations système.
        show_labels: Afficher les labels de classe.
        bbox_thickness: Épaisseur des bounding boxes.
        font_scale: Taille de la police.
    """

    def __init__(
        self,
        show_skeleton: bool = True,
        show_hud: bool = True,
        show_labels: bool = True,
        bbox_thickness: int = 2,
        font_scale: float = 0.6,
    ) -> None:
        self.show_skeleton = show_skeleton
        self.show_hud = show_hud
        self.show_labels = show_labels
        self.bbox_thickness = bbox_thickness
        self.font_scale = font_scale

        # Compteur FPS interne
        self._fps_timestamps: list[float] = []
        self._current_fps: float = 0.0

    def draw(
        self,
        frame: np.ndarray,
        entities: Optional[list[TrackedEntity]] = None,
        detections: Optional[list[Detection]] = None,
        fps: Optional[float] = None,
        extra_info: Optional[dict] = None,
    ) -> np.ndarray:
        """Dessine les annotations sur une copie de la frame.

        Args:
            frame: Frame BGR originale.
            entities: Entités trackées (priorité sur detections).
            detections: Détections brutes (si pas de tracking).
            fps: FPS à afficher (sinon calculé en interne).
            extra_info: Infos supplémentaires pour le HUD.

        Returns:
            Frame annotée (copie, l'original n'est pas modifié).
        """
        annotated = frame.copy()

        # Calcul FPS interne
        now = time.time()
        self._fps_timestamps.append(now)
        self._fps_timestamps = [
            t for t in self._fps_timestamps if now - t < 2.0
        ]
        if len(self._fps_timestamps) >= 2:
            elapsed = (
                self._fps_timestamps[-1]
                - self._fps_timestamps[0]
            )
            if elapsed > 0:
                self._current_fps = (
                    (len(self._fps_timestamps) - 1) / elapsed
                )

        display_fps = fps if fps is not None else self._current_fps

        # ── Dessiner les entités trackées ──
        person_count = 0
        if entities:
            for entity in entities:
                # En mode live, on masque les entites perdues pour eviter
                # les boites stale visibles a l'ecran.
                if entity.status in {EntityStatus.DISAPPEARED, EntityStatus.LOST}:
                    continue
                self._draw_entity(annotated, entity)
                if entity.is_person:
                    person_count += 1

        # ── Ou les détections brutes ──
        elif detections:
            for det in detections:
                self._draw_detection(annotated, det)
                if det.class_id == 0:
                    person_count += 1

        # ── HUD ──
        if self.show_hud:
            info = extra_info or {}
            info.setdefault("fps", display_fps)
            info.setdefault("persons", person_count)
            self._draw_hud(annotated, info)

        return annotated

    def _get_entity_color(
        self, entity: TrackedEntity
    ) -> tuple[int, int, int]:
        """Détermine la couleur d'une entité selon son statut.

        Args:
            entity: Entité trackée.

        Returns:
            Couleur BGR.
        """
        if entity.status == EntityStatus.LOST:
            return COLOR_LOST_ENTITY

        if not entity.is_person:
            return COLOR_OBJECT

        # Personne : couleur selon reconnaissance faciale
        if entity.face_status == "known":
            return COLOR_KNOWN_PERSON
        else:
            return COLOR_UNKNOWN_PERSON

    def _draw_entity(
        self,
        frame: np.ndarray,
        entity: TrackedEntity,
    ) -> None:
        """Dessine une entité trackée sur la frame.

        Args:
            frame: Frame à annoter (modifiée in-place).
            entity: Entité à dessiner.
        """
        if not entity.detections:
            return

        det = entity.detections[-1]
        color = self._get_entity_color(entity)
        x1, y1, x2, y2 = det.bbox

        # Bounding box
        thickness = self.bbox_thickness
        if entity.status == EntityStatus.LOST:
            # Pointillés pour les entités perdues
            self._draw_dashed_rect(
                frame, (x1, y1), (x2, y2), color, thickness
            )
        else:
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), color, thickness
            )

        # Label
        if self.show_labels:
            # Construire le label
            parts = [f"ID:{entity.track_id}"]

            if entity.face_name:
                parts.append(entity.face_name)
            elif entity.is_person:
                parts.append("Inconnu")
            else:
                parts.append(det.class_name)

            parts.append(f"{det.confidence:.0%}")

            if entity.status == EntityStatus.LOST:
                parts.append("PERDU")

            label = " | ".join(parts)
            self._draw_label(frame, label, (x1, y1 - 5), color)

        # Squelette
        if (self.show_skeleton
                and isinstance(det, PersonDetection)
                and det.keypoints):
            self._draw_skeleton(frame, det.keypoints)

    def _draw_detection(
        self,
        frame: np.ndarray,
        det: Detection,
    ) -> None:
        """Dessine une détection brute (sans tracking).

        Args:
            frame: Frame à annoter.
            det: Détection à dessiner.
        """
        color = (
            COLOR_UNKNOWN_PERSON if det.class_id == 0
            else COLOR_OBJECT
        )
        x1, y1, x2, y2 = det.bbox

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), color, self.bbox_thickness
        )

        if self.show_labels:
            label = f"{det.class_name} {det.confidence:.0%}"
            self._draw_label(frame, label, (x1, y1 - 5), color)

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """Dessine un label avec fond.

        Args:
            frame: Frame à annoter.
            text: Texte du label.
            position: Position (x, y) du coin bas-gauche.
            color: Couleur du fond.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), baseline = cv2.getTextSize(
            text, font, self.font_scale, 1
        )

        x, y = position
        y = max(y, h + 5)

        # Fond du label
        cv2.rectangle(
            frame,
            (x, y - h - 5),
            (x + w + 4, y + 2),
            color,
            -1,
        )

        # Texte
        cv2.putText(
            frame, text,
            (x + 2, y - 3),
            font, self.font_scale,
            (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: list[tuple[float, float, float]],
        min_confidence: float = 0.3,
    ) -> None:
        """Dessine le squelette COCO sur la frame.

        Args:
            frame: Frame à annoter.
            keypoints: 17 points clés (x, y, confidence).
            min_confidence: Seuil min pour dessiner un point.
        """
        # Dessiner les points
        for kp in keypoints:
            x, y, conf = kp
            if conf < min_confidence or (x == 0 and y == 0):
                continue
            cv2.circle(
                frame, (int(x), int(y)), 4, COLOR_SKELETON, -1
            )

        # Dessiner les connexions
        for i, j in SKELETON_CONNECTIONS:
            if i >= len(keypoints) or j >= len(keypoints):
                continue
            kp1 = keypoints[i]
            kp2 = keypoints[j]
            if (kp1[2] < min_confidence
                    or kp2[2] < min_confidence):
                continue
            if (kp1[0] == 0 and kp1[1] == 0) or \
                    (kp2[0] == 0 and kp2[1] == 0):
                continue
            cv2.line(
                frame,
                (int(kp1[0]), int(kp1[1])),
                (int(kp2[0]), int(kp2[1])),
                COLOR_SKELETON, 2,
                cv2.LINE_AA,
            )

    def _draw_dashed_rect(
        self,
        frame: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
        dash_length: int = 10,
    ) -> None:
        """Dessine un rectangle en pointillés.

        Args:
            frame: Frame à annoter.
            pt1: Coin supérieur gauche.
            pt2: Coin inférieur droit.
            color: Couleur.
            thickness: Épaisseur.
            dash_length: Longueur des tirets en pixels.
        """
        x1, y1 = pt1
        x2, y2 = pt2

        # Lignes horizontales
        for x in range(x1, x2, dash_length * 2):
            cv2.line(
                frame,
                (x, y1), (min(x + dash_length, x2), y1),
                color, thickness,
            )
            cv2.line(
                frame,
                (x, y2), (min(x + dash_length, x2), y2),
                color, thickness,
            )

        # Lignes verticales
        for y in range(y1, y2, dash_length * 2):
            cv2.line(
                frame,
                (x1, y), (x1, min(y + dash_length, y2)),
                color, thickness,
            )
            cv2.line(
                frame,
                (x2, y), (x2, min(y + dash_length, y2)),
                color, thickness,
            )

    def _draw_hud(
        self,
        frame: np.ndarray,
        info: dict,
    ) -> None:
        """Dessine le HUD (Head-Up Display) en haut de la frame.

        Affiche : FPS, personnes, timestamp, et infos système.

        Args:
            frame: Frame à annoter.
            info: Dictionnaire d'informations à afficher.
        """
        h, w = frame.shape[:2]

        # Fond semi-transparent en haut
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30

        # ── Nom du système ──
        cv2.putText(
            frame, "SENTINEL-AI",
            (10, y), font, 0.6, COLOR_HUD_ACCENT, 2, cv2.LINE_AA,
        )

        # ── FPS ──
        fps = info.get("fps", 0)
        fps_color = (
            (0, 200, 0) if fps >= 15
            else (0, 140, 255) if fps >= 10
            else (0, 0, 220)
        )
        fps_text = f"FPS: {fps:.0f}"
        cv2.putText(
            frame, fps_text,
            (180, y), font, 0.5, fps_color, 1, cv2.LINE_AA,
        )

        # ── Personnes ──
        persons = info.get("persons", 0)
        cv2.putText(
            frame, f"Personnes: {persons}",
            (290, y), font, 0.5, COLOR_HUD_TEXT, 1, cv2.LINE_AA,
        )

        # ── Timestamp ──
        ts = time.strftime("%H:%M:%S")
        cv2.putText(
            frame, ts,
            (w - 100, y), font, 0.5, COLOR_HUD_TEXT, 1, cv2.LINE_AA,
        )

        # ── Indicateur d'enregistrement ──
        status = info.get("status", "LIVE")
        status_color = (
            (0, 200, 0) if status == "LIVE"
            else (0, 0, 220)
        )
        cv2.circle(frame, (w - 130, y - 5), 5, status_color, -1)

    def __repr__(self) -> str:
        return (
            f"Visualizer("
            f"skeleton={self.show_skeleton}, "
            f"hud={self.show_hud}, "
            f"labels={self.show_labels})"
        )
