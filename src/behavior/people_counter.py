"""
Compteur d'entrées/sorties — Lignes de comptage.

Définit une ou plusieurs lignes de comptage sur l'image.
Quand le centre d'une personne traverse la ligne, c'est compté
comme une ENTRÉE ou une SORTIE selon la direction du mouvement.

Usage :
    counter = PeopleCounter(frame_width, frame_height)
    counter.update(detections)
    frame = counter.draw(frame)
"""
import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class CountingLine:
    """
    Une ligne de comptage avec direction.
    
    La direction d'entrée est définie par le vecteur normal :
    traverser de gauche à droite ou de haut en bas = ENTRÉE.
    """

    def __init__(self, pt1: Tuple[int, int], pt2: Tuple[int, int],
                 name: str = "Ligne 1", color: Tuple[int, int, int] = (0, 255, 255)):
        self.pt1 = pt1
        self.pt2 = pt2
        self.name = name
        self.color = color
        self.entries = 0
        self.exits = 0

        # Vecteur normal à la ligne (pour déterminer la direction)
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        # Normal pointe vers le "haut" / "gauche" = côté entrée
        self.normal = np.array([-dy, dx], dtype=float)
        norm = np.linalg.norm(self.normal)
        if norm > 0:
            self.normal /= norm

    def side(self, point: Tuple[float, float]) -> float:
        """
        Retourne le côté du point par rapport à la ligne.
        Positif = côté entrée, négatif = côté sortie.
        """
        v = np.array([point[0] - self.pt1[0], point[1] - self.pt1[1]], dtype=float)
        return float(np.dot(v, self.normal))

    def check_crossing(self, prev_pos: Tuple[float, float],
                       curr_pos: Tuple[float, float]) -> Optional[str]:
        """
        Vérifie si le mouvement prev_pos → curr_pos traverse la ligne.
        
        Returns:
            "ENTREE" | "SORTIE" | None
        """
        prev_side = self.side(prev_pos)
        curr_side = self.side(curr_pos)

        # Passage d'un côté à l'autre
        if prev_side > 0 and curr_side <= 0:
            self.exits += 1
            return "SORTIE"
        elif prev_side <= 0 and curr_side > 0:
            self.entries += 1
            return "ENTREE"

        return None


class PeopleCounter:
    """
    Compteur de personnes avec lignes de comptage configurables.
    """

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.lines: List[CountingLine] = []
        self.total_entries = 0
        self.total_exits = 0

        # Historique des positions {track_id: (x, y)}
        self._prev_positions: Dict[int, Tuple[float, float]] = {}

        # Log des événements
        self._events: List[dict] = []

        # Configurer la ligne par défaut (horizontale au milieu)
        self.set_default_line()

    def set_default_line(self):
        """Place une ligne horizontale au milieu de l'image."""
        y = self.frame_height // 2
        margin = int(self.frame_width * 0.1)
        self.lines = [
            CountingLine(
                pt1=(margin, y),
                pt2=(self.frame_width - margin, y),
                name="Ligne principale",
                color=(0, 255, 255)  # Jaune
            )
        ]

    def add_line(self, pt1: Tuple[int, int], pt2: Tuple[int, int],
                 name: str = "Custom", color: Tuple[int, int, int] = (255, 200, 0)):
        """Ajoute une ligne de comptage personnalisée."""
        self.lines.append(CountingLine(pt1, pt2, name, color))

    def update(self, detections: list) -> List[dict]:
        """
        Met à jour le compteur avec les détections courantes.
        
        Args:
            detections: Liste de PersonDetection avec .track_id et .center
            
        Returns:
            Liste d'événements de traversée [{track_id, type, line, time}]
        """
        events = []
        current_positions = {}

        for det in detections:
            tid = det.track_id
            cx, cy = det.center
            current_positions[tid] = (cx, cy)

            if tid in self._prev_positions:
                prev = self._prev_positions[tid]
                for line in self.lines:
                    crossing = line.check_crossing(prev, (cx, cy))
                    if crossing:
                        event = {
                            "track_id": tid,
                            "type": crossing,
                            "line": line.name,
                            "time": time.time(),
                            "name": getattr(det, 'name', 'INCONNU'),
                        }
                        events.append(event)
                        self._events.append(event)

                        if crossing == "ENTREE":
                            self.total_entries += 1
                        else:
                            self.total_exits += 1

                        print(f"  🚪 {crossing} — {event['name']} (ID:{tid}) [{line.name}]")

        self._prev_positions = current_positions
        return events

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Dessine les lignes de comptage et les compteurs sur le frame."""
        overlay = frame.copy()

        for line in self.lines:
            # Dessiner la ligne
            cv2.line(overlay, line.pt1, line.pt2, line.color, 3, cv2.LINE_AA)

            # Petites flèches pour indiquer la direction
            mid_x = (line.pt1[0] + line.pt2[0]) // 2
            mid_y = (line.pt1[1] + line.pt2[1]) // 2

            arrow_len = 25
            nx = int(line.normal[0] * arrow_len)
            ny = int(line.normal[1] * arrow_len)

            # Flèche entrée (verte)
            cv2.arrowedLine(overlay,
                            (mid_x - nx, mid_y - ny),
                            (mid_x, mid_y),
                            (0, 200, 0), 2, tipLength=0.4)
            # Flèche sortie (rouge)
            cv2.arrowedLine(overlay,
                            (mid_x + nx, mid_y + ny),
                            (mid_x, mid_y),
                            (0, 0, 200), 2, tipLength=0.4)

            # Labels "IN" / "OUT"
            cv2.putText(overlay, "IN", (mid_x - nx - 15, mid_y - ny - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
            cv2.putText(overlay, "OUT", (mid_x + nx - 15, mid_y + ny + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

        # Blend
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # === Compteur en haut à droite ===
        box_w, box_h = 200, 90
        bx = frame.shape[1] - box_w - 15
        by = 15

        # Fond semi-transparent
        sub_img = frame[by:by+box_h, bx:bx+box_w]
        dark = np.zeros_like(sub_img)
        cv2.addWeighted(sub_img, 0.3, dark, 0.7, 0, sub_img)
        frame[by:by+box_h, bx:bx+box_w] = sub_img

        # Bordure
        cv2.rectangle(frame, (bx, by), (bx+box_w, by+box_h), (0, 255, 255), 1)

        # Textes
        cv2.putText(frame, "COMPTEUR", (bx + 50, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Entrées (vert)
        cv2.putText(frame, f"IN:  {self.total_entries}", (bx + 15, by + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        # Sorties (rouge)
        cv2.putText(frame, f"OUT: {self.total_exits}", (bx + 15, by + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

        # Personnes présentes
        present = self.total_entries - self.total_exits
        if present < 0:
            present = 0
        cv2.putText(frame, f"= {present}", (bx + 130, by + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def cleanup_lost_ids(self, active_ids: set):
        """Nettoie les IDs perdus."""
        lost = set(self._prev_positions.keys()) - active_ids
        for tid in lost:
            del self._prev_positions[tid]

    def get_stats(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "present": max(0, self.total_entries - self.total_exits),
            "lines": len(self.lines),
            "recent_events": self._events[-10:],
        }
