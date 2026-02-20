"""
Heatmap de mouvement — Carte de chaleur des trajets.

Accumule les positions de tous les personnes détectées au fil du temps
et génère une carte de chaleur semi-transparente à superposer sur l'image.

Usage :
    hm = MovementHeatmap(width, height)
    hm.update(detections)
    frame = hm.draw(frame)       # Overlay sur le frame
    hm.save("heatmap.png")       # Export image
"""
import cv2
import numpy as np
from typing import List, Optional
from pathlib import Path
import time


class MovementHeatmap:
    """
    Carte de chaleur des mouvements accumulés.
    """

    def __init__(self, width: int = 1920, height: int = 1080,
                 decay: float = 0.999, point_radius: int = 30):
        """
        Args:
            width: Largeur de l'image
            height: Hauteur de l'image
            decay: Facteur de décroissance (0.999 = lente, 0.99 = rapide)
            point_radius: Rayon des points accumulés (pixels)
        """
        self.width = width
        self.height = height
        self.decay = decay
        self.point_radius = point_radius

        # Matrice d'accumulation (float32)
        self.accumulator = np.zeros((height, width), dtype=np.float32)

        # Statistiques
        self.total_points = 0
        self.start_time = time.time()

        # Cache de la colormap
        self._cached_overlay: Optional[np.ndarray] = None
        self._cache_frame = 0
        self._cache_interval = 5  # Recalcule la colormap toutes les N frames

    def update(self, detections: list, frame_count: int = 0):
        """
        Ajoute les positions des personnes détectées à l'accumulateur.
        
        Args:
            detections: Liste de PersonDetection avec .center
            frame_count: Numéro de frame
        """
        # Décroissance progressive
        self.accumulator *= self.decay

        for det in detections:
            cx, cy = det.center
            cx, cy = int(cx), int(cy)

            if 0 <= cx < self.width and 0 <= cy < self.height:
                # Ajouter un point gaussien à la position
                cv2.circle(self.accumulator, (cx, cy),
                           self.point_radius, 1.0, -1)  # Filled circle
                self.total_points += 1

        # Invalidate cache
        if frame_count % self._cache_interval == 0:
            self._cached_overlay = None
            self._cache_frame = frame_count

    def _generate_overlay(self) -> np.ndarray:
        """Génère l'overlay colormap à partir de l'accumulateur."""
        # Normaliser entre 0 et 255
        if self.accumulator.max() > 0:
            normalized = (self.accumulator / self.accumulator.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros((self.height, self.width), dtype=np.uint8)

        # Appliquer un flou gaussien pour lisser
        blurred = cv2.GaussianBlur(normalized, (41, 41), 0)

        # Appliquer la colormap JET (bleu → vert → rouge)
        colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

        # Masquer les zones à zéro (rester transparent)
        mask = blurred > 5  # Seuil minimum
        mask_3d = np.stack([mask, mask, mask], axis=-1)

        # Mettre à noir les zones non-actives
        colored[~mask_3d] = 0

        return colored, mask_3d

    def draw(self, frame: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Superpose la heatmap sur le frame.
        
        Args:
            frame: Image BGR
            alpha: Opacité de la heatmap (0.0 = invisible, 1.0 = opaque)
            
        Returns:
            Frame avec heatmap superposée
        """
        if self.accumulator.max() <= 0:
            return frame

        # Utiliser le cache si disponible
        if self._cached_overlay is None:
            self._cached_overlay = self._generate_overlay()

        colored, mask_3d = self._cached_overlay

        # Redimensionner si nécessaire
        if colored.shape[:2] != frame.shape[:2]:
            colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))
            mask_3d = cv2.resize(
                mask_3d.astype(np.uint8),
                (frame.shape[1], frame.shape[0])
            ).astype(bool)

        # Blend uniquement là où il y a de la chaleur
        result = frame.copy()
        result[mask_3d] = cv2.addWeighted(
            frame, 1.0 - alpha, colored, alpha, 0
        )[mask_3d]

        # Label "HEATMAP" en haut
        cv2.putText(result, "HEATMAP ACTIVE", (15, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        return result

    def save(self, path: str = "heatmap.png"):
        """Exporte la heatmap sous forme d'image."""
        if self.accumulator.max() > 0:
            normalized = (self.accumulator / self.accumulator.max() * 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(normalized, (41, 41), 0)
            colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
            cv2.imwrite(path, colored)
            print(f"  [HEATMAP] Exportée → {path}")
        else:
            print("  [HEATMAP] Aucune donnée à exporter")

    def reset(self):
        """Réinitialise la heatmap."""
        self.accumulator = np.zeros((self.height, self.width), dtype=np.float32)
        self.total_points = 0
        self.start_time = time.time()
        self._cached_overlay = None
        print("  [HEATMAP] Réinitialisée")

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time
        return {
            "total_points": self.total_points,
            "max_intensity": float(self.accumulator.max()),
            "active_duration_s": elapsed,
            "decay": self.decay,
        }
