"""
Buffer temporel pour stocker l'historique des squelettes.
Utilise collections.deque pour une mémoire glissante efficace.
"""
import numpy as np
from collections import deque
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import STGCN_BUFFER_SIZE, STGCN_NUM_KEYPOINTS, STGCN_IN_CHANNELS


class SkeletonBuffer:
    """
    Gère le buffer temporel d'un seul individu.

    Stocke les N dernières frames de keypoints sous forme de deque.
    Quand le buffer est plein, les anciennes frames sont automatiquement
    supprimées (FIFO).
    """

    def __init__(self, buffer_size: int = STGCN_BUFFER_SIZE,
                 num_keypoints: int = STGCN_NUM_KEYPOINTS,
                 in_channels: int = STGCN_IN_CHANNELS):
        """
        Args:
            buffer_size: Nombre de frames à garder (30 ≈ 1s à 30fps)
            num_keypoints: Nombre de keypoints (17 COCO)
            in_channels: Nombre de canaux par keypoint (2 = X,Y ou 3 = X,Y,conf)
        """
        self.buffer_size = buffer_size
        self.num_keypoints = num_keypoints
        self.in_channels = in_channels
        self.buffer: deque = deque(maxlen=buffer_size)

    def add_frame(self, keypoints_xy: np.ndarray):
        """
        Ajoute les keypoints d'une frame au buffer.

        Args:
            keypoints_xy: Array (17, 2) ou (17, 3) des coordonnées
        """
        if keypoints_xy.shape[0] != self.num_keypoints:
            raise ValueError(
                f"Attendu {self.num_keypoints} keypoints, "
                f"reçu {keypoints_xy.shape[0]}"
            )
        # Ne garder que les canaux voulus (X, Y) ou (X, Y, conf)
        kpts = keypoints_xy[:, :self.in_channels].copy()
        self.buffer.append(kpts)

    @property
    def is_ready(self) -> bool:
        """Le buffer est-il plein (prêt pour l'inférence) ?"""
        return len(self.buffer) == self.buffer_size

    @property
    def fill_ratio(self) -> float:
        """Ratio de remplissage (0.0 à 1.0)."""
        return len(self.buffer) / self.buffer_size

    def to_numpy(self) -> Optional[np.ndarray]:
        """
        Convertit le buffer en array numpy pour le ST-GCN.

        Returns:
            Array (C, T, V) = (in_channels, buffer_size, num_keypoints)
            ou None si le buffer n'est pas prêt
        """
        if not self.is_ready:
            return None

        # Stack les frames : (T, V, C)
        data = np.stack(list(self.buffer), axis=0)  # (T, 17, 2)

        # Normaliser les coordonnées (centrer sur la hanche)
        # Centre = moyenne des hanches gauche (11) et droite (12)
        hip_center = (data[:, 11, :] + data[:, 12, :]) / 2  # (T, 2)
        data = data - hip_center[:, np.newaxis, :]  # Centrer

        # Normaliser par la taille du squelette
        # Utiliser la distance épaule-hanche comme référence
        scale = np.mean(np.linalg.norm(
            data[:, 5, :] - data[:, 11, :], axis=-1
        ))
        if scale > 0:
            data = data / scale

        # Transposer en (C, T, V) pour le ST-GCN
        data = data.transpose(2, 0, 1)  # (2, 30, 17)

        return data.astype(np.float32)

    def reset(self):
        """Vide le buffer."""
        self.buffer.clear()


class MultiPersonBuffer:
    """
    Gère les buffers temporels de TOUTES les personnes à l'écran.
    Crée automatiquement un buffer par track_id.
    """

    def __init__(self, buffer_size: int = STGCN_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.buffers: Dict[int, SkeletonBuffer] = {}

    def update(self, track_id: int, keypoints_xy: np.ndarray):
        """
        Met à jour le buffer d'une personne.
        Crée le buffer si c'est un nouvel ID.
        """
        if track_id not in self.buffers:
            self.buffers[track_id] = SkeletonBuffer(self.buffer_size)
        self.buffers[track_id].add_frame(keypoints_xy)

    def get_ready_buffers(self) -> Dict[int, np.ndarray]:
        """
        Retourne les buffers prêts pour l'inférence ST-GCN.

        Returns:
            Dict {track_id: array (C, T, V)}
        """
        ready = {}
        for track_id, buf in self.buffers.items():
            if buf.is_ready:
                data = buf.to_numpy()
                if data is not None:
                    ready[track_id] = data
        return ready

    def cleanup_lost_ids(self, active_ids: set):
        """Supprime les buffers des personnes disparues."""
        lost = set(self.buffers.keys()) - active_ids
        for tid in lost:
            del self.buffers[tid]

    def get_stats(self) -> dict:
        """Retourne les statistiques des buffers."""
        return {
            "total_buffers": len(self.buffers),
            "ready_buffers": sum(1 for b in self.buffers.values() if b.is_ready),
            "fill_ratios": {
                tid: f"{buf.fill_ratio:.0%}"
                for tid, buf in self.buffers.items()
            }
        }
