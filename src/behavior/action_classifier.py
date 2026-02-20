"""
Interface de haut niveau pour le classificateur d'actions ST-GCN.
Gère le chargement du modèle, l'inférence, et les alertes.
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    STGCN_INFERENCE_INTERVAL, ALERT_ACTIONS,
    ACTION_LABELS, STGCN_BUFFER_SIZE
)
from src.models.stgcn.model import STGCN
from src.behavior.skeleton_buffer import MultiPersonBuffer


class ActionClassifier:
    """
    Classificateur d'actions utilisant le ST-GCN.

    Gère :
    - Le buffer temporel par personne
    - L'inférence périodique (toutes les N frames)
    - La détection d'alertes
    """

    def __init__(self, weights_path: str = None, device: str = "cuda"):
        """
        Args:
            weights_path: Chemin vers les poids pré-entraînés (None = modèle non entraîné)
            device: "cuda" ou "cpu"
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialiser le modèle
        self.model = STGCN(
            in_channels=2,
            num_classes=len(ACTION_LABELS)
        ).to(self.device)

        # Charger les poids pré-entraînés si disponibles
        if weights_path and Path(weights_path).exists():
            print(f"[ACTION] Chargement des poids : {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device,
                                    weights_only=True)
            self.model.load_state_dict(state_dict)
            print("[ACTION] Poids chargés avec succès")
        else:
            print("[ACTION] ⚠ Modèle non entraîné (poids aléatoires)")
            print("[ACTION]   Pour un système de production, entraîner sur NTU-RGB+D")

        self.model.eval()

        # Buffer multi-personnes
        self.buffer = MultiPersonBuffer(buffer_size=STGCN_BUFFER_SIZE)

        # Cache des dernières prédictions
        self._last_predictions: Dict[int, Dict[str, float]] = {}
        self._last_actions: Dict[int, str] = {}
        self._frame_count = 0

    def update(self, track_id: int, keypoints_xy: np.ndarray):
        """
        Met à jour le buffer d'une personne avec les nouvelles keypoints.

        Args:
            track_id: ID de suivi
            keypoints_xy: (17, 2) array des coordonnées
        """
        self.buffer.update(track_id, keypoints_xy)

    def should_infer(self, frame_count: int) -> bool:
        """Détermine si c'est le moment de lancer l'inférence ST-GCN."""
        return frame_count % STGCN_INFERENCE_INTERVAL == 0

    def classify(self, frame_count: int) -> Dict[int, Dict[str, float]]:
        """
        Lance l'inférence ST-GCN pour toutes les personnes avec un buffer plein.

        Args:
            frame_count: Numéro de frame actuel

        Returns:
            Dict {track_id: {action: probability, ...}}
        """
        self._frame_count = frame_count

        if not self.should_infer(frame_count):
            return self._last_predictions

        # Récupérer les buffers prêts
        ready_buffers = self.buffer.get_ready_buffers()

        if not ready_buffers:
            return self._last_predictions

        # Préparer le batch
        track_ids = list(ready_buffers.keys())
        batch = np.stack([ready_buffers[tid] for tid in track_ids])  # (B, C, T, V)

        # Convertir en tenseur PyTorch
        tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)

        # Inférence
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Stocker les résultats
        for i, tid in enumerate(track_ids):
            prediction = {}
            for j, label in enumerate(ACTION_LABELS):
                prediction[label] = float(probs[i, j])

            self._last_predictions[tid] = prediction
            self._last_actions[tid] = max(prediction, key=prediction.get)

        return self._last_predictions

    def get_action(self, track_id: int) -> str:
        """Retourne la dernière action prédite pour un track_id."""
        return self._last_actions.get(track_id, "N/A")

    def get_prediction(self, track_id: int) -> Optional[Dict[str, float]]:
        """Retourne les probabilités complètes pour un track_id."""
        return self._last_predictions.get(track_id, None)

    def check_alerts(self) -> list:
        """
        Vérifie si des actions d'alerte sont détectées.

        Returns:
            Liste de (track_id, action, confidence)
        """
        alerts = []
        for tid, preds in self._last_predictions.items():
            for action in ALERT_ACTIONS:
                if action in preds and preds[action] > 0.7:  # Seuil d'alerte
                    alerts.append((tid, action, preds[action]))
        return alerts

    def cleanup_lost_ids(self, active_ids: set):
        """Nettoie les buffers et prédictions des IDs perdus."""
        self.buffer.cleanup_lost_ids(active_ids)
        lost = set(self._last_predictions.keys()) - active_ids
        for tid in lost:
            del self._last_predictions[tid]
            if tid in self._last_actions:
                del self._last_actions[tid]

    def get_stats(self) -> dict:
        """Statistiques du classificateur."""
        return {
            "device": str(self.device),
            "buffer_stats": self.buffer.get_stats(),
            "active_predictions": len(self._last_predictions),
            "current_actions": self._last_actions.copy()
        }
