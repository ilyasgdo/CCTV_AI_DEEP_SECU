"""
Implémentation du ST-GCN (Spatial Temporal Graph Convolutional Network).
Basé sur l'architecture originale de Yan et al. (2018).

Architecture :
    Input: (N, C, T, V) = (batch, channels, frames, keypoints)
    → Couches ST-GCN empilées (convolution spatiale sur le graphe + convolution temporelle)
    → Global Average Pooling
    → Couche Dense → Classification

Graphe COCO (17 keypoints) :
    Les connexions sont définies par la matrice d'adjacence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.config import STGCN_NUM_KEYPOINTS, STGCN_IN_CHANNELS, ACTION_LABELS


# === GRAPHE COCO 17 KEYPOINTS ===
COCO_CONNECTIONS = [
    (0, 1), (0, 2),       # Nez → Yeux
    (1, 3), (2, 4),       # Yeux → Oreilles
    (0, 5), (0, 6),       # Nez → Épaules (approximation)
    (5, 7), (7, 9),       # Épaule G → Coude G → Poignet G
    (6, 8), (8, 10),      # Épaule D → Coude D → Poignet D
    (5, 11), (6, 12),     # Épaules → Hanches
    (11, 13), (13, 15),   # Hanche G → Genou G → Cheville G
    (12, 14), (14, 16),   # Hanche D → Genou D → Cheville D
    (11, 12),             # Hanche G → Hanche D
    (5, 6),               # Épaule G → Épaule D
]


def build_adjacency_matrix(num_nodes: int = 17,
                            connections: list = None) -> np.ndarray:
    """
    Construit la matrice d'adjacence normalisée du graphe squelettique.

    Args:
        num_nodes: Nombre de nœuds (keypoints)
        connections: Liste de tuples (i, j) des connexions

    Returns:
        Matrice d'adjacence normalisée (V, V)
    """
    if connections is None:
        connections = COCO_CONNECTIONS

    # Matrice d'adjacence avec self-loops
    A = np.eye(num_nodes, dtype=np.float32)
    for (i, j) in connections:
        A[i, j] = 1
        A[j, i] = 1

    # Normalisation par le degré (D^{-1/2} * A * D^{-1/2})
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return A_norm


class GraphConvolution(nn.Module):
    """Convolution sur le graphe spatial."""

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer('A', A)  # (V, V) matrice d'adjacence
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, V) — batch, channels, time, vertices
        Returns:
            (N, C_out, T, V)
        """
        # Multiplication par la matrice d'adjacence : agrégation spatiale
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        x = self.conv(x)
        x = self.bn(x)
        return x


class STGCNBlock(nn.Module):
    """
    Bloc ST-GCN = Convolution Spatiale (Graph) + Convolution Temporelle + Résidu.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 A: torch.Tensor, stride: int = 1):
        super().__init__()

        # Convolution spatiale (sur le graphe)
        self.gcn = GraphConvolution(in_channels, out_channels, A)

        # Convolution temporelle (1D le long de l'axe T)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )

        # Connexion résiduelle
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C_in, T, V)
        Returns:
            (N, C_out, T', V) — T' = T/stride
        """
        res = self.residual(x)
        x = self.gcn(x)
        x = self.relu(x)
        x = self.tcn(x)
        x = self.dropout(x)
        x = x + res
        x = self.relu(x)
        return x


class STGCN(nn.Module):
    """
    ST-GCN complet pour la classification d'actions.

    Architecture :
        Input (N, 2, 30, 17)
        → 6 blocs ST-GCN (64, 64, 128, 128, 256, 256 channels)
        → Global Average Pooling
        → FC → num_classes
    """

    def __init__(self, in_channels: int = STGCN_IN_CHANNELS,
                 num_classes: int = None,
                 num_keypoints: int = STGCN_NUM_KEYPOINTS):
        super().__init__()

        if num_classes is None:
            num_classes = len(ACTION_LABELS)

        # Matrice d'adjacence du squelette COCO
        A_np = build_adjacency_matrix(num_keypoints)
        A_tensor = torch.tensor(A_np, dtype=torch.float32)

        # Normalisation d'entrée
        self.data_bn = nn.BatchNorm1d(in_channels * num_keypoints)

        # Couches ST-GCN
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, A_tensor),
            STGCNBlock(64, 64, A_tensor),
            STGCNBlock(64, 128, A_tensor, stride=2),    # T: 30 → 15
            STGCNBlock(128, 128, A_tensor),
            STGCNBlock(128, 256, A_tensor, stride=2),    # T: 15 → 8
            STGCNBlock(256, 256, A_tensor),
        ])

        # Classificateur
        self.fc = nn.Linear(256, num_classes)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, V) = (batch, 2, 30, 17)

        Returns:
            (N, num_classes) — logits pour chaque action
        """
        N, C, T, V = x.shape

        # Batch normalization sur l'entrée
        x_bn = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x_bn = self.data_bn(x_bn)
        x = x_bn.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # Couches ST-GCN
        for layer in self.layers:
            x = layer(x)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (N, 256, 1, 1)
        x = x.view(N, -1)                      # (N, 256)

        # Classification
        x = self.fc(x)  # (N, num_classes)

        return x

    def predict(self, x: torch.Tensor) -> dict:
        """
        Fait une prédiction avec les probabilités par action.

        Args:
            x: (1, C, T, V) — un seul échantillon

        Returns:
            Dict {action_name: probability}
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)[0]

        result = {}
        for i, label in enumerate(ACTION_LABELS):
            result[label] = float(probs[i])

        return result
