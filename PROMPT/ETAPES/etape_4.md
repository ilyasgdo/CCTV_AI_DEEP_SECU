# 🧠 Étape 4 — Analyse Comportementale via ST-GCN (Le Cœur du Système)

## 📋 Summary (À lire AVANT de commencer)

**Objectif** : Implémenter l'analyse comportementale par **ST-GCN** (Spatial Temporal Graph Convolutional Network). Le système crée un "buffer temporel" pour chaque personne (30 dernières frames de squelette) et le soumet au ST-GCN pour classifier l'action en cours (marcher, chute, coup, immobile...). En complément, une règle spatiale de **maraudage** détecte les personnes restant trop longtemps dans une zone définie.

**Durée estimée** : 3-5 heures (la plus complexe de toutes les étapes)

**Prérequis** :
- ✅ Étapes 0, 1, 2 et 3 entièrement validées
- ✅ Le détecteur extrait les 17 keypoints COCO de manière fiable
- ✅ PyTorch fonctionne avec CUDA
- ✅ Connexion internet (pour télécharger le modèle ST-GCN pré-entraîné)

**Ce que vous aurez à la fin** :
- ✅ Buffer temporel (`deque`) par personne avec 30 frames de squelette
- ✅ Modèle ST-GCN chargé et fonctionnel sur GPU
- ✅ Classification d'actions en temps réel (marcher, chute, coup...)
- ✅ Détection de maraudage par polygone spatial
- ✅ Alertes visuelles et en base de données
- ✅ Test complet validé

---

## 📝 Étapes Détaillées

### 4.1 — Comprendre le ST-GCN (Théorie Essentielle)

> [!NOTE]
> **Pourquoi le ST-GCN est supérieur aux règles codées en dur ?**
> 
> - **Règle codée** : "Si épaules sous les genoux → chute" → Trop de faux positifs (la personne se penche pour ramasser quelque chose)
> - **ST-GCN** : Analyse la **séquence temporelle** du squelette sur 1-2 secondes → Comprend la **dynamique** du mouvement (vitesse de descente, angle, trajectoire) → Distingue une chute d'un simple mouvement de flexion

**Comment ça fonctionne :**

```
Frame t-29  Frame t-28  ...  Frame t-1   Frame t
   🧍          🧍               🧍         🧎
   │           │                │          │
   └─── Le ST-GCN analyse l'évolution du squelette ───┘
                    sur 30 frames
                         │
                         ▼
              [Marche: 5%, Chute: 93%, ...]
```

Le ST-GCN traite le squelette comme un **graphe** :
- **Nœuds** = les 17 keypoints
- **Arêtes spatiales** = les connexions du squelette (épaule-coude, etc.)
- **Arêtes temporelles** = le même keypoint entre 2 frames successives

---

### 4.2 — Télécharger et Préparer le Modèle ST-GCN Pré-entraîné

> [!IMPORTANT]
> Plusieurs options existent pour le modèle. On recommande **2s-AGCN** ou le ST-GCN original, pré-entraîné sur **NTU-RGB+D** (60 classes d'actions humaines incluant chutes, coups, etc.).

**Option A — ST-GCN Original (Recommandé pour commencer) :**

```powershell
# Cloner le dépôt ST-GCN dans un dossier temporaire
cd C:\Users\ilyas\Documents\CCTV_AI_DEEP_SECU
git clone https://github.com/yysijie/st-gcn.git external/st-gcn
```

**Option B — Utiliser pyskl (Plus moderne, plus de modèles) :**

```powershell
pip install pyskl
# OU
git clone https://github.com/kennymckormick/pyskl.git external/pyskl
```

**Option C — Implémenter un ST-GCN simplifié (Contrôle total) :**

Nous allons implémenter notre propre version simplifiée pour avoir le contrôle total et éviter les dépendances lourdes.

**✅ Critère de validation 4.2** :
```powershell
# Vérifier que le dossier externe existe :
dir external\
# OU vérifier que pyskl est installé :
python -c "import pyskl; print('pyskl OK')"
```

---

### 4.3 — Créer le Module de Buffer Temporel (`src/behavior/skeleton_buffer.py`)

**Actions :**

Créer `src/behavior/skeleton_buffer.py` :

```python
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
```

**✅ Critère de validation 4.3** :
```python
python -c "
import numpy as np
from src.behavior.skeleton_buffer import SkeletonBuffer, MultiPersonBuffer

# Test SkeletonBuffer
buf = SkeletonBuffer(buffer_size=30)
assert not buf.is_ready

# Remplir avec des données simulées
for i in range(30):
    kpts = np.random.randn(17, 2).astype(np.float32) * 100
    kpts += [960, 540]  # Centrer dans une image 1920x1080
    buf.add_frame(kpts)

assert buf.is_ready
data = buf.to_numpy()
assert data.shape == (2, 30, 17), f'Shape attendu (2, 30, 17), obtenu {data.shape}'
print(f'✅ SkeletonBuffer OK — shape: {data.shape}')

# Test MultiPersonBuffer
multi = MultiPersonBuffer()
for frame in range(35):
    multi.update(1, np.random.randn(17, 2).astype(np.float32))
    multi.update(2, np.random.randn(17, 2).astype(np.float32))

ready = multi.get_ready_buffers()
assert len(ready) == 2
print(f'✅ MultiPersonBuffer OK — {len(ready)} buffers prêts')
print(f'  Stats : {multi.get_stats()}')
"
```

---

### 4.4 — Implémenter le Modèle ST-GCN (`src/models/stgcn/model.py`)

> [!IMPORTANT]
> Nous implémentons un ST-GCN simplifié mais fonctionnel. Pour un système de production, utilisez un modèle pré-entraîné sur NTU-RGB+D.

**Actions :**

Créer `src/models/stgcn/model.py` :

```python
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
# Définition des connexions du squelette
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
        self.A = A  # (V, V) matrice d'adjacence
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
        # x @ A^T → message passing sur le graphe
        x = torch.einsum('nctv,vw->nctw', x, self.A.to(x.device))
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
        → 3 blocs ST-GCN (64, 128, 256 channels)
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
        self.register_buffer('A', torch.tensor(A_np, dtype=torch.float32))
        
        # Normalisation d'entrée
        self.data_bn = nn.BatchNorm1d(in_channels * num_keypoints)
        
        # Couches ST-GCN
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, self.A),
            STGCNBlock(64, 64, self.A),
            STGCNBlock(64, 128, self.A, stride=2),    # T: 30 → 15
            STGCNBlock(128, 128, self.A),
            STGCNBlock(128, 256, self.A, stride=2),    # T: 15 → 8
            STGCNBlock(256, 256, self.A),
        ])
        
        # Classificateur
        self.fc = nn.Linear(256, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, V) = (batch, 2, 30, 17)
            
        Returns:
            (N, num_classes) — probabilités pour chaque action
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
```

**✅ Critère de validation 4.4** :
```python
python -c "
import torch
from src.models.stgcn.model import STGCN, build_adjacency_matrix

# Vérifier la matrice d'adjacence
A = build_adjacency_matrix()
assert A.shape == (17, 17)
print(f'✅ Matrice d adjacence : {A.shape}')

# Créer le modèle
model = STGCN(in_channels=2, num_classes=8)
print(f'✅ Modèle ST-GCN créé')
print(f'  Paramètres : {sum(p.numel() for p in model.parameters()):,}')

# Test avec des données simulées
x = torch.randn(1, 2, 30, 17)  # 1 personne, 2 canaux, 30 frames, 17 keypoints
out = model(x)
assert out.shape == (1, 8), f'Shape attendu (1, 8), obtenu {out.shape}'
print(f'✅ Forward pass OK — sortie : {out.shape}')

# Test predict
preds = model.predict(x)
print(f'✅ Prédictions : {preds}')
total_prob = sum(preds.values())
print(f'  Somme des probabilités : {total_prob:.4f} (≈ 1.0)')
"
```

---

### 4.5 — Créer le Classificateur d'Actions (`src/behavior/action_classifier.py`)

**Actions :**

Créer `src/behavior/action_classifier.py` :

```python
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
            state_dict = torch.load(weights_path, map_location=self.device)
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
```

**✅ Critère de validation 4.5** :
```python
python -c "
import numpy as np
from src.behavior.action_classifier import ActionClassifier

classifier = ActionClassifier(device='cuda')
print(f'✅ ActionClassifier initialisé sur {classifier.device}')

# Simuler 35 frames pour 2 personnes
for frame in range(35):
    kpts1 = np.random.randn(17, 2).astype(np.float32) * 50
    kpts2 = np.random.randn(17, 2).astype(np.float32) * 50
    classifier.update(1, kpts1)
    classifier.update(2, kpts2)

# Inférence
preds = classifier.classify(frame_count=35)
print(f'✅ Prédictions pour {len(preds)} personne(s)')
for tid, pred in preds.items():
    action = classifier.get_action(tid)
    print(f'  ID:{tid} → {action} ({pred[action]:.1%})')

# Vérifier les alertes
alerts = classifier.check_alerts()
print(f'✅ Alertes détectées : {len(alerts)}')
print(f'  Stats : {classifier.get_stats()}')
"
```

---

### 4.6 — Créer le Détecteur de Maraudage (`src/behavior/loitering_detector.py`)

> [!NOTE]
> Le ST-GCN analyse le **mouvement** mais pas le **temps passé à un endroit**. Le détecteur de maraudage complète le ST-GCN avec une règle spatiale basée sur un polygone.

**Actions :**

Créer `src/behavior/loitering_detector.py` :

```python
"""
Détecteur de maraudage (loitering).
Vérifie si une personne reste trop longtemps dans une zone définie
par un polygone.

Le ST-GCN ne gère pas le temps → cette règle spatiale le complète.
"""
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import LOITERING_TIMEOUT


class LoiteringDetector:
    """
    Détecte le maraudage basé sur le temps passé dans une zone.
    
    Algorithme :
    1. Définir un (ou plusieurs) polygone(s) de surveillance
    2. Pour chaque personne, vérifier si son centre est dans le polygone
    3. Si elle y reste plus de LOITERING_TIMEOUT → alerte MARAUDAGE
    """

    def __init__(self, timeout: float = LOITERING_TIMEOUT):
        """
        Args:
            timeout: Temps en secondes avant alerte de maraudage
        """
        self.timeout = timeout
        self.zones: List[np.ndarray] = []  # Liste de polygones
        
        # Suivi du temps par personne par zone
        # {track_id: {"zone_idx": zone_index, "enter_time": float}}
        self._tracking: Dict[int, Dict] = {}

    def add_zone(self, polygon: list):
        """
        Ajoute une zone de surveillance.
        
        Args:
            polygon: Liste de points [(x1,y1), (x2,y2), ...] 
                     définissant le polygone
        """
        poly = np.array(polygon, dtype=np.int32)
        self.zones.append(poly)
        print(f"[LOITERING] Zone ajoutée ({len(poly)} points)")

    def set_default_zones(self, frame_width: int, frame_height: int):
        """
        Crée une zone par défaut couvrant le centre de l'image.
        Utile pour les tests. En production, définir les zones manuellement.
        """
        # Zone centrale (60% de l'image)
        margin_x = int(frame_width * 0.2)
        margin_y = int(frame_height * 0.2)
        default_zone = [
            (margin_x, margin_y),
            (frame_width - margin_x, margin_y),
            (frame_width - margin_x, frame_height - margin_y),
            (margin_x, frame_height - margin_y),
        ]
        self.add_zone(default_zone)

    def is_in_zone(self, point: tuple, zone_idx: int = 0) -> bool:
        """Vérifie si un point est dans la zone spécifiée."""
        if zone_idx >= len(self.zones):
            return False
        result = cv2.pointPolygonTest(self.zones[zone_idx], point, False)
        return result >= 0

    def update(self, track_id: int, center: tuple) -> Optional[Tuple[str, float]]:
        """
        Met à jour le suivi de maraudage pour une personne.
        
        Args:
            track_id: ID de suivi
            center: (x, y) centre de la personne
            
        Returns:
            ("MARAUDAGE", durée_en_secondes) si alerte, ou None
        """
        now = time.time()
        
        # Vérifier chaque zone
        in_any_zone = False
        for zone_idx, zone in enumerate(self.zones):
            if self.is_in_zone(center, zone_idx):
                in_any_zone = True
                
                if track_id not in self._tracking:
                    self._tracking[track_id] = {
                        "zone_idx": zone_idx,
                        "enter_time": now
                    }
                else:
                    duration = now - self._tracking[track_id]["enter_time"]
                    if duration >= self.timeout:
                        return ("MARAUDAGE", duration)
                break
        
        # Si pas dans une zone, réinitialiser le compteur
        if not in_any_zone and track_id in self._tracking:
            del self._tracking[track_id]
        
        return None

    def cleanup_lost_ids(self, active_ids: set):
        """Nettoie les IDs perdus."""
        lost = set(self._tracking.keys()) - active_ids
        for tid in lost:
            del self._tracking[tid]

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Dessine les zones de surveillance sur la frame."""
        overlay = frame.copy()
        for zone in self.zones:
            cv2.polylines(overlay, [zone], True, (0, 200, 255), 2)
            # Zone semi-transparente
            cv2.fillPoly(overlay, [zone], (0, 200, 255))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        return frame

    def get_stats(self) -> dict:
        """Statistiques du détecteur."""
        return {
            "num_zones": len(self.zones),
            "tracked_persons": len(self._tracking),
            "durations": {
                tid: f"{time.time() - info['enter_time']:.0f}s"
                for tid, info in self._tracking.items()
            }
        }
```

**✅ Critère de validation 4.6** :
```python
python -c "
import numpy as np
from src.behavior.loitering_detector import LoiteringDetector

detector = LoiteringDetector(timeout=5)  # 5s pour le test
detector.set_default_zones(1920, 1080)

print(f'✅ LoiteringDetector initialisé')
print(f'  Zones : {detector.get_stats()[\"num_zones\"]}')

# Test point dans la zone
assert detector.is_in_zone((960, 540), 0), 'Le centre devrait être dans la zone'
assert not detector.is_in_zone((10, 10), 0), 'Le coin devrait être hors zone'
print('✅ Détection de zone OK')

# Test maraudage (simulation rapide)
import time
result = detector.update(1, (960, 540))
assert result is None, 'Pas de maraudage immédiat'
time.sleep(6)  # Attendre plus que le timeout
result = detector.update(1, (960, 540))
assert result is not None
print(f'✅ Maraudage détecté : {result}')
"
```

---

### 4.7 — Créer le Test Complet ST-GCN

**Actions :**

Créer `tests/test_stgcn.py` :

```python
"""
Test complet du système ST-GCN intégré au pipeline.
"""
import cv2
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline.detector import PoseDetector
from src.behavior.action_classifier import ActionClassifier
from src.behavior.loitering_detector import LoiteringDetector
from src.utils.drawing import draw_detections, draw_fps, draw_alert


def test_stgcn_pipeline(source=0):
    """Test le pipeline complet avec ST-GCN et détection de maraudage."""
    
    print("[TEST ST-GCN] Initialisation...")
    detector = PoseDetector()
    classifier = ActionClassifier(device="cuda")
    loitering = LoiteringDetector(timeout=30)  # 30s pour le test
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir : {source}")
        return False
    
    # Zones de maraudage
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    loitering.set_default_zones(w, h)
    
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0
    
    print("[TEST ST-GCN] Démarrage ('q' pour quitter)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        # 1. Détection
        detections = detector.detect(frame)
        active_ids = {d.track_id for d in detections}
        
        # 2. Mise à jour des buffers + classification
        for det in detections:
            classifier.update(det.track_id, det.keypoints_xy)
        
        predictions = classifier.classify(detector.frame_count)
        
        # 3. Appliquer les actions aux détections
        for det in detections:
            det.action = classifier.get_action(det.track_id)
            
            # 4. Vérifier le maraudage
            loiter_result = loitering.update(det.track_id, det.center)
            if loiter_result:
                det.action = f"MARAUDAGE ({loiter_result[1]:.0f}s)"
        
        # 5. Vérifier les alertes ST-GCN
        alerts = classifier.check_alerts()
        
        # 6. Affichage
        frame = loitering.draw_zones(frame)
        annotated = draw_detections(frame, detections)
        annotated = draw_fps(annotated, current_fps)
        
        for alert_tid, alert_action, alert_conf in alerts:
            draw_alert(annotated, f"{alert_action} (ID:{alert_tid} - {alert_conf:.0%})")
        
        # Nettoyage
        classifier.cleanup_lost_ids(active_ids)
        loitering.cleanup_lost_ids(active_ids)
        
        # FPS
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()
        
        cv2.imshow("CCTV AI - Test ST-GCN", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[TEST ST-GCN] Stats classificateur : {classifier.get_stats()}")
    print(f"[TEST ST-GCN] Stats maraudage : {loitering.get_stats()}")
    return True


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    success = test_stgcn_pipeline(source)
    print(f"\n{'✅ TEST RÉUSSI' if success else '❌ TEST ÉCHOUÉ'}")
```

**✅ Critère de validation 4.7** :
```powershell
python tests/test_stgcn.py

# DOIT :
# - Afficher les boîtes + squelettes + IDs (étape 1)
# - Afficher l'action prédite pour chaque personne (ex: "marcher", "immobile")
# - Afficher les zones de surveillance en orange semi-transparent
# - Afficher une alerte rouge si une action dangereuse est détectée
# - Les actions changent dynamiquement avec le mouvement de la personne
```

---

## 📝 Note sur l'Entraînement du Modèle

> [!WARNING]
> Le modèle créé dans cette étape a des **poids aléatoires**. Pour un système de production, deux options :
> 
> **Option 1 — Transfer Learning (Recommandé)** :
> 1. Télécharger un modèle ST-GCN pré-entraîné sur NTU-RGB+D (60 classes)
> 2. Remplacer la dernière couche FC pour vos 8 classes
> 3. Fine-tuner sur vos propres données
> 
> **Option 2 — Utiliser un modèle pré-entraîné tel quel** :
> 1. Télécharger depuis [pyskl](https://github.com/kennymckormick/pyskl) ou [st-gcn](https://github.com/yysijie/st-gcn)
> 2. Adapter le mapping des classes de sortie
> 
> L'entraînement n'est PAS couvert dans ces étapes mais peut être ajouté comme Phase 6.

---

## ✅ Checklist de Validation Finale — Étape 4

| # | Critère | Commande/Action | Status |
|---|---------|-----------------|--------|
| 4.2 | Modèle ST-GCN disponible | Dépôt cloné ou pyskl installé | ⬜ |
| 4.3 | `skeleton_buffer.py` testé | Buffer (2,30,17) créé correctement | ⬜ |
| 4.4 | `model.py` (ST-GCN) testé | Forward pass (1,2,30,17)→(1,8) | ⬜ |
| 4.5 | `action_classifier.py` testé | Prédictions pour N personnes | ⬜ |
| 4.6 | `loitering_detector.py` testé | Maraudage détecté après timeout | ⬜ |
| 4.7 | Test pipeline complet | Vidéo avec actions + zones + alertes | ⬜ |

**Vérifications fonctionnelles obligatoires :**
- [ ] Le buffer se remplit correctement (30 frames = 1s à 30fps)
- [ ] Le ST-GCN produit des probabilités valides (somme ≈ 1.0)
- [ ] Les actions sont affichées à côté de chaque personne
- [ ] Les zones de maraudage sont visibles et fonctionnelles
- [ ] Les alertes de maraudage se déclenchent après le timeout
- [ ] Le système ne crash pas quand des personnes entrent/sortent du champ

> [!CAUTION]
> **Cette étape est la plus complexe.** Testez chaque sous-module individuellement avant le test intégré. Si le ST-GCN ne fonctionne pas, le reste du pipeline (Étapes 1-3) reste valide.

---

**⬅️ Étape précédente : [etape_3.md](etape_3.md)**
**➡️ Étape suivante : [etape_5.md](etape_5.md) — Optimisation RTX 3080 Ti (Pipeline Asynchrone)**
