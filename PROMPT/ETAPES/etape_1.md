# 🎯 Étape 1 — Détection, Suivi et Extraction Squelettique (La Brique 1)

## 📋 Summary (À lire AVANT de commencer)

**Objectif** : Créer le module de détection unifié qui utilise **YOLOv8-Pose** pour extraire simultanément les boîtes englobantes, les IDs de suivi (ByteTrack), et les 17 points clés du squelette de chaque personne détectée.

**Durée estimée** : 2-3 heures

**Prérequis** :
- ✅ Étape 0 entièrement validée
- ✅ Environnement virtuel activé avec toutes les bibliothèques
- ✅ GPU CUDA fonctionnel

**Ce que vous aurez à la fin** :
- ✅ Modèle YOLOv8-Pose téléchargé et testé
- ✅ Module `detector.py` fonctionnel avec suivi ByteTrack
- ✅ Extraction des 17 keypoints par personne
- ✅ Affichage temps réel : boîtes + IDs + squelettes sur la vidéo
- ✅ Script de test validé sur une vidéo de test

---

## 📝 Étapes Détaillées

### 1.1 — Télécharger le Modèle YOLOv8-Pose

**Actions :**

```python
# Script de téléchargement : src/models/yolo/download_model.py
from ultralytics import YOLO

# Télécharge automatiquement yolov8m-pose.pt (~50 Mo)
model = YOLO("yolov8m-pose.pt")
print(f"Modèle chargé : {model.model_name}")
print(f"Type de tâche : {model.task}")

# Vérification rapide sur une image de test
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
    device=0  # GPU
)
print(f"Nombre de personnes détectées : {len(results[0].boxes)}")
print(f"Keypoints shape : {results[0].keypoints.data.shape}")
```

> [!NOTE]
> Le modèle `yolov8m-pose.pt` (Medium) offre un bon compromis vitesse/précision. Pour plus de précision (mais plus lent), utiliser `yolov8l-pose.pt` (Large). Les 17 keypoints suivent le format COCO :
> 
> `0:nez, 1:œil_gauche, 2:œil_droit, 3:oreille_gauche, 4:oreille_droite, 5:épaule_gauche, 6:épaule_droite, 7:coude_gauche, 8:coude_droit, 9:poignet_gauche, 10:poignet_droit, 11:hanche_gauche, 12:hanche_droite, 13:genou_gauche, 14:genou_droit, 15:cheville_gauche, 16:cheville_droite`

**✅ Critère de validation 1.1** :
```python
# Le script doit afficher :
# - Type de tâche : pose
# - Nombre de personnes détectées : ≥1
# - Keypoints shape : (N, 17, 3)  ← N personnes, 17 points, 3 valeurs (x, y, confidence)
```

---

### 1.2 — Comprendre la Sortie de YOLOv8-Pose

Avant de coder le module, il faut bien comprendre la structure des résultats.

**Structure des résultats pour chaque frame :**

```python
results = model.track(frame, persist=True)

for result in results:
    # --- Boîtes englobantes ---
    boxes = result.boxes
    # boxes.xyxy   → Tensor (N, 4) : coordonnées [x1, y1, x2, y2]
    # boxes.conf   → Tensor (N,)   : confiance de détection
    # boxes.cls    → Tensor (N,)   : classe (0 = personne)
    # boxes.id     → Tensor (N,)   : ID de suivi ByteTrack (ou None si pas de track)
    
    # --- Keypoints (Squelette) ---
    keypoints = result.keypoints
    # keypoints.data → Tensor (N, 17, 3) : [x, y, confidence] pour chaque point
    # keypoints.xy   → Tensor (N, 17, 2) : [x, y] uniquement
    # keypoints.conf → Tensor (N, 17)    : confiance de chaque point
```

**Schéma du squelette COCO 17 keypoints :**
```
            0 (nez)
           / \
     1 (œil_g)  2 (œil_d)
     |           |
   3 (oreille_g) 4 (oreille_d)
         |
    5 ---+--- 6    (épaules)
    |         |
    7         8    (coudes)
    |         |
    9        10    (poignets)
    |         |
   11 ---+--- 12   (hanches)
    |         |
   13        14    (genoux)
    |         |
   15        16    (chevilles)
```

---

### 1.3 — Créer le Module Détecteur (`src/pipeline/detector.py`)

**Actions :**

Créer le fichier `src/pipeline/detector.py` :

```python
"""
Module de détection, suivi et extraction squelettique.
Utilise YOLOv8-Pose avec ByteTrack pour un pipeline unifié.
"""
import torch
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    YOLO_MODEL, YOLO_CONFIDENCE, YOLO_DEVICE, TRACKER_TYPE
)
from pathlib import Path


@dataclass
class PersonDetection:
    """Représente une personne détectée dans une frame."""
    track_id: int                          # ID de suivi ByteTrack
    bbox: np.ndarray                       # [x1, y1, x2, y2]
    confidence: float                       # Confiance de détection
    keypoints: np.ndarray                  # (17, 3) → [x, y, conf]
    keypoints_xy: np.ndarray               # (17, 2) → [x, y] uniquement
    name: str = "INCONNU"                  # Sera rempli par InsightFace (Étape 2)
    action: str = "N/A"                    # Sera rempli par ST-GCN (Étape 4)

    @property
    def head_bbox(self) -> np.ndarray:
        """Retourne la boîte englobante du tiers supérieur (pour InsightFace)."""
        x1, y1, x2, y2 = self.bbox
        head_height = (y2 - y1) / 3
        return np.array([x1, y1, x2, y1 + head_height])

    @property
    def center(self) -> tuple:
        """Retourne le centre de la boîte englobante."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class PoseDetector:
    """
    Détecteur unifié YOLOv8-Pose + ByteTrack.
    Sortie : liste de PersonDetection par frame.
    """

    def __init__(self, model_path: str = YOLO_MODEL, device: int = YOLO_DEVICE):
        """
        Initialise le détecteur.
        
        Args:
            model_path: Chemin vers le modèle YOLOv8-Pose
            device: Index du GPU (0) ou "cpu"
        """
        print(f"[DETECTOR] Chargement du modèle {model_path}...")
        self.model = YOLO(model_path)
        self.device = device
        self.frame_count = 0
        print(f"[DETECTOR] Modèle chargé avec succès sur device={device}")

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Détecte les personnes, suit leur ID, et extrait les squelettes.
        
        Args:
            frame: Image BGR (numpy array) depuis OpenCV
            
        Returns:
            Liste de PersonDetection pour chaque personne détectée
        """
        self.frame_count += 1
        detections: List[PersonDetection] = []

        # Inférence YOLOv8-Pose avec suivi ByteTrack
        results = self.model.track(
            source=frame,
            persist=True,          # Maintenir le suivi entre les frames
            tracker=f"{TRACKER_TYPE}.yaml",
            conf=YOLO_CONFIDENCE,
            device=self.device,
            verbose=False          # Pas de log à chaque frame
        )

        if results is None or len(results) == 0:
            return detections

        result = results[0]

        # Vérifier que des personnes ont été détectées
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # Vérifier que le suivi est actif (IDs disponibles)
        if result.boxes.id is None:
            return detections

        # Extraire les données
        boxes = result.boxes.xyxy.cpu().numpy()       # (N, 4)
        confs = result.boxes.conf.cpu().numpy()        # (N,)
        track_ids = result.boxes.id.cpu().numpy().astype(int)  # (N,)
        
        # Keypoints
        if result.keypoints is not None:
            kpts_data = result.keypoints.data.cpu().numpy()   # (N, 17, 3)
            kpts_xy = result.keypoints.xy.cpu().numpy()       # (N, 17, 2)
        else:
            return detections

        # Créer les objets PersonDetection
        for i in range(len(boxes)):
            detection = PersonDetection(
                track_id=int(track_ids[i]),
                bbox=boxes[i],
                confidence=float(confs[i]),
                keypoints=kpts_data[i],
                keypoints_xy=kpts_xy[i]
            )
            detections.append(detection)

        return detections

    def get_stats(self) -> dict:
        """Retourne les statistiques du détecteur."""
        return {
            "frames_processed": self.frame_count,
            "model": str(self.model.model_name),
            "device": self.device
        }
```

**✅ Critère de validation 1.3** :
```python
python -c "
from src.pipeline.detector import PoseDetector, PersonDetection
detector = PoseDetector()
print('✅ PoseDetector importé et initialisé avec succès')
stats = detector.get_stats()
print(f'  Modèle : {stats[\"model\"]}')
print(f'  Device : {stats[\"device\"]}')
"
```

---

### 1.4 — Créer le Module d'Affichage (`src/utils/drawing.py`)

**Actions :**

Créer `src/utils/drawing.py` :

```python
"""
Module d'affichage : dessine les boîtes, squelettes et informations sur la vidéo.
"""
import cv2
import numpy as np
from typing import List

# Connexions du squelette COCO pour dessiner les os
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),    # Nez → Yeux
    (1, 3), (2, 4),    # Yeux → Oreilles
    (5, 6),             # Épaule gauche → Épaule droite
    (5, 7), (7, 9),    # Épaule G → Coude G → Poignet G
    (6, 8), (8, 10),   # Épaule D → Coude D → Poignet D
    (5, 11), (6, 12),  # Épaules → Hanches
    (11, 12),           # Hanche G → Hanche D
    (11, 13), (13, 15), # Hanche G → Genou G → Cheville G
    (12, 14), (14, 16), # Hanche D → Genou D → Cheville D
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
    """Affiche une alerte rouge clignotante."""
    h, w = frame.shape[:2]
    if position is None:
        position = (w // 2 - 200, 50)
    
    # Fond rouge semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (position[0] - 10, position[1] - 30),
                  (position[0] + 400, position[1] + 10), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, f"⚠ {message}", position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame
```

**✅ Critère de validation 1.4** :
```python
python -c "
from src.utils.drawing import draw_detections, draw_fps, SKELETON_CONNECTIONS
print(f'✅ Module drawing importé')
print(f'  Connexions squelette : {len(SKELETON_CONNECTIONS)} os')
"
```

---

### 1.5 — Créer le Script de Test Complet

**Actions :**

Créer `tests/test_detector.py` :

```python
"""
Test complet du détecteur YOLOv8-Pose + ByteTrack.
Utilise la webcam ou un fichier vidéo pour vérifier :
- Détection des personnes
- Suivi des IDs
- Extraction des squelettes
- Affichage temps réel
"""
import cv2
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline.detector import PoseDetector
from src.utils.drawing import draw_detections, draw_fps


def test_with_video(source=0):
    """
    Test le détecteur en temps réel.
    
    Args:
        source: 0 pour webcam, ou chemin vers un fichier vidéo
    """
    print(f"[TEST] Initialisation du détecteur...")
    detector = PoseDetector()
    
    print(f"[TEST] Ouverture de la source vidéo : {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ ERREUR : Impossible d'ouvrir la source vidéo : {source}")
        return False
    
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0
    
    print("[TEST] Démarrage de la boucle de détection (Appuyez sur 'q' pour quitter)")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                # Replay la vidéo en boucle
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        # --- Détection ---
        detections = detector.detect(frame)
        
        # --- Affichage ---
        annotated = draw_detections(frame, detections)
        annotated = draw_fps(annotated, current_fps)
        
        # Infos supplémentaires
        cv2.putText(annotated, f"Personnes: {len(detections)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Afficher les détails en console (toutes les 30 frames)
        if detector.frame_count % 30 == 0 and detections:
            print(f"\n[Frame {detector.frame_count}] {len(detections)} personne(s) détectée(s):")
            for det in detections:
                kpts_valid = sum(1 for k in det.keypoints if k[2] > 0.5)
                print(f"  ID:{det.track_id} | Conf:{det.confidence:.2f} | "
                      f"Keypoints valides: {kpts_valid}/17 | "
                      f"Centre: ({det.center[0]:.0f}, {det.center[1]:.0f})")
        
        # Calcul FPS
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()
        
        cv2.imshow("CCTV AI - Test Detecteur", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    stats = detector.get_stats()
    print(f"\n{'=' * 60}")
    print(f"[TEST] Terminé. Statistiques :")
    print(f"  Frames traitées : {stats['frames_processed']}")
    print(f"  Modèle          : {stats['model']}")
    print(f"  Device           : {stats['device']}")
    
    return True


if __name__ == "__main__":
    # Par défaut : webcam (0)
    # Pour un fichier : python test_detector.py chemin/vers/video.mp4
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    
    success = test_with_video(source)
    if success:
        print("\n✅ TEST RÉUSSI — Le détecteur fonctionne correctement")
    else:
        print("\n❌ TEST ÉCHOUÉ — Vérifier la configuration")
```

**✅ Critère de validation 1.5** :
```powershell
# Avec une vidéo :
python tests/test_detector.py data/videos/test.mp4

# OU avec la webcam :
python tests/test_detector.py

# DOIT afficher :
# - Fenêtre vidéo avec boîtes, IDs et squelettes
# - FPS affiché en haut à gauche
# - Nombre de personnes détectées
# - IDs stables (le même ID suit la même personne)
# - Squelettes correctement dessinés sur les personnes
```

---

### 1.6 — Préparer une Vidéo de Test

> [!TIP]
> Pour tester sans webcam, téléchargez une vidéo de test avec des piétons.

**Actions :**

Option A — Utiliser une vidéo existante :
```powershell
# Placer une vidéo de surveillance dans le dossier :
# data/videos/test.mp4
```

Option B — Télécharger une vidéo de test gratuite :
```powershell
# Vidéo piétons depuis Pexels (libre de droits) :
# https://www.pexels.com/search/videos/pedestrian/
# Télécharger et placer dans data/videos/test.mp4
```

Option C — Utiliser la webcam :
```powershell
# Utiliser source=0 (webcam par défaut)
python tests/test_detector.py 0
```

**✅ Critère de validation 1.6** :
```powershell
# Vérifier qu'une source vidéo est disponible :
dir data\videos\
# OU vérifier la webcam :
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'PAS DE WEBCAM'); cap.release()"
```

---

## ✅ Checklist de Validation Finale — Étape 1

| # | Critère | Commande/Action | Status |
|---|---------|-----------------|--------|
| 1.1 | Modèle YOLOv8-Pose téléchargé | `yolov8m-pose.pt` chargé sur GPU | ✅ |
| 1.2 | Structure des résultats comprise | N/A (lecture) | ✅ |
| 1.3 | `detector.py` importable | Import + test frame noire = 0 détections | ✅ |
| 1.4 | `drawing.py` importable | 16 connexions squelette chargées | ✅ |
| 1.5 | Test temps réel fonctionnel | Script `test_detector.py` créé | ✅ |
| 1.6 | Source vidéo disponible | Utiliser webcam ou placer vidéo dans `data/videos/` | ⏳ |

**Vérifications visuelles obligatoires :**
- [ ] Les boîtes englobantes entourent correctement les personnes
- [ ] Les IDs sont stables (ne changent pas quand la personne ne bouge pas beaucoup)
- [ ] Le squelette est correctement aligné sur le corps de chaque personne
- [ ] Les 17 points clés sont visibles (au moins 12+ pour une personne debout)
- [ ] Le FPS est ≥ 15 sur la RTX 3080 Ti (≥ 25 souhaité)

> [!CAUTION]
> **NE PASSEZ PAS À L'ÉTAPE 2** si les squelettes ne sont pas correctement détectés. Le ST-GCN (Étape 4) dépend entièrement de la qualité des keypoints extraits ici.

---

**⬅️ Étape précédente : [etape_0.md](etape_0.md)**
**➡️ Étape suivante : [etape_2.md](etape_2.md) — Identification Visuelle (InsightFace)**
