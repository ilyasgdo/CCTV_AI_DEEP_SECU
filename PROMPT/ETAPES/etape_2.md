# 👤 Étape 2 — L'Identification Visuelle (InsightFace)

## 📋 Summary (À lire AVANT de commencer)

**Objectif** : Implémenter la reconnaissance faciale avec **InsightFace** pour identifier les personnes détectées via une **liste blanche**. Chaque personne se verra attribuer un nom ("Thomas", "Sarah") ou le label "INCONNU".

**Durée estimée** : 2-3 heures

**Prérequis** :
- ✅ Étape 0 ET Étape 1 entièrement validées
- ✅ Le détecteur YOLOv8-Pose fonctionne avec les boîtes englobantes
- ✅ Au moins 2-3 photos de visages pour la liste blanche (photos claires, de face)

**Ce que vous aurez à la fin** :
- ✅ Module d'encodage des visages de référence (whitelist)
- ✅ Module de reconnaissance faciale en temps réel
- ✅ Stratégie "paresseuse" : scan uniquement quand nécessaire
- ✅ Labels "Thomas" ou "INCONNU" affichés sur la vidéo
- ✅ Test complet avec identification en temps réel

---

## 📝 Étapes Détaillées

### 2.1 — Préparer les Photos de la Liste Blanche

> [!IMPORTANT]
> La qualité des photos de référence détermine directement la fiabilité de la reconnaissance. Utilisez des photos claires, bien éclairées, de face, sans lunettes de soleil.

**Actions :**

1. Placer les photos dans `data/whitelist_photos/` avec le format de nommage :
   ```
   data/whitelist_photos/
   ├── thomas_1.jpg
   ├── thomas_2.jpg     ← Plusieurs photos par personne = meilleure précision
   ├── sarah_1.jpg
   ├── sarah_2.jpg
   └── admin_1.jpg
   ```

2. **Convention de nommage** : `{prenom}_{numero}.jpg`
   - Le prénom (avant le `_`) sera utilisé comme label d'identification
   - Fournir 2-5 photos par personne (angles légèrement différents)

**✅ Critère de validation 2.1** :
```powershell
dir data\whitelist_photos\
# DOIT contenir au moins 1 photo par personne à identifier
# Format : prenom_numero.jpg
```

---

### 2.2 — Créer le Module d'Encodage (`src/face_recognition/encoder.py`)

**Actions :**

Créer `src/face_recognition/encoder.py` :

```python
"""
Module d'encodage des visages.
Convertit les photos de référence en vecteurs d'embedding (fichiers .npy).
Utilise InsightFace (ArcFace) pour l'extraction des features.
"""
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import insightface
from insightface.app import FaceAnalysis

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import WHITELIST_DIR, FACE_RECOGNITION_THRESHOLD


class FaceEncoder:
    """
    Encode les visages de référence en vecteurs d'embedding.
    """

    def __init__(self, det_size: tuple = (640, 640)):
        """
        Initialise InsightFace.
        
        Args:
            det_size: Taille de détection (plus grand = plus précis mais plus lent)
        """
        print("[ENCODER] Initialisation d'InsightFace...")
        self.app = FaceAnalysis(
            name="buffalo_l",  # Modèle de haute qualité
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=det_size)
        print("[ENCODER] InsightFace prêt.")

    def encode_photo(self, image_path: str) -> np.ndarray:
        """
        Encode une photo en vecteur d'embedding.
        
        Args:
            image_path: Chemin vers la photo
            
        Returns:
            Vecteur d'embedding (512,) ou None si aucun visage détecté
        """
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  ⚠ Impossible de lire : {image_path}")
            return None

        faces = self.app.get(img)
        if len(faces) == 0:
            print(f"  ⚠ Aucun visage détecté dans : {image_path}")
            return None

        if len(faces) > 1:
            print(f"  ⚠ {len(faces)} visages détectés dans {image_path}, "
                  f"utilisation du plus grand")
            # Prendre le visage avec la plus grande boîte
            faces = sorted(faces, key=lambda f: 
                          (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                          reverse=True)

        embedding = faces[0].embedding  # Vecteur (512,)
        # Normaliser le vecteur
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def build_whitelist(self, photos_dir: str = None) -> Dict[str, np.ndarray]:
        """
        Construit la liste blanche en encodant toutes les photos de référence.
        
        Convention : les fichiers sont nommés {prenom}_{numero}.jpg
        Moyenne des embeddings par personne pour plus de robustesse.
        
        Args:
            photos_dir: Dossier contenant les photos (défaut: config.WHITELIST_DIR)
            
        Returns:
            Dict {nom: embedding_moyen (512,)}
        """
        if photos_dir is None:
            photos_dir = WHITELIST_DIR

        photos_dir = Path(photos_dir)
        if not photos_dir.exists():
            print(f"❌ Dossier introuvable : {photos_dir}")
            return {}

        print(f"\n[ENCODER] Construction de la liste blanche depuis : {photos_dir}")
        print("=" * 50)

        # Regrouper les photos par nom
        name_embeddings: Dict[str, List[np.ndarray]] = {}
        
        for photo_path in sorted(photos_dir.glob("*.jpg")):
            # Extraire le nom depuis le fichier (ex: "thomas_1.jpg" → "thomas")
            name = photo_path.stem.rsplit("_", 1)[0].capitalize()
            
            print(f"  Encodage : {photo_path.name} → {name}...", end=" ")
            embedding = self.encode_photo(photo_path)
            
            if embedding is not None:
                if name not in name_embeddings:
                    name_embeddings[name] = []
                name_embeddings[name].append(embedding)
                print("✅")
            else:
                print("❌")

        # Calculer la moyenne des embeddings par personne
        whitelist: Dict[str, np.ndarray] = {}
        for name, embeddings in name_embeddings.items():
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)  # Re-normaliser
            whitelist[name] = mean_emb
            print(f"  → {name} : {len(embeddings)} photo(s) encodée(s)")

        print(f"\n[ENCODER] Liste blanche : {len(whitelist)} personne(s)")
        return whitelist

    def save_whitelist(self, whitelist: Dict[str, np.ndarray], 
                       output_dir: str = None):
        """
        Sauvegarde la liste blanche en fichiers .npy.
        
        Args:
            whitelist: Dict {nom: embedding}
            output_dir: Dossier de sortie (défaut: src/face_recognition/whitelist/)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "whitelist"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, embedding in whitelist.items():
            path = output_dir / f"{name.lower()}.npy"
            np.save(path, embedding)
            print(f"  Sauvegardé : {path}")

        # Sauvegarder aussi la liste des noms
        names_path = output_dir / "names.npy"
        np.save(names_path, list(whitelist.keys()))
        print(f"  Liste des noms : {names_path}")

    def load_whitelist(self, whitelist_dir: str = None) -> Dict[str, np.ndarray]:
        """
        Charge la liste blanche depuis les fichiers .npy.
        
        Returns:
            Dict {nom: embedding}
        """
        if whitelist_dir is None:
            whitelist_dir = Path(__file__).parent / "whitelist"
        
        whitelist_dir = Path(whitelist_dir)
        if not whitelist_dir.exists():
            print(f"⚠ Dossier whitelist introuvable : {whitelist_dir}")
            return {}

        whitelist = {}
        names_path = whitelist_dir / "names.npy"
        
        if names_path.exists():
            names = np.load(names_path, allow_pickle=True)
            for name in names:
                emb_path = whitelist_dir / f"{name.lower()}.npy"
                if emb_path.exists():
                    whitelist[name] = np.load(emb_path)
                    
        print(f"[ENCODER] Whitelist chargée : {len(whitelist)} personne(s)")
        return whitelist
```

**✅ Critère de validation 2.2** :
```python
python -c "
from src.face_recognition.encoder import FaceEncoder
encoder = FaceEncoder()
print('✅ FaceEncoder initialisé avec succès')

# Construire et sauvegarder la whitelist
whitelist = encoder.build_whitelist()
if whitelist:
    encoder.save_whitelist(whitelist)
    print(f'✅ Whitelist créée : {len(whitelist)} personne(s)')
    
    # Recharger pour vérifier
    loaded = encoder.load_whitelist()
    assert len(loaded) == len(whitelist), 'Erreur de rechargement !'
    print('✅ Whitelist rechargée avec succès')
else:
    print('⚠ Aucune photo trouvée dans data/whitelist_photos/')
"
```

---

### 2.3 — Créer le Module de Matching (`src/face_recognition/matcher.py`)

**Actions :**

Créer `src/face_recognition/matcher.py` :

```python
"""
Module de comparaison des visages.
Compare un visage détecté aux embeddings de la liste blanche.
Intègre la stratégie "paresseuse" pour économiser le GPU.
"""
import numpy as np
import cv2
import time
from typing import Optional, Dict, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    FACE_RECOGNITION_THRESHOLD, 
    FACE_RECOGNITION_INTERVAL,
    FACE_CONFIDENCE_LOCK
)
from src.face_recognition.encoder import FaceEncoder


class FaceMatcher:
    """
    Gère la reconnaissance faciale avec stratégie paresseuse.
    
    Stratégie :
    1. Ne scanner que les INCONNUS
    2. Scanner 1 fois toutes les FACE_RECOGNITION_INTERVAL frames
    3. Arrêter de scanner une fois identifié avec ≥ FACE_CONFIDENCE_LOCK
    """

    def __init__(self):
        """Initialise le matcher."""
        self.encoder = FaceEncoder()
        self.whitelist: Dict[str, np.ndarray] = {}
        self._identified: Dict[int, Tuple[str, float]] = {}  # {track_id: (nom, score)}
        self._last_scan: Dict[int, int] = {}  # {track_id: dernier frame scanné}

    def load_whitelist(self):
        """Charge la liste blanche depuis les fichiers .npy."""
        self.whitelist = self.encoder.load_whitelist()
        if not self.whitelist:
            print("⚠ [MATCHER] Liste blanche vide ! Lancer d'abord l'encodage.")

    def should_scan(self, track_id: int, frame_count: int) -> bool:
        """
        Détermine si on doit scanner le visage de cette personne.
        
        Règles :
        - Si déjà identifié avec haute confiance → NON
        - Si scanné trop récemment → NON
        - Sinon → OUI
        """
        # Déjà identifié avec certitude ?
        if track_id in self._identified:
            _, score = self._identified[track_id]
            if score >= FACE_CONFIDENCE_LOCK:
                return False

        # Scanné trop récemment ?
        if track_id in self._last_scan:
            if (frame_count - self._last_scan[track_id]) < FACE_RECOGNITION_INTERVAL:
                return False

        return True

    def identify(self, face_crop: np.ndarray, track_id: int, 
                 frame_count: int) -> Tuple[str, float]:
        """
        Identifie un visage par comparaison avec la liste blanche.
        
        Args:
            face_crop: Image recadrée du visage (BGR)
            track_id: ID de suivi de la personne
            frame_count: Numéro de frame actuel
            
        Returns:
            (nom, score) : ("Thomas", 0.87) ou ("INCONNU", 0.0)
        """
        # Vérifier si on doit scanner
        if not self.should_scan(track_id, frame_count):
            if track_id in self._identified:
                return self._identified[track_id]
            return ("INCONNU", 0.0)

        # Marquer le scan
        self._last_scan[track_id] = frame_count

        # Détecter le visage dans le crop
        faces = self.encoder.app.get(face_crop)
        if len(faces) == 0:
            if track_id in self._identified:
                return self._identified[track_id]
            return ("INCONNU", 0.0)

        # Prendre le visage principal
        query_emb = faces[0].embedding
        query_emb = query_emb / np.linalg.norm(query_emb)

        # Comparer avec la whitelist
        best_name = "INCONNU"
        best_score = 0.0

        for name, ref_emb in self.whitelist.items():
            # Similarité cosinus
            score = float(np.dot(query_emb, ref_emb))
            if score > best_score:
                best_score = score
                best_name = name

        # Seuil de reconnaissance
        if best_score < FACE_RECOGNITION_THRESHOLD:
            result = ("INCONNU", best_score)
        else:
            result = (best_name, best_score)
            self._identified[track_id] = result
            print(f"  🔍 ID:{track_id} identifié comme {best_name} "
                  f"(score: {best_score:.3f})")

        return result

    def get_name(self, track_id: int) -> str:
        """Retourne le nom connu d'un track_id, ou 'INCONNU'."""
        if track_id in self._identified:
            return self._identified[track_id][0]
        return "INCONNU"

    def cleanup_lost_ids(self, active_ids: set):
        """Nettoie les IDs qui ne sont plus suivis."""
        lost_ids = set(self._identified.keys()) - active_ids
        for lost_id in lost_ids:
            del self._identified[lost_id]
            if lost_id in self._last_scan:
                del self._last_scan[lost_id]

    def get_stats(self) -> dict:
        """Retourne les statistiques du matcher."""
        return {
            "whitelist_size": len(self.whitelist),
            "identified_count": len(self._identified),
            "identified_persons": {
                tid: (name, f"{score:.3f}") 
                for tid, (name, score) in self._identified.items()
            }
        }
```

**✅ Critère de validation 2.3** :
```python
python -c "
from src.face_recognition.matcher import FaceMatcher
matcher = FaceMatcher()
matcher.load_whitelist()
stats = matcher.get_stats()
print(f'✅ FaceMatcher initialisé')
print(f'  Whitelist : {stats[\"whitelist_size\"]} personne(s)')
print(f'  Identifiées : {stats[\"identified_count\"]}')
"
```

---

### 2.4 — Intégrer InsightFace dans le Pipeline de Détection

**Actions :**

Modifier le test pour inclure la reconnaissance faciale :

Créer `tests/test_face_recognition.py` :

```python
"""
Test complet de la reconnaissance faciale intégrée au détecteur.
"""
import cv2
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline.detector import PoseDetector
from src.face_recognition.matcher import FaceMatcher
from src.utils.drawing import draw_detections, draw_fps


def test_face_recognition(source=0):
    """Test la reconnaissance faciale en temps réel."""
    
    print("[TEST] Initialisation...")
    detector = PoseDetector()
    matcher = FaceMatcher()
    matcher.load_whitelist()
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir : {source}")
        return False
    
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0
    
    print("[TEST] Démarrage (Appuyez sur 'q' pour quitter)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        # Détection
        detections = detector.detect(frame)
        
        # Reconnaissance faciale
        for det in detections:
            if matcher.should_scan(det.track_id, detector.frame_count):
                # Extraire la zone de la tête (tiers supérieur de la bbox)
                head = det.head_bbox.astype(int)
                x1, y1, x2, y2 = head
                # Sécuriser les bornes
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    face_crop = frame[y1:y2, x1:x2]
                    name, score = matcher.identify(
                        face_crop, det.track_id, detector.frame_count
                    )
                    det.name = name
            else:
                det.name = matcher.get_name(det.track_id)
        
        # Nettoyer les IDs perdus
        active_ids = {d.track_id for d in detections}
        matcher.cleanup_lost_ids(active_ids)
        
        # Affichage
        annotated = draw_detections(frame, detections)
        annotated = draw_fps(annotated, current_fps)
        
        # FPS
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()
        
        cv2.imshow("CCTV AI - Test Face Recognition", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    stats = matcher.get_stats()
    print(f"\n[TEST] Statistiques reconnaissance faciale :")
    print(f"  Personnes identifiées : {stats['identified_persons']}")
    return True


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    success = test_face_recognition(source)
    print(f"\n{'✅ TEST RÉUSSI' if success else '❌ TEST ÉCHOUÉ'}")
```

**✅ Critère de validation 2.4** :
```powershell
python tests/test_face_recognition.py

# DOIT :
# - Afficher les boîtes + squelettes (comme étape 1)
# - Afficher le nom (Thomas/Sarah) ou INCONNU sur chaque personne
# - Le nom doit rester stable après identification (pas de changements aléatoires)
# - Le scan InsightFace ne doit pas tourner à chaque frame
```

---

## ✅ Checklist de Validation Finale — Étape 2

| # | Critère | Commande/Action | Status |
|---|---------|-----------------|--------|
| 2.1 | Photos de référence préparées | Placer photos dans `data/whitelist_photos/` | ⏳ |
| 2.2 | Whitelist encodée et sauvegardée (.npy) | Encoder SCRFD+ArcFace OK | ✅ |
| 2.3 | FaceMatcher fonctionnel | Import + init + stratégie lazy OK | ✅ |
| 2.4 | Reconnaissance en temps réel | Script `test_face_recognition.py` prêt | ✅ |

**Vérifications fonctionnelles obligatoires :**
- [ ] Les personnes connues (whitelist) sont correctement identifiées
- [ ] Les personnes inconnues affichent "INCONNU"
- [ ] Le scan InsightFace est "paresseux" (pas à chaque frame)
- [ ] Une fois identifié avec ≥ 95%, la personne garde son nom sans re-scan
- [ ] Le nom est stable (ne change pas entre les frames)

> [!WARNING]
> Si la reconnaissance est trop lente (< 15 FPS), augmenter `FACE_RECOGNITION_INTERVAL` dans `config.py` (ex: passer de 60 à 90 frames).

---

**⬅️ Étape précédente : [etape_1.md](etape_1.md)**
**➡️ Étape suivante : [etape_3.md](etape_3.md) — Historique et Temps (SQLite)**
