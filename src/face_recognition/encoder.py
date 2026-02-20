"""
Module d'encodage des visages.
Convertit les photos de référence en vecteurs d'embedding (fichiers .npy).
Utilise InsightFace (SCRFD + ArcFace) pour la détection et extraction des features.

Compatible avec InsightFace v0.2.x via chargement direct des modèles ONNX.
"""
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from insightface.model_zoo import ArcFaceONNX, SCRFD
from insightface.utils.face_align import norm_crop

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import WHITELIST_DIR, FACE_RECOGNITION_THRESHOLD


class FaceEncoder:
    """
    Encode les visages de référence en vecteurs d'embedding.
    Utilise SCRFD pour la détection et ArcFace pour l'embedding.
    """

    def __init__(self, det_size: tuple = (640, 640), ctx_id: int = -1):
        """
        Initialise les modèles InsightFace.

        Args:
            det_size: Taille de détection (plus grand = plus précis mais plus lent)
            ctx_id: GPU id (0, 1, ...) ou -1 pour CPU
        """
        print("[ENCODER] Initialisation d'InsightFace...")
        model_dir = os.path.expanduser("~/.insightface/models/buffalo_l")

        if not os.path.exists(model_dir):
            raise RuntimeError(
                f"Modèles InsightFace introuvables dans {model_dir}. "
                "Téléchargez buffalo_l depuis: "
                "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
            )

        # Charger le détecteur de visages (SCRFD)
        det_path = os.path.join(model_dir, "det_10g.onnx")
        print(f"[ENCODER] Chargement du détecteur : {os.path.basename(det_path)}")
        self.detector = SCRFD(det_path)
        self.detector.prepare(ctx_id=ctx_id, input_size=det_size)

        # Charger le modèle de reconnaissance (ArcFace)
        rec_path = os.path.join(model_dir, "w600k_r50.onnx")
        print(f"[ENCODER] Chargement du recognizer : {os.path.basename(rec_path)}")
        self.recognizer = ArcFaceONNX(rec_path)
        self.recognizer.prepare(ctx_id=ctx_id)

        print("[ENCODER] InsightFace prêt (SCRFD + ArcFace).")

    def _get_embedding(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """
        Aligne le visage via les keypoints et extrait l'embedding.

        Args:
            img: Image BGR complète
            kps: 5 landmarks du visage (5, 2)

        Returns:
            Vecteur d'embedding normalisé (512,)
        """
        # Aligner le visage (crop 112x112 normalisé)
        aligned = norm_crop(img, kps, image_size=112)
        # Extraire l'embedding
        embedding = self.recognizer.get_feat(aligned)
        embedding = embedding.flatten()
        # Normaliser
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def encode_photo(self, image_path: str) -> Optional[np.ndarray]:
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

        # Détecter les visages (thresh=0.5 pour être permissif sur les photos)
        bboxes, kpss = self.detector.detect(img, threshold=0.5)
        if len(bboxes) == 0:
            print(f"  ⚠ Aucun visage détecté dans : {image_path}")
            return None

        if len(bboxes) > 1:
            print(f"  ⚠ {len(bboxes)} visages détectés, utilisation du plus grand")
            # Prendre le visage le plus grand (surface de la bbox)
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            idx = np.argmax(areas)
            kpss = kpss[idx:idx+1]

        return self._get_embedding(img, kpss[0])

    def detect_and_encode(self, img: np.ndarray, thresh: float = 0.5) -> list:
        """
        Détecte et encode tous les visages dans une image.

        Args:
            img: Image BGR (numpy array)
            thresh: Seuil de détection

        Returns:
            Liste de (bbox, embedding) tuples
        """
        bboxes, kpss = self.detector.detect(img, threshold=thresh)
        results = []
        for i in range(len(bboxes)):
            embedding = self._get_embedding(img, kpss[i])
            results.append((bboxes[i], embedding))
        return results

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

        # Supporter jpg et png
        photo_files = sorted(
            list(photos_dir.glob("*.jpg")) +
            list(photos_dir.glob("*.jpeg")) +
            list(photos_dir.glob("*.png"))
        )

        if not photo_files:
            print("  ⚠ Aucune photo trouvée !")
            return {}

        for photo_path in photo_files:
            # Extraire le nom depuis le fichier (ex: "thomas_1.jpg" → "Thomas")
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
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "whitelist"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, embedding in whitelist.items():
            path = output_dir / f"{name.lower()}.npy"
            np.save(path, embedding)
            print(f"  Sauvegardé : {path}")

        names_path = output_dir / "names.npy"
        np.save(names_path, list(whitelist.keys()))
        print(f"  Liste des noms : {names_path}")

    def load_whitelist(self, whitelist_dir: str = None) -> Dict[str, np.ndarray]:
        """
        Charge la liste blanche depuis les fichiers .npy.
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
