"""
Module de reconnaissance faciale et gestion de whitelist pour Sentinel-AI.

Ce module implémente l'etape 3:
- extraction d'embeddings via InsightFace
- comparaison whitelist (similarite cosinus)
- enrollement par photo/camera
- cache de reconnaissance lie au track_id
- snapshots et evenements pour visages inconnus
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import cv2
import numpy as np

from src.core.config import Config
from src.core.tracker import TrackedEntity
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from scipy.spatial.distance import cosine as scipy_cosine_distance
except ImportError:
    scipy_cosine_distance = None

try:
    from cryptography.fernet import Fernet  # type: ignore[import-not-found]
except ImportError:
    Fernet = None  # type: ignore[assignment]


class FaceManagerError(Exception):
    """Erreur de base du module de reconnaissance faciale."""


class EnrollmentError(FaceManagerError):
    """Erreur pendant l'enrollement d'une personne."""


class RecognitionError(FaceManagerError):
    """Erreur pendant la reconnaissance faciale."""


class EmbeddingBackend(Protocol):
    """Interface d'un backend d'extraction d'embeddings faciaux."""

    def extract_embeddings(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        """Extrait zero, un ou plusieurs embeddings faciaux d'une image."""


class InsightFaceEmbedder:
    """Backend InsightFace (ONNX Runtime) pour embeddings 512D."""

    def __init__(self, providers: Optional[list[str]] = None) -> None:
        self._providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._app: Any = None

    def _ensure_initialized(self) -> None:
        """Initialise FaceAnalysis au premier usage."""
        if self._app is not None:
            return

        try:
            from insightface.app import FaceAnalysis  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RecognitionError(
                "insightface n'est pas installe. Executez: pip install insightface"
            ) from exc

        try:
            self._app = FaceAnalysis(name="buffalo_l", providers=self._providers)
        except TypeError:
            # Compatibilite avec certaines versions d'insightface
            # qui ne supportent pas l'argument `providers`.
            # Certaines versions anciennes echouent sur le pack complet
            # buffalo_l (routing de modeles non supportes). On utilise un
            # pack minimal detection+recognition.
            root = Path.home() / ".insightface" / "models"
            src_dir = root / "buffalo_l"
            dst_name = "buffalo_l_min"
            dst_dir = root / dst_name
            det_src = src_dir / "det_10g.onnx"
            rec_src = src_dir / "w600k_r50.onnx"

            try:
                dst_dir.mkdir(parents=True, exist_ok=True)
                if det_src.exists() and not (dst_dir / det_src.name).exists():
                    shutil.copy2(det_src, dst_dir / det_src.name)
                if rec_src.exists() and not (dst_dir / rec_src.name).exists():
                    shutil.copy2(rec_src, dst_dir / rec_src.name)
                self._app = FaceAnalysis(name=dst_name)
            except Exception:
                self._app = FaceAnalysis(name="buffalo_l")

        ctx_id = 0 if "CUDAExecutionProvider" in self._providers else -1
        # det_size modeste pour un bon compromis qualite/perf.
        try:
            self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        except TypeError:
            self._app.prepare(ctx_id=ctx_id)

    def extract_embeddings(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        """Extrait les embeddings des visages detectes dans une image BGR.

        Args:
            image_bgr: Image source BGR.

        Returns:
            Liste d'embeddings float32 normalises.
        """
        self._ensure_initialized()
        faces = self._app.get(image_bgr)

        embeddings: list[np.ndarray] = []
        for face in faces:
            emb = np.asarray(face.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0.0:
                emb = emb / norm
                embeddings.append(emb)

        return embeddings


@dataclass
class FaceMatchResult:
    """Resultat de comparaison d'un embedding avec la whitelist."""

    status: str
    confidence: float
    person_id: Optional[str] = None
    name: Optional[str] = None
    role: Optional[str] = None


class WhitelistRepository:
    """Gestion CRUD de la whitelist locale (registry + fichiers)."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.registry_path = self.base_dir / "registry.json"
        self.embeddings_dir = self.base_dir / "embeddings"
        self.photos_dir = self.base_dir / "photos"
        self._encryption_key = os.getenv("SENTINEL_WHITELIST_ENCRYPTION_KEY", "").strip()

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.photos_dir.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            self._save_registry({"persons": []})

    def _encrypt_bytes_if_enabled(self, payload: bytes) -> bytes:
        """Chiffre le payload si une cle valide est configuree."""
        if not self._encryption_key:
            return payload
        if Fernet is None:
            logger.warning("Chiffrement whitelist desactive: cryptography indisponible")
            return payload

        try:
            return Fernet(self._encryption_key.encode("utf-8")).encrypt(payload)
        except Exception:
            logger.error("Cle de chiffrement whitelist invalide, ecriture en clair")
            return payload

    def _decrypt_bytes_if_enabled(self, payload: bytes) -> bytes:
        """Dechiffre le payload si une cle valide est configuree."""
        if not self._encryption_key:
            return payload
        if Fernet is None:
            return payload

        try:
            return Fernet(self._encryption_key.encode("utf-8")).decrypt(payload)
        except Exception:
            logger.warning("Dechiffrement whitelist impossible, tentative lecture brute")
            return payload

    def _load_registry(self) -> dict[str, Any]:
        """Charge le registre JSON depuis disque."""
        with self.registry_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "persons" not in data or not isinstance(data["persons"], list):
            data = {"persons": []}

        return data

    def _save_registry(self, registry: dict[str, Any]) -> None:
        """Sauvegarde le registre JSON de whitelist."""
        with self.registry_path.open("w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    def list_persons(self) -> list[dict[str, Any]]:
        """Retourne toutes les personnes du registre."""
        return self._load_registry()["persons"]

    def get_person(self, person_id: str) -> Optional[dict[str, Any]]:
        """Retourne une personne par son identifiant."""
        for person in self.list_persons():
            if person.get("id") == person_id:
                return person
        return None

    def add_person(self, person: dict[str, Any]) -> None:
        """Ajoute une personne au registre.

        Args:
            person: Enregistrement personne compatible avec registry.json.

        Raises:
            EnrollmentError: Si l'identifiant existe deja.
        """
        registry = self._load_registry()
        person_id = person.get("id")
        if not person_id:
            raise EnrollmentError("person.id est obligatoire")

        if any(p.get("id") == person_id for p in registry["persons"]):
            raise EnrollmentError(f"Personne deja presente: {person_id}")

        registry["persons"].append(person)
        self._save_registry(registry)

    def update_person(self, person_id: str, updates: dict[str, Any]) -> bool:
        """Met a jour une personne existante.

        Args:
            person_id: Identifiant de la personne.
            updates: Champs a mettre a jour.

        Returns:
            True si la personne est trouvee.
        """
        registry = self._load_registry()
        for person in registry["persons"]:
            if person.get("id") == person_id:
                person.update(updates)
                self._save_registry(registry)
                return True
        return False

    def remove_person(self, person_id: str) -> bool:
        """Supprime une personne et ses fichiers associes.

        Args:
            person_id: Identifiant de la personne.

        Returns:
            True si suppression effectuee.
        """
        registry = self._load_registry()
        target: Optional[dict[str, Any]] = None
        kept: list[dict[str, Any]] = []

        for person in registry["persons"]:
            if person.get("id") == person_id:
                target = person
            else:
                kept.append(person)

        if target is None:
            return False

        for emb_name in target.get("embeddings", []):
            emb_path = self.embeddings_dir / emb_name
            if emb_path.exists():
                emb_path.unlink()

        for photo_name in target.get("photos", []):
            photo_path = self.photos_dir / photo_name
            if photo_path.exists():
                photo_path.unlink()

        registry["persons"] = kept
        self._save_registry(registry)
        return True

    def save_embedding(self, filename: str, embedding: np.ndarray) -> str:
        """Sauvegarde un embedding numpy dans le repertoire dedie."""
        path = self.embeddings_dir / filename
        buffer = io.BytesIO()
        np.save(buffer, embedding.astype(np.float32))
        payload = self._encrypt_bytes_if_enabled(buffer.getvalue())
        path.write_bytes(payload)
        return filename

    def save_photo(self, filename: str, image_bgr: np.ndarray) -> str:
        """Sauvegarde une photo de reference dans le repertoire dedie."""
        path = self.photos_dir / filename
        cv2.imwrite(str(path), image_bgr)
        return filename

    def load_embeddings_index(self) -> list[tuple[str, str, np.ndarray]]:
        """Charge tous les embeddings du registre en memoire.

        Returns:
            Liste de tuples (person_id, name, embedding).
        """
        index: list[tuple[str, str, np.ndarray]] = []

        for person in self.list_persons():
            person_id = str(person.get("id", ""))
            name = str(person.get("name", person_id))
            for emb_file in person.get("embeddings", []):
                emb_path = self.embeddings_dir / emb_file
                if not emb_path.exists():
                    continue

                raw_payload = emb_path.read_bytes()
                decrypted = self._decrypt_bytes_if_enabled(raw_payload)
                embedding = np.load(io.BytesIO(decrypted), allow_pickle=False).astype(np.float32)
                norm = float(np.linalg.norm(embedding))
                if norm > 0.0:
                    embedding = embedding / norm
                index.append((person_id, name, embedding))

        return index


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcule une similarite cosinus dans [0, 1]."""
    if scipy_cosine_distance is not None:
        distance = float(scipy_cosine_distance(v1, v2))
        return float(max(0.0, min(1.0, 1.0 - distance)))

    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 0.0:
        return 0.0

    score = float(np.dot(v1, v2) / denom)
    return float(max(0.0, min(1.0, score)))


class FaceManager:
    """Gestion complete de la reconnaissance faciale Sentinel-AI."""

    def __init__(
        self,
        config: Config,
        event_bus: Optional[EventBus] = None,
        embedder: Optional[EmbeddingBackend] = None,
        repository: Optional[WhitelistRepository] = None,
        unknown_snapshots_dir: Optional[str | Path] = None,
    ) -> None:
        self._config = config.face
        self._event_bus = event_bus

        self._repository = repository or WhitelistRepository(self._config.whitelist_dir)
        self._embedder = embedder or InsightFaceEmbedder()

        self._similarity_threshold = float(self._config.similarity_threshold)
        self._uncertain_threshold = float(self._config.uncertain_threshold)
        self._recalculate_interval = int(self._config.recalculate_interval)
        self._margin_percent = float(self._config.margin_percent)

        project_root = config.project_root
        self._unknown_dir = (
            Path(unknown_snapshots_dir)
            if unknown_snapshots_dir is not None
            else project_root / "data" / "snapshots" / "unknowns"
        )
        self._unknown_dir.mkdir(parents=True, exist_ok=True)

        self._known_embeddings: list[tuple[str, str, np.ndarray]] = []
        self._person_roles: dict[str, str] = {}
        self._track_cache: dict[int, dict[str, Any]] = {}
        self._quality_face_detector: Any = None

        self.reload_whitelist()

    def reload_whitelist(self) -> None:
        """Recharge la whitelist (registre + embeddings) en memoire."""
        self._known_embeddings = self._repository.load_embeddings_index()
        self._person_roles.clear()
        for person in self._repository.list_persons():
            self._person_roles[str(person.get("id"))] = str(person.get("role", ""))

    @staticmethod
    def _now_iso() -> str:
        """Retourne un timestamp UTC ISO8601."""
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    @staticmethod
    def _slugify(value: str) -> str:
        """Construit un identifiant de fichier simple ASCII."""
        sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        return sanitized.strip("_") or "person"

    def _get_quality_face_detector(self) -> Optional[Any]:
        """Initialise paresseusement le detecteur de visage OpenCV pour QA."""
        if self._quality_face_detector is not None:
            return self._quality_face_detector

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return None

        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            return None

        self._quality_face_detector = detector
        return detector

    def _select_quality_roi(self, image_bgr: np.ndarray) -> np.ndarray:
        """Retourne la zone la plus pertinente pour evaluer la qualite.

        Sur de grandes images (ex: frame camera 1280x720), la qualite doit etre
        evaluee sur le visage et non sur toute la scene.
        """
        h, w = image_bgr.shape[:2]
        if h < 220 and w < 220:
            return image_bgr

        detector = self._get_quality_face_detector()
        if detector is None:
            return image_bgr

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )
        if len(faces) == 0:
            return image_bgr

        x, y, fw, fh = max(faces, key=lambda f: int(f[2] * f[3]))
        x2 = min(w, x + fw)
        y2 = min(h, y + fh)
        roi = image_bgr[y:y2, x:x2]
        if roi.size == 0:
            return image_bgr

        return roi

    def validate_face_quality(self, face_bgr: np.ndarray) -> bool:
        """Valide la qualite minimale d'un crop visage.

        Critere utilises:
        - taille min 40x40
        - luminosite moyenne entre 35 et 225
        - nettete min via variance du Laplacien

        Args:
            face_bgr: Crop visage BGR.

        Returns:
            True si la qualite est suffisante.
        """
        if face_bgr is None or face_bgr.size == 0:
            return False

        quality_roi = self._select_quality_roi(face_bgr)

        h, w = quality_roi.shape[:2]
        if h < 40 or w < 40:
            return False

        gray = cv2.cvtColor(quality_roi, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if brightness < 25.0 or brightness > 235.0:
            return False

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < 20.0:
            return False

        return True

    def extract_face_crop(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        margin_percent: Optional[float] = None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Extrait un crop visage avec marge autour du bbox personne.

        Args:
            frame_bgr: Frame source BGR.
            bbox: Bounding box (x1, y1, x2, y2).
            margin_percent: Marge optionnelle autour du bbox.

        Returns:
            Tuple (crop_bgr, bbox_corrige).
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        margin = margin_percent if margin_percent is not None else self._margin_percent

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        mx = int(bw * margin)
        my = int(bh * margin)

        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        return crop, (cx1, cy1, cx2, cy2)

    def _extract_single_embedding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extrait un embedding unique depuis une image BGR."""
        embeddings = self._embedder.extract_embeddings(image_bgr)
        if not embeddings:
            return None
        return embeddings[0]

    def _match_embedding(self, embedding: np.ndarray) -> FaceMatchResult:
        """Compare un embedding a toute la whitelist."""
        if not self._known_embeddings:
            return FaceMatchResult(status="unknown", confidence=0.0)

        best_score = 0.0
        best_person_id: Optional[str] = None
        best_name: Optional[str] = None

        for person_id, name, known_emb in self._known_embeddings:
            score = cosine_similarity(embedding, known_emb)
            if score > best_score:
                best_score = score
                best_person_id = person_id
                best_name = name

        if best_score >= self._similarity_threshold and best_person_id:
            return FaceMatchResult(
                status="known",
                confidence=best_score,
                person_id=best_person_id,
                name=best_name,
                role=self._person_roles.get(best_person_id),
            )

        if best_score >= self._uncertain_threshold:
            return FaceMatchResult(
                status="uncertain",
                confidence=best_score,
                person_id=best_person_id,
                name=best_name,
                role=self._person_roles.get(best_person_id) if best_person_id else None,
            )

        return FaceMatchResult(status="unknown", confidence=best_score)

    def _bbox_changed_significantly(
        self,
        prev_bbox: tuple[int, int, int, int],
        curr_bbox: tuple[int, int, int, int],
    ) -> bool:
        """Detecte un changement de cadrage significatif."""
        px1, py1, px2, py2 = prev_bbox
        cx1, cy1, cx2, cy2 = curr_bbox

        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        ccx = (cx1 + cx2) / 2.0
        ccy = (cy1 + cy2) / 2.0

        center_shift = float(np.sqrt((pcx - ccx) ** 2 + (pcy - ccy) ** 2))
        prev_diag = float(np.sqrt(max(1, (px2 - px1)) ** 2 + max(1, (py2 - py1)) ** 2))
        motion_ratio = center_shift / max(1.0, prev_diag)

        prev_area = max(1.0, float((px2 - px1) * (py2 - py1)))
        curr_area = max(1.0, float((cx2 - cx1) * (cy2 - cy1)))
        scale_ratio = abs(curr_area - prev_area) / prev_area

        return motion_ratio > 0.20 or scale_ratio > 0.25

    def _should_recalculate(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
        frame_id: int,
    ) -> bool:
        """Determine si l'embedding doit etre recalcule."""
        cached = self._track_cache.get(track_id)
        if cached is None:
            return True

        last_frame = int(cached.get("frame_id", -10_000))
        if frame_id - last_frame >= self._recalculate_interval:
            return True

        last_bbox = cached.get("bbox")
        if last_bbox is None:
            return True

        return self._bbox_changed_significantly(last_bbox, bbox)

    def _save_unknown_snapshot(
        self,
        face_crop: np.ndarray,
        track_id: int,
        frame_id: int,
    ) -> Path:
        """Sauvegarde un snapshot d'inconnu sur disque."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"unknown_t{track_id}_f{frame_id}_{stamp}.jpg"
        path = self._unknown_dir / name
        cv2.imwrite(str(path), face_crop)
        return path

    def recognize_entity(
        self,
        frame_bgr: np.ndarray,
        entity: TrackedEntity,
        frame_id: int,
    ) -> TrackedEntity:
        """Reconnaissance faciale pour une entite trackee.

        Args:
            frame_bgr: Frame source BGR.
            entity: Entite trackee a enrichir.
            frame_id: Index/frame_id courant.

        Returns:
            Entite enrichie (modifiee in-place et retournee).
        """
        if not entity.is_person or entity.last_bbox is None:
            return entity

        if not self._should_recalculate(entity.track_id, entity.last_bbox, frame_id):
            return entity

        face_crop, face_bbox = self.extract_face_crop(frame_bgr, entity.last_bbox)
        if not self.validate_face_quality(face_crop):
            entity.face_status = "unknown"
            entity.face_confidence = 0.0
            return entity

        embedding = self._extract_single_embedding(face_crop)
        if embedding is None:
            entity.face_status = "unknown"
            entity.face_confidence = 0.0
            return entity

        match = self._match_embedding(embedding)
        entity.face_status = match.status
        entity.face_confidence = float(match.confidence)

        if match.status == "known":
            entity.face_id = match.person_id
            entity.face_name = match.name
            if match.person_id:
                self._repository.update_person(match.person_id, {"last_seen": self._now_iso()})

            if self._event_bus:
                self._event_bus.emit("face_recognized", {
                    "person_id": match.person_id,
                    "name": match.name,
                    "confidence": match.confidence,
                    "bbox": face_bbox,
                    "track_id": entity.track_id,
                })

        elif match.status == "uncertain":
            entity.face_id = None
            entity.face_name = match.name

            if self._event_bus:
                self._event_bus.emit("face_uncertain", {
                    "bbox": face_bbox,
                    "best_match": match.name,
                    "confidence": match.confidence,
                    "track_id": entity.track_id,
                })

        else:
            entity.face_id = None
            entity.face_name = None

            snapshot_path = self._save_unknown_snapshot(face_crop, entity.track_id, frame_id)
            logger.info(
                "Visage inconnu track_id=%s confidence=%.3f snapshot=%s",
                entity.track_id,
                match.confidence,
                snapshot_path,
            )

            if self._event_bus:
                self._event_bus.emit("face_unknown", {
                    "bbox": face_bbox,
                    "snapshot_path": str(snapshot_path),
                    "track_id": entity.track_id,
                })

        self._track_cache[entity.track_id] = {
            "frame_id": frame_id,
            "bbox": entity.last_bbox,
            "status": entity.face_status,
            "confidence": entity.face_confidence,
        }

        return entity

    def recognize_entities(
        self,
        frame_bgr: np.ndarray,
        entities: list[TrackedEntity],
        frame_id: int,
    ) -> list[TrackedEntity]:
        """Reconnaissance faciale pour toutes les entites d'une frame."""
        for entity in entities:
            self.recognize_entity(frame_bgr, entity, frame_id)
        return entities

    def enroll_from_photos(
        self,
        name: str,
        role: str,
        access_level: str,
        photo_paths: list[str],
        notes: str = "",
    ) -> dict[str, Any]:
        """Enrole une personne a partir de photos de reference.

        Args:
            name: Nom complet.
            role: Role de securite.
            access_level: Niveau d'acces.
            photo_paths: Liste de 3 a 5 photos.
            notes: Notes libres.

        Returns:
            Enregistrement personne ajoute au registre.

        Raises:
            EnrollmentError: Si l'enrollement echoue.
        """
        if len(photo_paths) < 3 or len(photo_paths) > 5:
            raise EnrollmentError("L'enrollement photo exige entre 3 et 5 images.")

        slug = self._slugify(name)
        person_id = f"person_{int(time.time() * 1000)}"

        embedding_files: list[str] = []
        photo_files: list[str] = []
        rejected_reasons: list[str] = []

        for idx, photo_path_str in enumerate(photo_paths, start=1):
            photo_path = Path(photo_path_str)
            image = cv2.imread(str(photo_path))
            if image is None:
                rejected_reasons.append(f"{photo_path} (image illisible)")
                continue

            embedding = self._extract_single_embedding(image)
            if embedding is None:
                rejected_reasons.append(f"{photo_path} (aucun visage detecte)")
                continue

            if not self.validate_face_quality(image):
                rejected_reasons.append(f"{photo_path} (qualite insuffisante)")
                continue

            emb_name = f"{slug}_{idx:03d}.npy"
            photo_name = f"{slug}_ref_{idx}.jpg"

            self._repository.save_embedding(emb_name, embedding)
            self._repository.save_photo(photo_name, image)
            embedding_files.append(emb_name)
            photo_files.append(photo_name)

        if len(embedding_files) < 3:
            rejected = "; ".join(rejected_reasons) if rejected_reasons else "raison inconnue"
            raise EnrollmentError(
                "Enrollement photo impossible: "
                f"{len(embedding_files)}/{len(photo_paths)} captures valides "
                f"(minimum requis: 3). Rejets: {rejected}"
            )

        person = {
            "id": person_id,
            "name": name,
            "role": role,
            "access_level": access_level,
            "embeddings": embedding_files,
            "photos": photo_files,
            "enrolled_at": self._now_iso(),
            "last_seen": None,
            "notes": notes,
        }

        self._repository.add_person(person)
        self.reload_whitelist()

        logger.info("Personne enrolee: %s (%s) embeddings=%d", name, person_id, len(embedding_files))
        return person

    def enroll_from_camera(
        self,
        camera: Any,
        name: str,
        role: str,
        access_level: str,
        notes: str = "",
        samples: int = 5,
        interval_seconds: float = 0.7,
        max_attempts: int = 200,
    ) -> dict[str, Any]:
        """Enrole une personne via capture live camera.

        Args:
            camera: Instance camera avec methode get_frame().
            name: Nom complet.
            role: Role de securite.
            access_level: Niveau d'acces.
            notes: Notes libres.
            samples: Nombre d'echantillons a capturer.
            interval_seconds: Delai entre captures utiles.
            max_attempts: Nombre max de frames inspectees.

        Returns:
            Enregistrement personne ajoute au registre.
        """
        if samples < 1:
            raise EnrollmentError("samples doit etre >= 1")

        slug = self._slugify(name)
        person_id = f"person_{int(time.time() * 1000)}"

        embedding_files: list[str] = []
        photo_files: list[str] = []

        captured = 0
        attempts = 0
        last_capture_ts = 0.0

        while captured < samples and attempts < max_attempts:
            attempts += 1
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            now = time.time()
            if now - last_capture_ts < interval_seconds:
                continue

            embedding = self._extract_single_embedding(frame)
            if embedding is None:
                continue

            if not self.validate_face_quality(frame):
                continue

            captured += 1
            last_capture_ts = now

            emb_name = f"{slug}_{captured:03d}.npy"
            photo_name = f"{slug}_ref_{captured}.jpg"
            self._repository.save_embedding(emb_name, embedding)
            self._repository.save_photo(photo_name, frame)

            embedding_files.append(emb_name)
            photo_files.append(photo_name)

        if captured < samples:
            raise EnrollmentError(
                f"Enrollement incomplet: {captured}/{samples} captures valides."
            )

        person = {
            "id": person_id,
            "name": name,
            "role": role,
            "access_level": access_level,
            "embeddings": embedding_files,
            "photos": photo_files,
            "enrolled_at": self._now_iso(),
            "last_seen": None,
            "notes": notes,
        }

        self._repository.add_person(person)
        self.reload_whitelist()

        logger.info("Personne enrolee par camera: %s (%s)", name, person_id)
        return person

    @property
    def repository(self) -> WhitelistRepository:
        """Expose le repository whitelist."""
        return self._repository

    @property
    def known_embeddings_count(self) -> int:
        """Nombre d'embeddings actuellement charges en memoire."""
        return len(self._known_embeddings)
