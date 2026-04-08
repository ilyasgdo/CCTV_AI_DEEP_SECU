"""
Module de détection d'objets pour Sentinel-AI.

Détecteur multi-modèle basé sur Ultralytics YOLO.
Supporte la détection classique et la pose estimation.
Le modèle est chargé UNE SEULE FOIS au démarrage.

Usage:
    from src.core.detector import ObjectDetector
    from src.core.config import Config

    config = Config()
    detector = ObjectDetector(config)

    detections = detector.detect(frame)
    persons = detector.detect_persons(frame)
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.core.config import Config
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────

class DetectorError(Exception):
    """Erreur liée au détecteur."""
    pass


class ModelLoadError(DetectorError):
    """Impossible de charger le modèle YOLO."""
    pass


# ──────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────

@dataclass
class Detection:
    """Résultat de détection d'un objet.

    Attributes:
        class_id: ID de classe YOLO.
        class_name: Nom lisible de la classe.
        confidence: Score de confiance [0.0 - 1.0].
        bbox: Bounding box (x1, y1, x2, y2) en pixels.
        center: Centre du bounding box (cx, cy).
        frame_id: ID de la frame source.
        timestamp: Timestamp de la détection.
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class PersonDetection(Detection):
    """Détection de personne avec informations de pose.

    Attributes:
        keypoints: Points clés du squelette (17 points COCO)
                   sous forme de liste de (x, y, confidence).
        pose_confidence: Score de confiance global de la pose.
    """
    keypoints: Optional[list[tuple[float, float, float]]] = None
    pose_confidence: float = 0.0


@dataclass
class DetectorMetrics:
    """Métriques du détecteur."""
    total_detections: int = 0
    total_persons: int = 0
    total_inferences: int = 0
    avg_inference_ms: float = 0.0
    last_inference_ms: float = 0.0
    device: str = "cpu"


# ──────────────────────────────────────────────
# Classes notables pour la sécurité
# ──────────────────────────────────────────────

# Classes YOLO COCO qui intéressent un gardien de sécurité
NOTABLE_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    43: "knife",
    44: "spoon",
    56: "chair",
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "cell phone",
}

PERSON_CLASS_ID = 0


# ──────────────────────────────────────────────
# Détecteur
# ──────────────────────────────────────────────

class ObjectDetector:
    """
    Détecteur d'objets multi-modèle basé sur Ultralytics YOLO.

    Le modèle est chargé une seule fois au démarrage.
    Supporte la détection classique et la pose estimation.
    Auto-détection du device (GPU/CPU).

    Args:
        config: Configuration du projet.
        event_bus: Bus d'événements (optionnel).

    Raises:
        ModelLoadError: Si le modèle YOLO ne peut pas être chargé.
    """

    def __init__(
        self,
        config: Config,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._config = config.detection
        self._event_bus = event_bus
        self._metrics = DetectorMetrics()
        self._frame_counter: int = 0
        self._inference_times: list[float] = []

        # Charger le modèle
        self._model = self._load_model()

        # Flag pose : si le modèle supporte la pose
        self._is_pose_model = "pose" in self._config.model_path

    def _load_model(self) -> object:
        """Charge le modèle YOLO.

        Returns:
            Instance du modèle YOLO.

        Raises:
            ModelLoadError: Si le chargement échoue.
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ModelLoadError(
                "ultralytics n'est pas installé. "
                "Exécutez: pip install ultralytics"
            ) from e

        try:
            model_path = self._config.model_path
            logger.info(f"🔍 Chargement du modèle: {model_path}")

            model = YOLO(model_path)

            # Déterminer le device
            device = self._config.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda:0" if torch.cuda.is_available() \
                        else "cpu"
                except ImportError:
                    device = "cpu"

            self._metrics.device = device
            logger.info(
                f"🔍 Modèle chargé sur: {device} "
                f"(pose={'oui' if 'pose' in model_path else 'non'})"
            )

            return model

        except Exception as e:
            raise ModelLoadError(
                f"Impossible de charger le modèle "
                f"'{self._config.model_path}': {e}"
            ) from e

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
    ) -> list[Detection]:
        """Détecte tous les objets dans une frame.

        Args:
            frame: Image BGR (numpy array).
            frame_id: ID de la frame (pour le tracking).

        Returns:
            Liste de Detection triée par confiance décroissante.
        """
        self._frame_counter += 1

        # Skip frames si configuré
        if self._config.skip_frames > 1:
            if self._frame_counter % self._config.skip_frames != 0:
                return []

        # Inférence
        t_start = time.perf_counter()

        try:
            results = self._model(
                frame,
                imgsz=self._config.input_size,
                conf=self._config.confidence,
                iou=self._config.iou_threshold,
                device=self._metrics.device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Erreur d'inférence YOLO: {e}")
            return []

        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000

        # Métriques
        self._metrics.last_inference_ms = inference_ms
        self._metrics.total_inferences += 1
        self._inference_times.append(inference_ms)
        if len(self._inference_times) > 100:
            self._inference_times = self._inference_times[-100:]
        self._metrics.avg_inference_ms = (
            sum(self._inference_times) / len(self._inference_times)
        )

        # Parser les résultats
        detections: list[Detection] = []
        now = time.time()

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = [
                        int(v) for v in box.xyxy[0].tolist()
                    ]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Nom de la classe
                    class_names = result.names or {}
                    class_name = class_names.get(
                        cls_id, f"class_{cls_id}"
                    )

                    detection = Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        frame_id=frame_id,
                        timestamp=now,
                    )
                    detections.append(detection)

        # Trier par confiance décroissante
        detections.sort(key=lambda d: d.confidence, reverse=True)

        # Statistiques
        person_count = sum(
            1 for d in detections if d.class_id == PERSON_CLASS_ID
        )
        self._metrics.total_detections += len(detections)
        self._metrics.total_persons += person_count

        # Événements
        if self._event_bus:
            if person_count > 0:
                self._event_bus.emit("person_detected", {
                    "count": person_count,
                    "frame_id": frame_id,
                    "detections": [
                        {
                            "bbox": d.bbox,
                            "confidence": d.confidence,
                        }
                        for d in detections
                        if d.class_id == PERSON_CLASS_ID
                    ],
                })

            notable = [
                d for d in detections
                if d.class_id != PERSON_CLASS_ID
                and d.class_id in NOTABLE_CLASSES
            ]
            if notable:
                self._event_bus.emit("object_detected", {
                    "objects": [
                        {
                            "class": d.class_name,
                            "confidence": d.confidence,
                            "bbox": d.bbox,
                        }
                        for d in notable
                    ],
                })

            if person_count == 0 and len(detections) == 0:
                self._event_bus.emit("scene_empty", {
                    "frame_id": frame_id,
                })

        return detections

    def detect_persons(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
    ) -> list[PersonDetection]:
        """Détecte les personnes avec informations de pose.

        Si le modèle est un modèle pose (ex: yolo26n-pose.pt),
        les keypoints sont inclus. Sinon, seuls les bboxes sont
        retournés.

        Args:
            frame: Image BGR (numpy array).
            frame_id: ID de la frame.

        Returns:
            Liste de PersonDetection.
        """
        self._frame_counter += 1

        if self._config.skip_frames > 1:
            if self._frame_counter % self._config.skip_frames != 0:
                return []

        t_start = time.perf_counter()

        try:
            results = self._model(
                frame,
                imgsz=self._config.input_size,
                conf=self._config.confidence,
                iou=self._config.iou_threshold,
                device=self._metrics.device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Erreur d'inférence pose: {e}")
            return []

        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000
        self._metrics.last_inference_ms = inference_ms
        self._metrics.total_inferences += 1

        persons: list[PersonDetection] = []
        now = time.time()

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0].item())

                    # Uniquement les personnes
                    if cls_id != PERSON_CLASS_ID:
                        continue

                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = [
                        int(v) for v in box.xyxy[0].tolist()
                    ]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Keypoints si modèle pose
                    kpts = None
                    pose_conf = 0.0
                    if (self._is_pose_model
                            and result.keypoints is not None
                            and i < len(result.keypoints)):
                        kp_data = result.keypoints[i]
                        if kp_data.xy is not None:
                            xy = kp_data.xy[0].cpu().numpy()
                            if kp_data.conf is not None:
                                confs = (
                                    kp_data.conf[0].cpu().numpy()
                                )
                                kpts = [
                                    (
                                        float(xy[j][0]),
                                        float(xy[j][1]),
                                        float(confs[j]),
                                    )
                                    for j in range(len(xy))
                                ]
                                pose_conf = float(confs.mean())
                            else:
                                kpts = [
                                    (
                                        float(xy[j][0]),
                                        float(xy[j][1]),
                                        1.0,
                                    )
                                    for j in range(len(xy))
                                ]
                                pose_conf = 1.0

                    person = PersonDetection(
                        class_id=PERSON_CLASS_ID,
                        class_name="person",
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        frame_id=frame_id,
                        timestamp=now,
                        keypoints=kpts,
                        pose_confidence=pose_conf,
                    )
                    persons.append(person)

        persons.sort(key=lambda p: p.confidence, reverse=True)
        self._metrics.total_persons += len(persons)

        return persons

    @property
    def metrics(self) -> DetectorMetrics:
        """Retourne les métriques du détecteur."""
        return self._metrics

    @property
    def is_pose_model(self) -> bool:
        """Indique si le modèle supporte la pose estimation."""
        return self._is_pose_model

    def __repr__(self) -> str:
        return (
            f"ObjectDetector("
            f"model={self._config.model_path}, "
            f"device={self._metrics.device}, "
            f"inferences={self._metrics.total_inferences}, "
            f"avg_ms={self._metrics.avg_inference_ms:.1f})"
        )
