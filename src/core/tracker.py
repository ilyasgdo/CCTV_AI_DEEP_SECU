"""
Module de tracking (suivi) d'entités pour Sentinel-AI.

Tracker basé sur distance euclidienne + IoU pour suivre les
personnes et objets à travers les frames. Assigne un ID unique
stable à chaque entité et gère leur cycle de vie.

Usage:
    from src.core.tracker import Tracker
    from src.core.detector import Detection

    tracker = Tracker(config)
    tracked = tracker.update(detections)
    for entity in tracked:
        print(f"ID={entity.track_id}, status={entity.status}")
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src.core.config import Config
from src.core.detector import Detection
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Constantes et Enums
# ──────────────────────────────────────────────

class EntityStatus(str, Enum):
    """Statut du cycle de vie d'une entité trackée."""
    APPEARED = "APPEARED"
    TRACKED = "TRACKED"
    LOST = "LOST"
    DISAPPEARED = "DISAPPEARED"


# Seuils par défaut
DEFAULT_MAX_LOST_FRAMES = 30
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MAX_DISTANCE = 150  # pixels
DEFAULT_LINGERING_SECONDS = 120


# ──────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────

@dataclass
class TrackedEntity:
    """Entité suivie à travers les frames.

    Attributes:
        track_id: ID unique de l'entité.
        detections: Historique des dernières détections.
        status: Statut actuel du cycle de vie.
        first_seen: Timestamp de la première détection.
        last_seen: Timestamp de la dernière détection.
        total_frames: Nombre total de frames où l'entité a été vue.
        frames_lost: Nombre de frames consécutives sans détection.
        face_id: ID de la personne reconnue (étape 3).
        face_name: Nom de la personne reconnue (étape 3).
        face_confidence: Score de reconnaissance faciale.
        face_status: Statut de reconnaissance.
    """
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    status: EntityStatus = EntityStatus.APPEARED
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    total_frames: int = 0
    frames_lost: int = 0

    # Champs reconnaissance faciale (remplis à l'étape 3)
    face_id: Optional[str] = None
    face_name: Optional[str] = None
    face_confidence: float = 0.0
    face_status: str = "unknown"

    @property
    def last_detection(self) -> Optional[Detection]:
        """Retourne la dernière détection."""
        return self._detections[-1] if self.detections else None

    @property
    def last_bbox(self) -> Optional[tuple[int, int, int, int]]:
        """Retourne le dernier bounding box."""
        if self.detections:
            return self.detections[-1].bbox
        return None

    @property
    def last_center(self) -> Optional[tuple[int, int]]:
        """Retourne le dernier centre."""
        if self.detections:
            return self.detections[-1].center
        return None

    @property
    def duration_seconds(self) -> float:
        """Durée de présence en secondes."""
        return self.last_seen - self.first_seen

    @property
    def is_person(self) -> bool:
        """Indique si l'entité est une personne."""
        if self.detections:
            return self.detections[-1].class_id == 0
        return False

    def add_detection(self, detection: Detection) -> None:
        """Ajoute une nouvelle détection et met à jour les stats.

        Args:
            detection: Nouvelle détection à associer.
        """
        self.detections.append(detection)
        # Garder un historique limité (30 dernières)
        if len(self.detections) > 30:
            self.detections = self.detections[-30:]

        self.last_seen = detection.timestamp
        self.total_frames += 1
        self.frames_lost = 0

        if self.status == EntityStatus.APPEARED:
            self.status = EntityStatus.TRACKED
        elif self.status == EntityStatus.LOST:
            self.status = EntityStatus.TRACKED


# ──────────────────────────────────────────────
# Fonctions utilitaires
# ──────────────────────────────────────────────

def compute_iou(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
) -> float:
    """Calcule l'Intersection over Union entre deux bounding boxes.

    Args:
        box1: (x1, y1, x2, y2) du premier box.
        box2: (x1, y1, x2, y2) du second box.

    Returns:
        Score IoU entre 0.0 et 1.0.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union == 0:
        return 0.0

    return intersection / union


def euclidean_distance(
    p1: tuple[int, int],
    p2: tuple[int, int],
) -> float:
    """Calcule la distance euclidienne entre deux points.

    Args:
        p1: Premier point (x, y).
        p2: Second point (x, y).

    Returns:
        Distance en pixels.
    """
    return float(np.sqrt(
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    ))


# ──────────────────────────────────────────────
# Tracker
# ──────────────────────────────────────────────

class Tracker:
    """
    Tracker multi-objets basé sur distance euclidienne + IoU.

    Assigne un ID unique à chaque entité détectée et maintient
    son suivi à travers les frames. Gère le cycle de vie complet :
    APPEARED → TRACKED → LOST → DISAPPEARED.

    Args:
        config: Configuration du projet.
        event_bus: Bus d'événements (optionnel).
        max_lost_frames: Frames sans détection avant DISAPPEARED.
        max_distance: Distance max pour l'association (pixels).
        iou_threshold: Seuil IoU minimum pour le matching.

    Example:
        >>> tracker = Tracker(config)
        >>> entities = tracker.update(detections)
        >>> for e in entities:
        ...     print(f"ID={e.track_id} status={e.status}")
    """

    def __init__(
        self,
        config: Config,
        event_bus: Optional[EventBus] = None,
        max_lost_frames: int = DEFAULT_MAX_LOST_FRAMES,
        max_distance: float = DEFAULT_MAX_DISTANCE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._max_lost_frames = max_lost_frames
        self._max_distance = max_distance
        self._iou_threshold = iou_threshold

        # Entités trackées
        self._entities: dict[int, TrackedEntity] = {}
        self._next_id: int = 1

        # Statistiques
        self._total_appeared: int = 0
        self._total_disappeared: int = 0

        # Seuil de rôdeur (configurable via config.alerts)
        self._lingering_threshold = getattr(
            config.alerts, 'lingering_threshold',
            DEFAULT_LINGERING_SECONDS
        )

    def update(
        self, detections: list[Detection]
    ) -> list[TrackedEntity]:
        """Met à jour le tracker avec de nouvelles détections.

        Algorithme :
        1. Calculer la matrice de coût (distance + IoU) entre
           les entités existantes et les nouvelles détections.
        2. Effectuer l'association greedy (meilleur score d'abord).
        3. Créer de nouvelles entités pour les détections
           non-matchées.
        4. Marquer les entités non-matchées comme LOST.
        5. Supprimer les entités LOST depuis trop longtemps.

        Args:
            detections: Liste des détections de la frame courante.

        Returns:
            Liste des entités activement trackées (pas DISAPPEARED).
        """
        now = time.time()

        # ── Étape 1 : Matrice de coût ──
        active_entities = {
            tid: entity
            for tid, entity in self._entities.items()
            if entity.status != EntityStatus.DISAPPEARED
        }

        if not active_entities and not detections:
            return []

        matched_entities: set[int] = set()
        matched_detections: set[int] = set()

        # ── Étape 2 : Association greedy ──
        if active_entities and detections:
            # Construire les paires (score, entity_id, det_idx)
            candidates: list[tuple[float, int, int]] = []

            for tid, entity in active_entities.items():
                if entity.last_bbox is None:
                    continue
                for di, det in enumerate(detections):
                    # Même classe
                    if (entity.detections
                            and entity.detections[-1].class_id
                            != det.class_id):
                        continue

                    # Distance euclidienne
                    dist = euclidean_distance(
                        entity.last_center, det.center
                    )
                    if dist > self._max_distance:
                        continue

                    # IoU
                    iou = compute_iou(entity.last_bbox, det.bbox)

                    # Score combiné (plus haut = meilleur match)
                    # Normaliser la distance [0, max_distance] -> [0, 1]
                    dist_score = 1.0 - (dist / self._max_distance)
                    score = 0.5 * iou + 0.5 * dist_score

                    candidates.append((score, tid, di))

            # Trier par score décroissant (meilleur match d'abord)
            candidates.sort(key=lambda x: x[0], reverse=True)

            for score, tid, di in candidates:
                if tid in matched_entities or di in matched_detections:
                    continue

                # Match trouvé
                entity = self._entities[tid]
                entity.add_detection(detections[di])
                matched_entities.add(tid)
                matched_detections.add(di)

        # ── Étape 3 : Nouvelles entités ──
        for di, det in enumerate(detections):
            if di not in matched_detections:
                new_entity = TrackedEntity(
                    track_id=self._next_id,
                    detections=[det],
                    status=EntityStatus.APPEARED,
                    first_seen=det.timestamp,
                    last_seen=det.timestamp,
                    total_frames=1,
                )
                self._entities[self._next_id] = new_entity
                self._next_id += 1
                self._total_appeared += 1

                logger.debug(
                    f"🆕 Entité #{new_entity.track_id} apparue "
                    f"({det.class_name} @ {det.center})"
                )

                if self._event_bus:
                    self._event_bus.emit("entity_appeared", {
                        "track_id": new_entity.track_id,
                        "class": det.class_name,
                        "bbox": det.bbox,
                        "confidence": det.confidence,
                    })

        # ── Étape 4 : Entités perdues ──
        for tid, entity in active_entities.items():
            if tid not in matched_entities:
                entity.frames_lost += 1

                if entity.status != EntityStatus.LOST:
                    entity.status = EntityStatus.LOST

        # ── Étape 5 : Suppression des entités disparues ──
        to_remove: list[int] = []
        for tid, entity in self._entities.items():
            if entity.status == EntityStatus.DISAPPEARED:
                continue

            if entity.frames_lost >= self._max_lost_frames:
                entity.status = EntityStatus.DISAPPEARED
                self._total_disappeared += 1

                logger.debug(
                    f"👋 Entité #{entity.track_id} disparue "
                    f"(présente {entity.duration_seconds:.1f}s)"
                )

                if self._event_bus:
                    self._event_bus.emit("entity_disappeared", {
                        "track_id": entity.track_id,
                        "duration": entity.duration_seconds,
                        "total_frames": entity.total_frames,
                    })

                to_remove.append(tid)

        # Nettoyer les entités disparues depuis longtemps
        for tid in to_remove:
            if self._entities[tid].total_frames < 3:
                # Très éphémère → supprimer complètement
                del self._entities[tid]

        # ── Étape 6 : Détection de rôdeurs ──
        for tid, entity in self._entities.items():
            if (entity.status == EntityStatus.TRACKED
                    and entity.is_person
                    and entity.duration_seconds
                    > self._lingering_threshold):
                if self._event_bus:
                    self._event_bus.emit("entity_lingering", {
                        "track_id": entity.track_id,
                        "duration": entity.duration_seconds,
                    })

        # Retourner les entités actives
        return self.active_entities

    @property
    def active_entities(self) -> list[TrackedEntity]:
        """Retourne les entités activement trackées.

        Returns:
            Liste des entités (APPEARED, TRACKED, ou LOST).
        """
        return [
            entity for entity in self._entities.values()
            if entity.status != EntityStatus.DISAPPEARED
        ]

    @property
    def active_count(self) -> int:
        """Nombre d'entités activement trackées."""
        return len(self.active_entities)

    @property
    def person_count(self) -> int:
        """Nombre de personnes actuellement trackées."""
        return sum(
            1 for e in self.active_entities if e.is_person
        )

    def get_entity(self, track_id: int) -> Optional[TrackedEntity]:
        """Retourne une entité par son ID.

        Args:
            track_id: ID de l'entité.

        Returns:
            TrackedEntity ou None.
        """
        return self._entities.get(track_id)

    @property
    def total_appeared(self) -> int:
        """Nombre total d'entités apparues."""
        return self._total_appeared

    @property
    def total_disappeared(self) -> int:
        """Nombre total d'entités disparues."""
        return self._total_disappeared

    def reset(self) -> None:
        """Réinitialise le tracker."""
        self._entities.clear()
        self._next_id = 1
        self._total_appeared = 0
        self._total_disappeared = 0

    def __repr__(self) -> str:
        return (
            f"Tracker(active={self.active_count}, "
            f"persons={self.person_count}, "
            f"total_appeared={self._total_appeared}, "
            f"total_disappeared={self._total_disappeared})"
        )
