"""
Tests unitaires pour src/core/tracker.py

Teste le tracking : assignation d'IDs, cycle de vie,
IoU, distance euclidienne, événements.
"""

import time

import numpy as np
import pytest

from src.core.config import Config
from src.core.detector import Detection
from src.core.tracker import (
    Tracker,
    TrackedEntity,
    EntityStatus,
    compute_iou,
    euclidean_distance,
)
from src.utils.event_bus import EventBus
from src.utils.logger import reset_logging


@pytest.fixture(autouse=True)
def clean() -> None:
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def tracker(config: Config) -> Tracker:
    return Tracker(config, max_lost_frames=5)


def make_detection(
    x1: int, y1: int, x2: int, y2: int,
    class_id: int = 0,
    confidence: float = 0.9,
    frame_id: int = 0,
) -> Detection:
    """Helper pour créer une detection."""
    return Detection(
        class_id=class_id,
        class_name="person" if class_id == 0 else "object",
        confidence=confidence,
        bbox=(x1, y1, x2, y2),
        center=((x1 + x2) // 2, (y1 + y2) // 2),
        frame_id=frame_id,
        timestamp=time.time(),
    )


# ──────────────────────────────────────────────
# Tests — Fonctions utilitaires
# ──────────────────────────────────────────────

class TestUtilities:
    """Tests des fonctions utilitaires IoU et distance."""

    def test_iou_perfect_overlap(self) -> None:
        """IoU = 1.0 pour des boxes identiques."""
        iou = compute_iou(
            (100, 100, 200, 200), (100, 100, 200, 200)
        )
        assert iou == pytest.approx(1.0)

    def test_iou_no_overlap(self) -> None:
        """IoU = 0.0 pour des boxes sans intersection."""
        iou = compute_iou(
            (0, 0, 100, 100), (200, 200, 300, 300)
        )
        assert iou == 0.0

    def test_iou_partial_overlap(self) -> None:
        """IoU entre 0 et 1 pour un overlap partiel."""
        iou = compute_iou(
            (0, 0, 100, 100), (50, 50, 150, 150)
        )
        assert 0.0 < iou < 1.0

    def test_iou_contained(self) -> None:
        """IoU quand un box est contenu dans l'autre."""
        iou = compute_iou(
            (0, 0, 200, 200), (50, 50, 150, 150)
        )
        assert 0.0 < iou < 1.0

    def test_euclidean_same_point(self) -> None:
        """Distance = 0 entre le même point."""
        d = euclidean_distance((100, 100), (100, 100))
        assert d == 0.0

    def test_euclidean_horizontal(self) -> None:
        """Distance horizontale correcte."""
        d = euclidean_distance((0, 0), (100, 0))
        assert d == pytest.approx(100.0)

    def test_euclidean_diagonal(self) -> None:
        """Distance diagonale correcte (3-4-5 triangle)."""
        d = euclidean_distance((0, 0), (3, 4))
        assert d == pytest.approx(5.0)


# ──────────────────────────────────────────────
# Tests — TrackedEntity
# ──────────────────────────────────────────────

class TestTrackedEntity:
    """Tests de la dataclass TrackedEntity."""

    def test_create_entity(self) -> None:
        """On peut créer une TrackedEntity."""
        entity = TrackedEntity(track_id=1)
        assert entity.track_id == 1
        assert entity.status == EntityStatus.APPEARED
        assert entity.total_frames == 0

    def test_add_detection(self) -> None:
        """add_detection met à jour l'entité."""
        entity = TrackedEntity(track_id=1)
        det = make_detection(10, 20, 110, 220)
        entity.add_detection(det)

        assert entity.total_frames == 1
        assert entity.frames_lost == 0
        assert entity.status == EntityStatus.TRACKED
        assert entity.last_bbox == (10, 20, 110, 220)

    def test_is_person(self) -> None:
        """is_person détecte les class_id=0."""
        entity = TrackedEntity(track_id=1)
        entity.add_detection(make_detection(0, 0, 50, 50, class_id=0))
        assert entity.is_person is True

        entity2 = TrackedEntity(track_id=2)
        entity2.add_detection(
            make_detection(0, 0, 50, 50, class_id=24)
        )
        assert entity2.is_person is False

    def test_duration(self) -> None:
        """duration_seconds calcule la durée correctement."""
        entity = TrackedEntity(
            track_id=1, first_seen=100.0, last_seen=105.0
        )
        assert entity.duration_seconds == pytest.approx(5.0)


# ──────────────────────────────────────────────
# Tests — Tracker : Assignation d'IDs
# ──────────────────────────────────────────────

class TestTrackerAssignment:
    """Tests d'assignation et maintien des IDs."""

    def test_single_detection_creates_entity(
        self, tracker: Tracker
    ) -> None:
        """Une détection crée une nouvelle entité."""
        det = make_detection(100, 100, 200, 200)
        entities = tracker.update([det])

        assert len(entities) == 1
        assert entities[0].track_id == 1
        assert entities[0].status == EntityStatus.APPEARED

    def test_two_detections_two_entities(
        self, tracker: Tracker
    ) -> None:
        """Deux détections éloignées créent 2 entités."""
        dets = [
            make_detection(10, 10, 60, 60),
            make_detection(400, 400, 500, 500),
        ]
        entities = tracker.update(dets)

        assert len(entities) == 2
        ids = {e.track_id for e in entities}
        assert len(ids) == 2  # IDs distincts

    def test_same_position_keeps_id(self, tracker: Tracker) -> None:
        """La même position dans 2 frames garde le même ID."""
        det1 = make_detection(100, 100, 200, 200)
        entities1 = tracker.update([det1])
        tid = entities1[0].track_id

        det2 = make_detection(105, 105, 205, 205)  # Légèrement bougé
        entities2 = tracker.update([det2])

        assert len(entities2) == 1
        assert entities2[0].track_id == tid

    def test_stable_ids_multiple_frames(
        self, tracker: Tracker
    ) -> None:
        """Les IDs restent stables sur plusieurs frames."""
        # 2 personnes dans la scène
        for i in range(5):
            dets = [
                make_detection(100 + i, 100, 200 + i, 200),
                make_detection(400 + i, 400, 500 + i, 500),
            ]
            entities = tracker.update(dets)

        assert len(entities) == 2
        # Les IDs devraient être 1 et 2
        ids = sorted([e.track_id for e in entities])
        assert ids == [1, 2]


# ──────────────────────────────────────────────
# Tests — Tracker : Cycle de vie
# ──────────────────────────────────────────────

class TestTrackerLifecycle:
    """Tests du cycle de vie APPEARED → TRACKED → LOST → DISAPPEARED."""

    def test_appeared_then_tracked(self, tracker: Tracker) -> None:
        """APPEARED → TRACKED après une seconde détection."""
        det = make_detection(100, 100, 200, 200)
        entities = tracker.update([det])
        assert entities[0].status == EntityStatus.APPEARED

        det2 = make_detection(105, 105, 205, 205)
        entities = tracker.update([det2])
        assert entities[0].status == EntityStatus.TRACKED


# ──────────────────────────────────────────────
# Tests — Tracker : Intégration FaceManager
# ──────────────────────────────────────────────

class TestTrackerFaceIntegration:
    """Tests de l'intégration tracker -> face_manager."""

    def test_apply_face_recognition_calls_manager(
        self, tracker: Tracker
    ) -> None:
        """Le tracker appelle recognize_entity pour chaque personne."""
        det = make_detection(100, 100, 200, 200, class_id=0)
        tracker.update([det])

        calls: list[int] = []

        class FakeFaceManager:
            def recognize_entity(self, frame, entity, frame_id):
                calls.append(entity.track_id)
                entity.face_status = "known"
                return entity

        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        entities = tracker.apply_face_recognition(
            frame, FakeFaceManager(), frame_id=1
        )

        assert len(calls) == 1
        assert entities[0].face_status == "known"

    def test_lost_after_no_detection(self, tracker: Tracker) -> None:
        """TRACKED → LOST quand pas de détection."""
        det = make_detection(100, 100, 200, 200)
        tracker.update([det])

        # Frames sans détection
        entities = tracker.update([])
        assert len(entities) >= 1
        lost = [
            e for e in entities
            if e.status == EntityStatus.LOST
        ]
        assert len(lost) >= 1

    def test_disappeared_after_max_lost(
        self, tracker: Tracker
    ) -> None:
        """LOST → DISAPPEARED après max_lost_frames."""
        det = make_detection(100, 100, 200, 200)
        tracker.update([det])

        # Assez de frames vides pour disparaître (max_lost=5)
        for _ in range(6):
            tracker.update([])

        assert tracker.active_count == 0

    def test_recover_from_lost(self, tracker: Tracker) -> None:
        """Une entité LOST redevient TRACKED si re-détectée."""
        det = make_detection(100, 100, 200, 200)
        entities = tracker.update([det])
        tid = entities[0].track_id

        # Perdre pendant 2 frames
        tracker.update([])
        tracker.update([])

        # Retrouver
        det2 = make_detection(105, 105, 205, 205)
        entities = tracker.update([det2])

        entity = tracker.get_entity(tid)
        assert entity is not None
        assert entity.status == EntityStatus.TRACKED
        assert entity.frames_lost == 0

    def test_no_detection_empty_update(
        self, tracker: Tracker
    ) -> None:
        """Mise à jour avec 0 détections et 0 entités."""
        entities = tracker.update([])
        assert entities == []


# ──────────────────────────────────────────────
# Tests — Tracker : Statistiques
# ──────────────────────────────────────────────

class TestTrackerStats:
    """Tests des statistiques du tracker."""

    def test_person_count(self, tracker: Tracker) -> None:
        """person_count compte les personnes."""
        dets = [
            make_detection(10, 10, 60, 60, class_id=0),
            make_detection(200, 200, 300, 300, class_id=24),
        ]
        tracker.update(dets)

        assert tracker.person_count == 1
        assert tracker.active_count == 2

    def test_total_appeared(self, tracker: Tracker) -> None:
        """total_appeared s'incrémente."""
        tracker.update([make_detection(10, 10, 60, 60)])
        tracker.update([
            make_detection(10, 10, 60, 60),
            make_detection(400, 400, 500, 500),
        ])

        assert tracker.total_appeared >= 2

    def test_reset(self, tracker: Tracker) -> None:
        """reset() remet tout à zéro."""
        tracker.update([make_detection(10, 10, 60, 60)])
        tracker.reset()

        assert tracker.active_count == 0
        assert tracker.total_appeared == 0

    def test_repr(self, tracker: Tracker) -> None:
        """__repr__ est informatif."""
        r = repr(tracker)
        assert "Tracker(" in r


# ──────────────────────────────────────────────
# Tests — Tracker : Événements
# ──────────────────────────────────────────────

class TestTrackerEvents:
    """Tests d'émission d'événements."""

    def test_entity_appeared_event(
        self, config: Config, event_bus: EventBus
    ) -> None:
        """L'événement 'entity_appeared' est émis."""
        events: list[dict] = []
        event_bus.subscribe(
            "entity_appeared", lambda d: events.append(d)
        )

        tracker = Tracker(config, event_bus=event_bus)
        tracker.update([make_detection(10, 10, 60, 60)])

        assert len(events) == 1
        assert "track_id" in events[0]

    def test_entity_disappeared_event(
        self, config: Config, event_bus: EventBus
    ) -> None:
        """L'événement 'entity_disappeared' est émis."""
        events: list[dict] = []
        event_bus.subscribe(
            "entity_disappeared", lambda d: events.append(d)
        )

        tracker = Tracker(
            config, event_bus=event_bus, max_lost_frames=3
        )
        tracker.update([make_detection(10, 10, 60, 60)])

        for _ in range(4):
            tracker.update([])

        assert len(events) >= 1
        assert "track_id" in events[0]
        assert "duration" in events[0]
