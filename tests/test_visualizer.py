"""Tests unitaires pour src/core/visualizer.py (Etape 8)."""

from __future__ import annotations

import time

import numpy as np

from src.core.detector import Detection, PersonDetection
from src.core.tracker import EntityStatus, TrackedEntity
from src.core.visualizer import Visualizer


def _make_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_visualizer_draw_with_detections() -> None:
    """Le visualizer annote une frame avec des detections brutes."""
    viz = Visualizer(show_hud=True, show_labels=True)
    frame = _make_frame()

    detections = [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.91,
            bbox=(100, 120, 220, 360),
            center=(160, 240),
            frame_id=1,
            timestamp=time.time(),
        ),
        Detection(
            class_id=2,
            class_name="car",
            confidence=0.75,
            bbox=(280, 180, 420, 320),
            center=(350, 250),
            frame_id=1,
            timestamp=time.time(),
        ),
    ]

    annotated = viz.draw(frame, detections=detections)
    assert annotated.shape == frame.shape
    assert np.any(annotated != frame)


def test_visualizer_draw_with_entities_and_pose() -> None:
    """Le visualizer dessine entities trackees et squelette pose."""
    viz = Visualizer(show_hud=True, show_labels=True, show_skeleton=True)
    frame = _make_frame()

    person_det = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.95,
        bbox=(120, 80, 260, 360),
        center=(190, 220),
        frame_id=2,
        timestamp=time.time(),
        keypoints=[(150.0, 100.0, 0.9)] * 17,
        pose_confidence=0.9,
    )

    entity = TrackedEntity(track_id=1, detections=[person_det], status=EntityStatus.TRACKED)
    entity.face_status = "known"
    entity.face_name = "Ilyas"

    annotated = viz.draw(frame, entities=[entity])
    assert annotated.shape == frame.shape
    assert np.any(annotated != frame)


def test_visualizer_lost_entity_hidden_from_overlay() -> None:
    """Une entite LOST n'est pas affichee dans le flux live."""
    viz = Visualizer(show_hud=False, show_labels=True)
    frame = _make_frame()

    det = Detection(
        class_id=0,
        class_name="person",
        confidence=0.8,
        bbox=(60, 60, 180, 260),
        center=(120, 160),
        frame_id=3,
        timestamp=time.time(),
    )
    entity = TrackedEntity(track_id=7, detections=[det], status=EntityStatus.LOST)
    entity.face_status = "unknown"

    annotated = viz.draw(frame, entities=[entity])
    assert annotated.shape == frame.shape
    assert np.all(annotated == frame)
