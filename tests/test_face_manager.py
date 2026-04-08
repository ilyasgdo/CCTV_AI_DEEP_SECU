"""Tests unitaires pour le module de reconnaissance faciale."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import Config
from src.core.detector import Detection
from src.core.face_manager import FaceManager, WhitelistRepository
from src.core.tracker import TrackedEntity
from src.utils.event_bus import EventBus


class MarkerEmbedder:
    """Embedder de test base sur un marqueur pixel [0,0,0]."""

    def __init__(self, mapping: dict[int, np.ndarray]) -> None:
        self._mapping = mapping

    def extract_embeddings(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        marker = int(image_bgr[0, 0, 0])
        embedding = self._mapping.get(marker)
        if embedding is None:
            return []
        return [embedding]


def make_embedding(index: int) -> np.ndarray:
    """Construit un embedding unitaire normalise."""
    vec = np.zeros((512,), dtype=np.float32)
    vec[index % 512] = 1.0
    return vec


def make_face_like_image(marker: int) -> np.ndarray:
    """Construit une image avec texture/contraste suffisants."""
    image = np.zeros((180, 180, 3), dtype=np.uint8)
    image[:, :] = 110
    cv2.circle(image, (90, 90), 65, (170, 170, 170), -1)
    cv2.circle(image, (70, 80), 12, (40, 40, 40), -1)
    cv2.circle(image, (110, 80), 12, (40, 40, 40), -1)
    cv2.ellipse(image, (90, 120), (25, 12), 0, 0, 180, (60, 60, 60), 2)
    image[0, 0, 0] = marker
    return image


def make_entity(track_id: int, bbox: tuple[int, int, int, int]) -> TrackedEntity:
    """Cree une entite personne trackee pour les tests."""
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.95,
        bbox=bbox,
        center=((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
        frame_id=1,
        timestamp=time.time(),
    )
    entity = TrackedEntity(track_id=track_id)
    entity.add_detection(detection)
    return entity


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def whitelist_repo(tmp_path: Path) -> WhitelistRepository:
    return WhitelistRepository(tmp_path / "whitelist")


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


def test_enroll_from_photos_creates_registry_and_embeddings(
    config: Config,
    whitelist_repo: WhitelistRepository,
    tmp_path: Path,
) -> None:
    """Enrollement photo avec 3 images cree les fichiers attendus."""
    known_embedding = make_embedding(7)
    embedder = MarkerEmbedder({10: known_embedding, 11: known_embedding, 12: known_embedding})

    manager = FaceManager(
        config=config,
        repository=whitelist_repo,
        embedder=embedder,
        unknown_snapshots_dir=tmp_path / "unknowns",
    )

    photo_paths: list[str] = []
    for marker in [10, 11, 12]:
        image = make_face_like_image(marker)
        path = tmp_path / f"input_{marker}.png"
        cv2.imwrite(str(path), image)
        photo_paths.append(str(path))

    person = manager.enroll_from_photos(
        name="Ilyas Ghandaoui",
        role="Proprietaire",
        access_level="admin",
        photo_paths=photo_paths,
        notes="test",
    )

    assert person["id"].startswith("person_")
    assert len(person["embeddings"]) == 3
    assert whitelist_repo.get_person(person["id"]) is not None


def test_recognize_known_person_emits_event(
    config: Config,
    whitelist_repo: WhitelistRepository,
    bus: EventBus,
    tmp_path: Path,
) -> None:
    """Une personne connue est reconnue avec event face_recognized."""
    known = make_embedding(21)
    embedder = MarkerEmbedder({21: known})

    whitelist_repo.save_embedding("known_001.npy", known)
    whitelist_repo.add_person(
        {
            "id": "person_001",
            "name": "Ilyas",
            "role": "Owner",
            "access_level": "admin",
            "embeddings": ["known_001.npy"],
            "photos": [],
            "enrolled_at": "2026-04-08T12:00:00Z",
            "last_seen": None,
            "notes": "",
        }
    )

    events: list[dict] = []
    bus.subscribe("face_recognized", lambda d: events.append(d))

    manager = FaceManager(
        config=config,
        event_bus=bus,
        repository=whitelist_repo,
        embedder=embedder,
        unknown_snapshots_dir=tmp_path / "unknowns",
    )

    frame = make_face_like_image(21)
    entity = make_entity(track_id=1, bbox=(20, 20, 160, 160))
    manager.recognize_entity(frame, entity, frame_id=1)

    assert entity.face_status == "known"
    assert entity.face_id == "person_001"
    assert entity.face_name == "Ilyas"
    assert len(events) == 1


def test_recognize_unknown_creates_snapshot_and_event(
    config: Config,
    whitelist_repo: WhitelistRepository,
    bus: EventBus,
    tmp_path: Path,
) -> None:
    """Un inconnu produit snapshot + evenement face_unknown."""
    known = make_embedding(1)
    unknown = make_embedding(99)
    embedder = MarkerEmbedder({1: known, 99: unknown})

    whitelist_repo.save_embedding("known.npy", known)
    whitelist_repo.add_person(
        {
            "id": "person_002",
            "name": "Marie",
            "role": "Guest",
            "access_level": "user",
            "embeddings": ["known.npy"],
            "photos": [],
            "enrolled_at": "2026-04-08T12:00:00Z",
            "last_seen": None,
            "notes": "",
        }
    )

    unknown_events: list[dict] = []
    bus.subscribe("face_unknown", lambda d: unknown_events.append(d))

    unknown_dir = tmp_path / "unknowns"
    manager = FaceManager(
        config=config,
        event_bus=bus,
        repository=whitelist_repo,
        embedder=embedder,
        unknown_snapshots_dir=unknown_dir,
    )

    frame = make_face_like_image(99)
    entity = make_entity(track_id=2, bbox=(20, 20, 160, 160))
    manager.recognize_entity(frame, entity, frame_id=4)

    assert entity.face_status == "unknown"
    assert len(unknown_events) == 1

    snapshot_path = Path(unknown_events[0]["snapshot_path"])
    assert snapshot_path.exists()
    assert str(snapshot_path).startswith(str(unknown_dir))


def test_recognition_cache_respects_recalculate_interval(
    config: Config,
    whitelist_repo: WhitelistRepository,
    tmp_path: Path,
) -> None:
    """Le cache evite le recalcul avant l'intervalle configure."""

    class CountingEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def extract_embeddings(self, image_bgr: np.ndarray) -> list[np.ndarray]:
            self.calls += 1
            return [make_embedding(42)]

    embedder = CountingEmbedder()
    whitelist_repo.save_embedding("known.npy", make_embedding(42))
    whitelist_repo.add_person(
        {
            "id": "person_003",
            "name": "Known",
            "role": "Owner",
            "access_level": "admin",
            "embeddings": ["known.npy"],
            "photos": [],
            "enrolled_at": "2026-04-08T12:00:00Z",
            "last_seen": None,
            "notes": "",
        }
    )

    config.face.recalculate_interval = 30

    manager = FaceManager(
        config=config,
        repository=whitelist_repo,
        embedder=embedder,
        unknown_snapshots_dir=tmp_path / "unknowns",
    )

    frame = make_face_like_image(42)
    entity = make_entity(track_id=5, bbox=(20, 20, 160, 160))

    manager.recognize_entity(frame, entity, frame_id=1)
    manager.recognize_entity(frame, entity, frame_id=10)

    assert embedder.calls == 1


def test_recognition_latency_under_100ms_with_mock_backend(
    config: Config,
    whitelist_repo: WhitelistRepository,
    tmp_path: Path,
) -> None:
    """Verification perf cible (<100ms) avec backend mocke."""
    emb = make_embedding(8)
    whitelist_repo.save_embedding("p.npy", emb)
    whitelist_repo.add_person(
        {
            "id": "person_010",
            "name": "Perf",
            "role": "Owner",
            "access_level": "admin",
            "embeddings": ["p.npy"],
            "photos": [],
            "enrolled_at": "2026-04-08T12:00:00Z",
            "last_seen": None,
            "notes": "",
        }
    )

    manager = FaceManager(
        config=config,
        repository=whitelist_repo,
        embedder=MarkerEmbedder({8: emb}),
        unknown_snapshots_dir=tmp_path / "unknowns",
    )

    frame = make_face_like_image(8)
    entity = make_entity(track_id=9, bbox=(20, 20, 160, 160))

    t0 = time.perf_counter()
    manager.recognize_entity(frame, entity, frame_id=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 100.0
