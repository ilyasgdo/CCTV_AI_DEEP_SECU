"""
Tests unitaires pour src/core/camera.py

Teste la caméra avec un VideoCapture mocké pour éviter de
dépendre d'un périphérique réel.
"""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.core.camera import Camera, CameraMetrics, CameraError
from src.core.config import Config
from src.utils.event_bus import EventBus
from src.utils.logger import reset_logging


@pytest.fixture(autouse=True)
def clean() -> None:
    """Reset le logging entre les tests."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def config() -> Config:
    """Config par défaut."""
    return Config()


@pytest.fixture
def event_bus() -> EventBus:
    """EventBus frais."""
    return EventBus()


@pytest.fixture
def fake_frame() -> np.ndarray:
    """Crée une frame test 640x480 BGR."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ──────────────────────────────────────────────
# Tests — Initialisation
# ──────────────────────────────────────────────

class TestCameraInit:
    """Tests d'initialisation de la caméra."""

    def test_init_default(self, config: Config) -> None:
        """La caméra s'initialise avec les valeurs par défaut."""
        camera = Camera(config)
        assert camera.is_running is False
        assert camera.is_connected is False
        assert camera.frame_id == 0

    def test_init_with_event_bus(
        self, config: Config, event_bus: EventBus
    ) -> None:
        """La caméra accepte un EventBus."""
        camera = Camera(config, event_bus=event_bus)
        assert camera is not None

    def test_parse_source_int(self) -> None:
        """Les sources numériques sont parsées en int."""
        assert Camera._parse_source("0") == 0
        assert Camera._parse_source("1") == 1

    def test_parse_source_url(self) -> None:
        """Les URLs restent en string."""
        url = "rtsp://192.168.1.10:554/stream"
        assert Camera._parse_source(url) == url

    def test_parse_source_file(self) -> None:
        """Les chemins de fichiers restent en string."""
        path = "./test_video.mp4"
        assert Camera._parse_source(path) == path

    def test_metrics_initial(self, config: Config) -> None:
        """Les métriques initiales sont à zéro."""
        camera = Camera(config)
        m = camera.metrics
        assert m.fps_current == 0.0
        assert m.frames_captured == 0
        assert m.frames_dropped == 0
        assert m.is_connected is False


# ──────────────────────────────────────────────
# Tests — get_frame / get_snapshot
# ──────────────────────────────────────────────

class TestCameraFrames:
    """Tests de récupération de frames."""

    def test_get_frame_no_frames(self, config: Config) -> None:
        """get_frame retourne None si pas de frame."""
        camera = Camera(config)
        assert camera.get_frame() is None

    def test_get_frame_returns_copy(
        self, config: Config, fake_frame: np.ndarray
    ) -> None:
        """get_frame retourne une copie de la frame."""
        camera = Camera(config)
        # Simuler un frame dans le buffer
        camera._buffer.append(fake_frame)

        frame1 = camera.get_frame()
        frame2 = camera.get_frame()
        assert frame1 is not frame2  # Copies distinctes
        assert np.array_equal(frame1, frame2)

    def test_get_snapshot_no_frame(self, config: Config) -> None:
        """get_snapshot retourne None sans frame."""
        camera = Camera(config)
        assert camera.get_snapshot() is None

    def test_get_snapshot_returns_jpeg(
        self, config: Config, fake_frame: np.ndarray
    ) -> None:
        """get_snapshot retourne des bytes JPEG."""
        camera = Camera(config)
        camera._buffer.append(fake_frame)

        jpeg = camera.get_snapshot()
        assert jpeg is not None
        assert isinstance(jpeg, bytes)
        # Les JPEG commencent par le magic number FF D8
        assert jpeg[:2] == b'\xff\xd8'

    def test_get_snapshot_base64(
        self, config: Config, fake_frame: np.ndarray
    ) -> None:
        """get_snapshot_base64 retourne une string base64."""
        camera = Camera(config)
        camera._buffer.append(fake_frame)

        b64 = camera.get_snapshot_base64()
        assert b64 is not None
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_get_buffer_frames(
        self, config: Config, fake_frame: np.ndarray
    ) -> None:
        """get_buffer_frames retourne les frames du buffer."""
        camera = Camera(config)
        for _ in range(5):
            camera._buffer.append(fake_frame.copy())

        frames = camera.get_buffer_frames()
        assert len(frames) == 5

        frames_2 = camera.get_buffer_frames(count=3)
        assert len(frames_2) == 3

    def test_buffer_circular(
        self, config: Config, fake_frame: np.ndarray
    ) -> None:
        """Le buffer circulaire limite la taille."""
        config.camera.buffer_size = 5
        camera = Camera(config)

        for i in range(10):
            frame = fake_frame.copy()
            frame[0, 0, 0] = i  # Marquer chaque frame
            camera._buffer.append(frame)

        assert len(camera._buffer) == 5
        # Le premier frame doit être le 6ème ajouté (index 5)
        assert camera._buffer[0][0, 0, 0] == 5


# ──────────────────────────────────────────────
# Tests — Propriétés
# ──────────────────────────────────────────────

class TestCameraProperties:
    """Tests des propriétés."""

    def test_repr(self, config: Config) -> None:
        """__repr__ est informatif."""
        camera = Camera(config)
        r = repr(camera)
        assert "Camera(" in r
        assert "source=" in r

    def test_stop_without_start(self, config: Config) -> None:
        """stop() sans start() ne plante pas."""
        camera = Camera(config)
        camera.stop()  # Ne doit pas lever d'exception


# ──────────────────────────────────────────────
# Tests — Événements
# ──────────────────────────────────────────────

class TestCameraEvents:
    """Tests d'émission d'événements."""

    def test_connect_emits_event(
        self, config: Config, event_bus: EventBus
    ) -> None:
        """La connexion émet 'camera_connected'."""
        events: list[dict] = []
        event_bus.subscribe(
            "camera_connected", lambda d: events.append(d)
        )

        camera = Camera(config, event_bus=event_bus)

        # Mock VideoCapture
        with patch("src.core.camera.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_instance.get.return_value = 30.0
            mock_cap.return_value = mock_instance

            result = camera._connect()
            assert result is True
            assert len(events) == 1
            assert "source" in events[0]

        camera._release_capture()
