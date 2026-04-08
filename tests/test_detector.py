"""
Tests unitaires pour src/core/detector.py

Teste le détecteur en mockant le modèle YOLO pour éviter
la dépendance au modèle réel dans les tests unitaires.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.detector import (
    ObjectDetector,
    Detection,
    PersonDetection,
    DetectorMetrics,
    ModelLoadError,
    PERSON_CLASS_ID,
    NOTABLE_CLASSES,
)
from src.core.config import Config
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
def fake_frame() -> np.ndarray:
    """Frame test 640x480."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ──────────────────────────────────────────────
# Tests — Detection dataclass
# ──────────────────────────────────────────────

class TestDetection:
    """Tests de la dataclass Detection."""

    def test_create_detection(self) -> None:
        """On peut créer une Detection."""
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.92,
            bbox=(100, 200, 300, 400),
            center=(200, 300),
            frame_id=1,
        )
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.confidence == 0.92
        assert det.bbox == (100, 200, 300, 400)
        assert det.center == (200, 300)

    def test_person_detection(self) -> None:
        """PersonDetection hérite de Detection."""
        pd = PersonDetection(
            class_id=0,
            class_name="person",
            confidence=0.88,
            bbox=(50, 50, 200, 400),
            center=(125, 225),
            keypoints=[(100, 100, 0.9)] * 17,
            pose_confidence=0.85,
        )
        assert pd.keypoints is not None
        assert len(pd.keypoints) == 17
        assert pd.pose_confidence == 0.85


# ──────────────────────────────────────────────
# Tests — ObjectDetector (Mocked YOLO)
# ──────────────────────────────────────────────

class TestObjectDetectorMocked:
    """Tests du détecteur avec YOLO mocké."""

    def _make_detector_with_mock(
        self,
        config: Config,
        event_bus: EventBus = None,
    ) -> tuple:
        """Crée un détecteur avec un modèle YOLO mocké.

        Returns:
            (detector, mock_model)
        """
        mock_model = MagicMock()

        with patch(
            "src.core.detector.ObjectDetector._load_model",
            return_value=mock_model,
        ):
            detector = ObjectDetector(config, event_bus)

        return detector, mock_model

    def _make_yolo_result(
        self,
        detections: list[dict],
    ) -> MagicMock:
        """Crée un résultat YOLO mocké.

        Args:
            detections: Liste de dict avec class_id, confidence,
                        bbox (x1, y1, x2, y2).
        """
        import torch

        result = MagicMock()

        if not detections:
            result.boxes = None
            result.names = {}
            return [result]

        boxes = []
        for det in detections:
            box = MagicMock()
            box.cls = torch.tensor([det["class_id"]],
                                   dtype=torch.float32)
            box.conf = torch.tensor([det["confidence"]],
                                    dtype=torch.float32)
            box.xyxy = torch.tensor(
                [det["bbox"]], dtype=torch.float32
            )
            boxes.append(box)

        result.boxes = boxes
        result.names = {
            0: "person", 2: "car", 24: "backpack",
            43: "knife", 67: "cell phone",
        }
        result.keypoints = None

        return [result]

    def test_detect_persons(
        self, config: Config, fake_frame: np.ndarray
    ) -> None:
        """detect() retourne des détections de personnes."""
        detector, mock_model = self._make_detector_with_mock(config)

        mock_model.return_value = self._make_yolo_result([
            {
                "class_id": 0,
                "confidence": 0.92,
                "bbox": [100, 200, 300, 400],
            },
        ])

        dets = detector.detect(fake_frame, frame_id=1)

        assert len(dets) == 1
        assert dets[0].class_id == 0
        assert dets[0].class_name == "person"
        assert dets[0].confidence == pytest.approx(0.92, abs=0.01)
        assert dets[0].bbox == (100, 200, 300, 400)
        assert dets[0].center == (200, 300)

    def test_detect_multiple(
        self, config: Config, fake_frame: np.ndarray
    ) -> None:
        """detect() retourne plusieurs objets triés par confiance."""
        detector, mock_model = self._make_detector_with_mock(config)

        mock_model.return_value = self._make_yolo_result([
            {"class_id": 0, "confidence": 0.80,
             "bbox": [10, 20, 100, 200]},
            {"class_id": 24, "confidence": 0.95,
             "bbox": [300, 100, 400, 200]},
            {"class_id": 0, "confidence": 0.92,
             "bbox": [150, 50, 250, 350]},
        ])

        dets = detector.detect(fake_frame)

        assert len(dets) == 3
        # Triés par confiance décroissante
        assert dets[0].confidence >= dets[1].confidence
        assert dets[1].confidence >= dets[2].confidence

    def test_detect_empty(
        self, config: Config, fake_frame: np.ndarray
    ) -> None:
        """detect() retourne une liste vide si rien détecté."""
        detector, mock_model = self._make_detector_with_mock(config)
        mock_model.return_value = self._make_yolo_result([])

        dets = detector.detect(fake_frame)
        assert dets == []

    def test_detect_emits_person_event(
        self,
        config: Config,
        fake_frame: np.ndarray,
        event_bus: EventBus,
    ) -> None:
        """detect() émet 'person_detected' quand personne vue."""
        events: list[dict] = []
        event_bus.subscribe(
            "person_detected", lambda d: events.append(d)
        )

        detector, mock_model = self._make_detector_with_mock(
            config, event_bus
        )
        mock_model.return_value = self._make_yolo_result([
            {"class_id": 0, "confidence": 0.90,
             "bbox": [10, 20, 100, 200]},
        ])

        detector.detect(fake_frame)

        assert len(events) == 1
        assert events[0]["count"] == 1

    def test_detect_emits_scene_empty(
        self,
        config: Config,
        fake_frame: np.ndarray,
        event_bus: EventBus,
    ) -> None:
        """detect() émet 'scene_empty' si rien détecté."""
        events: list[dict] = []
        event_bus.subscribe(
            "scene_empty", lambda d: events.append(d)
        )

        detector, mock_model = self._make_detector_with_mock(
            config, event_bus
        )
        mock_model.return_value = self._make_yolo_result([])

        detector.detect(fake_frame)
        assert len(events) == 1

    def test_metrics_updated(
        self, config: Config, fake_frame: np.ndarray
    ) -> None:
        """Les métriques sont mises à jour après chaque inférence."""
        detector, mock_model = self._make_detector_with_mock(config)
        mock_model.return_value = self._make_yolo_result([
            {"class_id": 0, "confidence": 0.90,
             "bbox": [10, 20, 100, 200]},
        ])

        detector.detect(fake_frame)

        m = detector.metrics
        assert m.total_inferences == 1
        assert m.total_detections == 1
        assert m.total_persons == 1
        assert m.last_inference_ms > 0

    def test_skip_frames(
        self, config: Config, fake_frame: np.ndarray
    ) -> None:
        """skip_frames=2 détecte 1 frame sur 2."""
        config.detection.skip_frames = 2
        detector, mock_model = self._make_detector_with_mock(config)
        mock_model.return_value = self._make_yolo_result([
            {"class_id": 0, "confidence": 0.90,
             "bbox": [10, 20, 100, 200]},
        ])

        result1 = detector.detect(fake_frame)  # frame 1 → skipped ?
        result2 = detector.detect(fake_frame)  # frame 2 → processed ?

        # Un des deux doit être vide (skipped)
        total = len(result1) + len(result2)
        assert total == 1  # Seulement 1 frame sur 2 est traitée

    def test_repr(self, config: Config) -> None:
        """__repr__ est informatif."""
        detector, _ = self._make_detector_with_mock(config)
        r = repr(detector)
        assert "ObjectDetector(" in r
        assert "model=" in r
