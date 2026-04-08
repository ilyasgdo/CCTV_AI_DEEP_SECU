"""Tests d'integration Etape 7 - pipeline principal Sentinel-AI."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import main as sentinel_main


@dataclass
class FakeEntity:
    """Entite minimale pour tests integration."""

    track_id: int
    face_status: str
    face_name: str | None
    duration_seconds: float

    @property
    def is_person(self) -> bool:
        return True


class FakeCamera:
    """Camera factice non bloquante."""

    class Metrics:
        fps_current = 25.0

    metrics = Metrics()

    def __init__(self, *_: Any, **__: Any) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_frame(self) -> np.ndarray:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def get_snapshot(self) -> bytes:
        return b"jpeg-bytes"


class FakeDetector:
    """Detecteur factice renvoyant une detection personne."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.calls = 0

    def detect(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        self.calls += 1
        return [{"class_name": "person", "confidence": 0.99}]


class FakeTracker:
    """Tracker factice avec deux personnes (connue et inconnue)."""

    def __init__(self, *_: Any, **kwargs: Any) -> None:
        self.event_bus = kwargs.get("event_bus")
        self._entities = [
            FakeEntity(1, "known", "Ilyas", 8.2),
            FakeEntity(2, "unknown", None, 3.1),
        ]

    def update(self, detections: list[dict[str, Any]]) -> list[FakeEntity]:
        _ = detections
        return self._entities

    def apply_face_recognition(self, frame: np.ndarray, face_manager: Any, frame_id: int) -> list[FakeEntity]:
        _ = frame
        _ = face_manager
        _ = frame_id
        return self._entities

    @property
    def active_entities(self) -> list[FakeEntity]:
        return self._entities

    @property
    def person_count(self) -> int:
        return len(self._entities)


class FakeFaceManager:
    """FaceManager factice."""

    def __init__(self, *_: Any, **__: Any) -> None:
        pass


class FakeLLMClient:
    """Client LLM factice configurable."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.healthy = True

    async def health_check(self) -> bool:
        return self.healthy


class FakeTTS:
    """TTS factice avec historique des messages."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.messages: list[str] = []
        self.shutdown_called = False

    async def speak(self, text: str) -> None:
        self.messages.append(text)

    async def shutdown(self) -> None:
        self.shutdown_called = True


class FakeSTT:
    """STT factice."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.started = False
        self.stopped = False
        self.current = "personne devant la porte"

    def start_listening(self) -> None:
        self.started = True

    def stop_listening(self) -> None:
        self.stopped = True

    def get_transcript(self) -> str:
        return self.current


class FakeToolExecutor:
    """Tool executor factice."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.executed: list[list[dict[str, Any]]] = []

    async def execute(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.executed.append(actions)
        return []


class FakeDashboardServer:
    """Serveur dashboard factice."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeOrchestrator:
    """Orchestrateur factice qui emet une reponse LLM."""

    instances: list["FakeOrchestrator"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.event_bus = kwargs["event_bus"]
        self.stopped = False
        FakeOrchestrator.instances.append(self)

    async def analysis_loop(self) -> None:
        self.event_bus.emit(
            "llm_response",
            {
                "action_vocale": "Bonjour Ilyas",
                "niveau_alerte": "normal",
                "outils_a_lancer": [
                    {"tool_name": "send_email", "parametres": {"subject": "ok"}}
                ],
            },
        )
        while not self.stopped:
            await asyncio.sleep(0.05)

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def patched_stage7(monkeypatch: pytest.MonkeyPatch):
    """Patches des composants pour tests d'integration stables."""
    FakeOrchestrator.instances.clear()
    monkeypatch.setattr(sentinel_main, "Camera", FakeCamera)
    monkeypatch.setattr(sentinel_main, "ObjectDetector", FakeDetector)
    monkeypatch.setattr(sentinel_main, "Tracker", FakeTracker)
    monkeypatch.setattr(sentinel_main, "FaceManager", FakeFaceManager)
    monkeypatch.setattr(sentinel_main, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(sentinel_main, "TTSEngine", FakeTTS)
    monkeypatch.setattr(sentinel_main, "STTEngine", FakeSTT)
    monkeypatch.setattr(sentinel_main, "ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr(sentinel_main, "AnalysisOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(sentinel_main, "create_app", lambda **_: object())
    monkeypatch.setattr(sentinel_main, "DashboardServer", FakeDashboardServer)
    yield


def _run(coro: Any) -> Any:
    """Execute une coroutine dans une boucle asyncio locale."""
    return asyncio.run(coro)


def test_stage7_startup_complet(patched_stage7) -> None:
    """1) Startup complet: modules principaux demarres."""
    app = sentinel_main.SentinelAI()
    _run(app.start())
    assert app.camera.started is True
    assert app.stt is not None and app.stt.started is True
    assert app.dashboard_server is not None and app.dashboard_server.started is True
    _run(app.shutdown())


def test_stage7_workflow_5s(patched_stage7) -> None:
    """2) Workflow 5s: reponse LLM declenche TTS + tools."""
    app = sentinel_main.SentinelAI()
    _run(app.start())
    time.sleep(0.2)
    assert app.tts is not None and "Bonjour Ilyas" in app.tts.messages
    assert app.tool_executor is not None
    assert len(app.tool_executor.executed) >= 1
    _run(app.shutdown())


def test_stage7_connu_reconnu(patched_stage7) -> None:
    """3) Personne connue presente dans le tracker."""
    app = sentinel_main.SentinelAI()
    _run(app.start())
    time.sleep(0.1)
    entities = app._get_latest_entities()
    assert any(entity.face_status == "known" and entity.face_name == "Ilyas" for entity in entities)
    _run(app.shutdown())


def test_stage7_inconnu_detecte(patched_stage7) -> None:
    """4) Personne inconnue detectee dans les entites."""
    app = sentinel_main.SentinelAI()
    _run(app.start())
    time.sleep(0.1)
    entities = app._get_latest_entities()
    assert any(entity.face_status == "unknown" for entity in entities)
    _run(app.shutdown())


def test_stage7_tool_email(patched_stage7) -> None:
    """5) Outil send_email demande par le LLM est execute."""
    app = sentinel_main.SentinelAI()
    _run(app.start())
    time.sleep(0.2)
    assert app.tool_executor is not None
    sent = any(
        any(tool.get("tool_name") == "send_email" for tool in batch)
        for batch in app.tool_executor.executed
    )
    assert sent is True
    _run(app.shutdown())


def test_stage7_dashboard_live(patched_stage7) -> None:
    """6) Dashboard demarre en mode nominal."""
    app = sentinel_main.SentinelAI()
    _run(app.start())
    assert app.dashboard_server is not None and app.dashboard_server.started is True
    _run(app.shutdown())


def test_stage7_change_llm_url(patched_stage7) -> None:
    """7) URL LLM modifiable a chaud via config."""
    app = sentinel_main.SentinelAI()
    app.config.llm.api_url = "http://127.0.0.1:11555"
    assert app.config.llm.api_url.endswith("11555")
    _run(app.start())
    _run(app.shutdown())


def test_stage7_graceful_stop(patched_stage7) -> None:
    """8) Arret propre: camera/stt/tts/dashboard stoppees."""
    app = sentinel_main.SentinelAI()
    _run(app.start())
    _run(app.shutdown())
    assert app.camera.stopped is True
    assert app.stt is not None and app.stt.stopped is True
    assert app.tts is not None and app.tts.shutdown_called is True
    assert app.dashboard_server is not None and app.dashboard_server.stopped is True


def test_stage7_no_llm_mode_degrade(patched_stage7) -> None:
    """9) LLM indisponible active le mode degrade."""
    app = sentinel_main.SentinelAI()
    assert isinstance(app.llm_client, FakeLLMClient)
    app.llm_client.healthy = False
    _run(app.start())
    assert app._degraded_llm is True
    assert app._orchestrator_task is None
    _run(app.shutdown())


def test_stage7_camera_lost_event(patched_stage7) -> None:
    """10) Camera lost: l'evenement est accepte sans crash."""
    app = sentinel_main.SentinelAI()
    _run(app.start())
    notified = app.event_bus.emit("camera_disconnected", {"reason": "test"})
    assert notified >= 1
    _run(app.shutdown())


def test_stage7_multi_person(patched_stage7) -> None:
    """11) Multi-person: connue + inconnue differenciees."""
    app = sentinel_main.SentinelAI()
    _run(app.start())
    time.sleep(0.1)
    entities = app._get_latest_entities()
    known_count = sum(1 for e in entities if e.face_status == "known")
    unknown_count = sum(1 for e in entities if e.face_status == "unknown")
    assert known_count == 1
    assert unknown_count == 1
    _run(app.shutdown())


def test_stage7_conversation_context(patched_stage7) -> None:
    """12) Contexte conversationnel glissant disponible."""
    app = sentinel_main.SentinelAI()
    app.memory.add_exchange("entree", "bonjour", "normal")
    app.memory.add_exchange("sortie", "au revoir", "normal")
    summary = app.memory.get_context_summary()
    assert "Contexte=entree" in summary
    assert "Contexte=sortie" in summary
