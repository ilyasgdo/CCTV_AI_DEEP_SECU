"""Tests unitaires des outils effecteurs et du dispatcher."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from src.audio.tts_engine import TTSEngine
from src.cognitive.response_parser import ToolAction
from src.core.config import Config
from src.effector.alarm_tool import AlarmTool
from src.effector.email_tool import EmailTool
from src.effector.event_log_tool import EventLogTool
from src.effector.snapshot_tool import SnapshotTool
from src.effector.tool_executor import ToolExecutor
from src.utils.event_bus import EventBus


class FakeSMTP:
    """Fake SMTP client pour tests email."""

    def __init__(self, host: str, port: int, timeout: int) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sent = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starttls(self) -> None:
        return None

    def login(self, user: str, pwd: str) -> None:
        return None

    def send_message(self, msg) -> None:
        self.sent = True


def test_email_tool_success(monkeypatch) -> None:
    """EmailTool renvoie success quand SMTP est configure."""
    cfg = Config()

    monkeypatch.setenv("SMTP_HOST", "smtp.test")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "bot@test")
    monkeypatch.setenv("SMTP_PASSWORD", "pwd")
    monkeypatch.setenv("ALERT_EMAIL_TO", "to@test")

    tool = EmailTool(cfg, smtp_factory=FakeSMTP)
    result = asyncio.run(tool.run({"sujet": "Alerte", "urgence": "haute"}))

    assert result.success is True


def test_alarm_tool_basic() -> None:
    """AlarmTool execute correctement une alarme."""

    called = {"ok": False}

    async def fake_play(kind: str, duration: int) -> None:
        called["ok"] = True

    tool = AlarmTool(play_fn=fake_play)
    result = asyncio.run(tool.run({"type": "intrusion", "duration": 3}))

    assert result.success is True
    assert called["ok"] is True


def test_snapshot_tool_saves_file(tmp_path: Path) -> None:
    """SnapshotTool enregistre une image JPEG dans data/snapshots."""
    cfg = Config()

    class FakeCamera:
        def get_frame(self):
            return np.zeros((80, 120, 3), dtype=np.uint8)

    tool = SnapshotTool(cfg, camera=FakeCamera())
    result = asyncio.run(tool.run({"reason": "test"}))

    assert result.success is True
    path = Path(result.data["snapshot_path"])
    assert path.exists()


def test_event_log_tool_appends_event() -> None:
    """EventLogTool ajoute bien un event JSON."""
    cfg = Config()
    tool = EventLogTool(cfg)

    result = asyncio.run(tool.run({"event_type": "security", "message": "ok"}))

    assert result.success is True
    log_file = cfg.project_root / "data" / "event_log.json"
    assert log_file.exists()
    data = json.loads(log_file.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert any(e.get("event_type") == "security" for e in data)


def test_tool_executor_runs_and_handles_unknown() -> None:
    """ToolExecutor execute announce + gere unknown tool sans crash."""
    cfg = Config()
    bus = EventBus()

    async def fake_synth(text: str, voice: str) -> bytes:
        return text.encode("utf-8")

    async def scenario() -> list:
        tts = TTSEngine(cfg, bus, synthesize_fn=fake_synth)
        executor = ToolExecutor(cfg, bus, tts_engine=tts)

        actions = [
            ToolAction(tool_name="announce", parametres={"text": "bonjour"}),
            ToolAction(tool_name="unknown_tool", parametres={}),
        ]

        results = await executor.execute(actions)
        await asyncio.sleep(0.05)
        await tts.shutdown()
        return results

    results = asyncio.run(scenario())

    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is False
