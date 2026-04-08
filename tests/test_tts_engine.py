"""Tests unitaires du moteur TTS."""

from __future__ import annotations

import asyncio

from src.audio.tts_engine import TTSEngine
from src.core.config import Config
from src.utils.event_bus import EventBus


def test_tts_basic_emits_events() -> None:
    """speak emet tts_speaking puis tts_done."""
    cfg = Config()
    bus = EventBus()
    events: list[str] = []

    bus.subscribe("tts_speaking", lambda d: events.append("start"))
    bus.subscribe("tts_done", lambda d: events.append("done"))

    async def fake_synth(text: str, voice: str) -> bytes:
        await asyncio.sleep(0.01)
        return text.encode("utf-8")

    async def scenario() -> None:
        tts = TTSEngine(cfg, bus, synthesize_fn=fake_synth)
        await tts.speak("bonjour")
        await asyncio.sleep(0.05)
        await tts.shutdown()

    asyncio.run(scenario())

    assert events[:2] == ["start", "done"]


def test_tts_priority_clears_queue() -> None:
    """speak_priority purge la file et priorise le nouveau message."""
    cfg = Config()
    spoken: list[str] = []

    async def fake_synth(text: str, voice: str) -> bytes:
        spoken.append(text)
        await asyncio.sleep(0.02)
        return text.encode("utf-8")

    async def scenario() -> None:
        tts = TTSEngine(cfg, synthesize_fn=fake_synth)
        await tts.speak("m1")
        await tts.speak("m2")
        await tts.speak_priority("urgent")

        await asyncio.sleep(0.12)
        await tts.shutdown()

    asyncio.run(scenario())

    assert "urgent" in spoken
    assert "m2" not in spoken
