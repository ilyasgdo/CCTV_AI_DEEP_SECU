"""Tests unitaires du moteur STT et synchronisation avec TTS."""

from __future__ import annotations

import time

from src.audio.stt_engine import STTEngine
from src.core.config import Config
from src.utils.event_bus import EventBus


def test_stt_basic_transcription() -> None:
    """Un segment injecte est transcrit puis expose par get_transcript."""
    cfg = Config()
    bus = EventBus()
    events: list[dict] = []
    bus.subscribe("audio_transcribed", lambda d: events.append(d))

    stt = STTEngine(cfg, bus)
    stt.start_listening()
    stt.submit_audio_text("bonjour sentinel", confidence=0.95, duration=1.2)

    time.sleep(0.35)
    stt.stop_listening()

    assert stt.get_transcript() == "bonjour sentinel"
    assert len(events) >= 1


def test_stt_silence_event() -> None:
    """Sans segment, le moteur emet audio_silence."""
    cfg = Config()
    bus = EventBus()
    silence_count = {"n": 0}
    bus.subscribe("audio_silence", lambda d: silence_count.__setitem__("n", silence_count["n"] + 1))

    stt = STTEngine(cfg, bus)
    stt.start_listening()

    time.sleep(0.35)
    stt.stop_listening()

    assert silence_count["n"] >= 1


def test_tts_stt_sync_pause_resume() -> None:
    """Quand TTS parle, STT se met en pause puis reprend."""
    cfg = Config()
    bus = EventBus()
    transcribed: list[str] = []
    bus.subscribe("audio_transcribed", lambda d: transcribed.append(d["text"]))

    stt = STTEngine(cfg, bus)
    stt.start_listening()

    bus.emit("tts_speaking", {"text": "ia parle"})
    stt.submit_audio_text("doit attendre")
    time.sleep(0.2)
    assert transcribed == []

    bus.emit("tts_done", {"text": "fin"})
    time.sleep(0.75)
    stt.stop_listening()

    assert "doit attendre" in transcribed
