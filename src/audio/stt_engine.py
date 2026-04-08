"""Moteur Speech-to-Text simplifie avec mode simulation et sync TTS/STT."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from src.core.config import Config
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Transcript:
    """Resultat d'une transcription audio."""

    text: str
    confidence: float
    duration: float


class STTEngine:
    """Moteur de transcription avec boucle thread et VAD simplifie.

    Notes:
        - Utilise un mode simulation via `submit_audio_text` pour tests.
        - Peut etre branche a une source micro plus tard.
    """

    def __init__(self, config: Config, event_bus: Optional[EventBus] = None) -> None:
        self.model_size = config.audio.stt_model
        self.language = config.audio.stt_language
        self.enabled = bool(config.audio.stt_enabled)
        self.buffer_seconds = int(config.audio.buffer_seconds)

        self._event_bus = event_bus
        self._running = threading.Event()
        self._paused_by_tts = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._segments: deque[Transcript] = deque(maxlen=50)
        self._lock = threading.Lock()
        self._last_transcript: Optional[Transcript] = None

        if self._event_bus:
            self._event_bus.subscribe("tts_speaking", self._on_tts_speaking)
            self._event_bus.subscribe("tts_done", self._on_tts_done)

    def _on_tts_speaking(self, _: dict) -> None:
        """Met STT en pause pendant la parole TTS."""
        self._paused_by_tts.set()

    def _on_tts_done(self, _: dict) -> None:
        """Reprend STT 500 ms apres la fin TTS."""
        def resume() -> None:
            time.sleep(0.5)
            self._paused_by_tts.clear()

        threading.Thread(target=resume, daemon=True).start()

    def start_listening(self) -> None:
        """Demarre la boucle d'ecoute STT en arriere-plan."""
        if not self.enabled or self._running.is_set():
            return

        self._running.set()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop_listening(self) -> None:
        """Arrete la boucle d'ecoute STT."""
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def submit_audio_text(self, text: str, confidence: float = 0.9, duration: float = 1.0) -> None:
        """Injection d'un segment texte pour simulation/tests."""
        if not text.strip():
            return

        with self._lock:
            self._segments.append(Transcript(text=text.strip(), confidence=confidence, duration=duration))

    def get_transcript(self) -> Optional[str]:
        """Retourne la derniere transcription disponible."""
        with self._lock:
            if self._last_transcript is None:
                return None
            return self._last_transcript.text

    def _listen_loop(self) -> None:
        """Boucle STT simplifiee (consomme segments injectes)."""
        while self._running.is_set():
            if self._paused_by_tts.is_set():
                time.sleep(0.05)
                continue

            segment: Optional[Transcript] = None
            with self._lock:
                if self._segments:
                    segment = self._segments.popleft()

            if segment is None:
                if self._event_bus:
                    self._event_bus.emit("audio_silence", {})
                time.sleep(0.2)
                continue

            with self._lock:
                self._last_transcript = segment

            if self._event_bus:
                self._event_bus.emit(
                    "audio_transcribed",
                    {
                        "text": segment.text,
                        "confidence": segment.confidence,
                        "duration": segment.duration,
                    },
                )

            time.sleep(0.05)
