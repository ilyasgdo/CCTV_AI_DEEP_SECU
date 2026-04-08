"""Moteur Text-to-Speech asynchrone base sur edge-tts."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from src.core.config import Config
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger

logger = get_logger(__name__)

SynthesizeFn = Callable[[str, str], Awaitable[bytes]]


class TTSEngine:
    """Moteur de synthese vocale avec file FIFO et priorite.

    Args:
        config: Configuration globale.
        event_bus: Bus d'evenements optionnel.
        synthesize_fn: Fonction de synthese injectable (tests).
    """

    def __init__(
        self,
        config: Config,
        event_bus: Optional[EventBus] = None,
        synthesize_fn: Optional[SynthesizeFn] = None,
    ) -> None:
        self.voice = config.audio.tts_voice
        self.enabled = bool(config.audio.tts_enabled)
        self.speaking = False

        self._event_bus = event_bus
        self._synthesize_fn = synthesize_fn or self._default_synthesize
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._cache: dict[str, bytes] = {}
        self._cache_limit = 128

    def _ensure_worker(self) -> None:
        """Demarre le worker si necessaire."""
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(self._worker())

    async def speak(self, text: str) -> None:
        """Ajoute un message dans la file de synthese."""
        if not self.enabled or not text.strip():
            return
        self._ensure_worker()
        await self._queue.put(text.strip())

    async def speak_priority(self, text: str) -> None:
        """Interrompt la file courante et place un message prioritaire."""
        if not self.enabled or not text.strip():
            return
        self._ensure_worker()

        # Purger la file pour donner priorite au nouveau message.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        await self._queue.put(text.strip())

    async def shutdown(self) -> None:
        """Arrete proprement le worker TTS."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)

    async def _worker(self) -> None:
        """Consomme les messages TTS sequentiellement."""
        while True:
            text = await self._queue.get()
            try:
                self.speaking = True
                if self._event_bus:
                    self._event_bus.emit("tts_speaking", {"text": text})

                await self._speak_once(text)

            finally:
                self.speaking = False
                if self._event_bus:
                    self._event_bus.emit("tts_done", {"text": text})
                self._queue.task_done()

    async def _speak_once(self, text: str) -> None:
        """Synthese d'un message unique (avec cache memoire)."""
        if text in self._cache:
            _ = self._cache[text]
            # Simule une petite duree de sortie audio en lecture cache.
            await asyncio.sleep(min(1.5, max(0.05, len(text) * 0.01)))
            return

        audio_bytes = await self._synthesize_fn(text, self.voice)

        if len(self._cache) >= self._cache_limit:
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = audio_bytes

    async def _default_synthesize(self, text: str, voice: str) -> bytes:
        """Synthese edge-tts par defaut.

        Cette implementation produit un flux audio en memoire sans lecture
        locale obligatoire, afin de rester robuste sur tous environnements.
        """
        try:
            import edge_tts  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("edge-tts indisponible, fallback silencieux")
            await asyncio.sleep(min(1.5, max(0.05, len(text) * 0.01)))
            return text.encode("utf-8")

        communicate = edge_tts.Communicate(text=text, voice=voice)
        chunks: list[bytes] = []
        async for part in communicate.stream():
            if part.get("type") == "audio":
                data = part.get("data")
                if isinstance(data, bytes):
                    chunks.append(data)

        if not chunks:
            await asyncio.sleep(min(1.5, max(0.05, len(text) * 0.01)))
            return b""

        return b"".join(chunks)
