"""Moteur Text-to-Speech asynchrone base sur edge-tts."""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
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
        self._custom_synthesize_fn = synthesize_fn is not None
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._cache: dict[str, bytes] = {}
        self._cache_limit = 128

        self._playback_enabled = os.getenv("SENTINEL_TTS_PLAYBACK", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        self._windows_sapi_enabled = (
            os.name == "nt"
            and os.getenv("SENTINEL_TTS_WINDOWS_SAPI", "true").lower() in {"1", "true", "yes"}
        )

        logger.info(
            "TTS initialise enabled=%s playback=%s windows_sapi=%s voice=%s",
            self.enabled,
            self._playback_enabled,
            self._windows_sapi_enabled,
            self.voice,
        )

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

                started = asyncio.get_running_loop().time()
                logger.info("TTS lecture lancee (%d caracteres)", len(text))
                try:
                    await self._speak_once(text)
                except Exception as exc:  # pragma: no cover - securite runtime
                    logger.error("Echec lecture TTS: %s", exc, exc_info=True)
                finally:
                    elapsed = asyncio.get_running_loop().time() - started
                    logger.info("TTS lecture terminee en %.2fs", elapsed)

            finally:
                self.speaking = False
                if self._event_bus:
                    self._event_bus.emit("tts_done", {"text": text})
                self._queue.task_done()

    async def _speak_once(self, text: str) -> None:
        """Synthese d'un message unique (avec cache memoire)."""
        # En tests, garder strictement le comportement injectable.
        if self._custom_synthesize_fn:
            if text in self._cache:
                _ = self._cache[text]
                await asyncio.sleep(min(1.5, max(0.05, len(text) * 0.01)))
                return

            audio_bytes = await self._synthesize_fn(text, self.voice)
            if len(self._cache) >= self._cache_limit:
                self._cache.pop(next(iter(self._cache)))
            self._cache[text] = audio_bytes
            return

        # Runtime Windows: sortie audio locale reelle via SAPI.
        if self._playback_enabled and self._windows_sapi_enabled:
            ok = await asyncio.to_thread(self._speak_windows_sapi_sync, text)
            if ok:
                return
            logger.warning("TTS Windows SAPI indisponible, fallback edge-tts")

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

    def _speak_windows_sapi_sync(self, text: str) -> bool:
        """Parle le texte via System.Speech (Windows)."""
        powershell_exe = os.getenv(
            "SENTINEL_TTS_POWERSHELL_EXE",
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        )
        if not os.path.exists(powershell_exe):
            powershell_exe = "powershell"

        try:
            encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
            command = (
                "$raw='" + encoded + "';"
                "$text=[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($raw));"
                "Add-Type -AssemblyName System.Speech;"
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                "$s.Speak($text);"
                "$s.Dispose();"
            )
            subprocess.run(
                [
                    powershell_exe,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as exc:
            logger.warning(
                "TTS Windows SAPI indisponible (%s), fallback silencieux: %s",
                powershell_exe,
                exc,
            )
            return False
