"""Orchestrateur asynchrone du cycle cognitif Sentinel-AI."""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

from src.cognitive.conversation_memory import ConversationMemory
from src.cognitive.llm_client import LLMClient, LLMClientError
from src.cognitive.prompt_manager import PromptManager
from src.cognitive.response_parser import ActionResponse, ResponseParser
from src.core.config import Config
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisOrchestrator:
    """Orchestre l'analyse periodique non-bloquante du contexte scene.

    Args:
        config: Configuration globale.
        event_bus: Bus d'evenements pour emettre llm_response.
        camera: Source camera expose get_snapshot().
        llm_client: Client LLM asynchrone.
        prompt_manager: Constructeur de prompt.
        parser: Parseur de reponse LLM.
        memory: Memoire conversationnelle glissante.
        detections_provider: Callback renvoyant la liste des detections.
        entities_provider: Callback renvoyant les entites trackees.
        audio_provider: Callback renvoyant la transcription audio courante.
    """

    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        camera: object,
        llm_client: LLMClient,
        prompt_manager: PromptManager,
        parser: ResponseParser,
        memory: ConversationMemory,
        detections_provider: Callable[[], list],
        entities_provider: Callable[[], list],
        audio_provider: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self._interval = max(1, int(config.llm.analysis_interval))

        self._event_bus = event_bus
        self._camera = camera
        self._llm_client = llm_client
        self._prompt_manager = prompt_manager
        self._parser = parser
        self._memory = memory

        self._detections_provider = detections_provider
        self._entities_provider = entities_provider
        self._audio_provider = audio_provider

        self._running = False
        self._active_cycle_task: Optional[asyncio.Task[None]] = None

    async def analysis_loop(self) -> None:
        """Boucle principale d'analyse toutes les N secondes.

        Si une analyse est deja en cours, le cycle est saute.
        """
        self._running = True
        logger.info("Orchestrateur cognitif demarre (interval=%ss)", self._interval)

        while self._running:
            if self._active_cycle_task and not self._active_cycle_task.done():
                logger.debug("Cycle cognitif skip: analyse precedente encore en cours")
            else:
                self._active_cycle_task = asyncio.create_task(self._run_cycle())

            await asyncio.sleep(self._interval)

        if self._active_cycle_task:
            await asyncio.gather(self._active_cycle_task, return_exceptions=True)

        logger.info("Orchestrateur cognitif arrete")

    def stop(self) -> None:
        """Demande l'arret propre de la boucle d'analyse."""
        self._running = False

    async def _run_cycle(self) -> None:
        """Execute un cycle d'analyse complet."""
        snapshot = self._camera.get_snapshot() if hasattr(self._camera, "get_snapshot") else None
        if snapshot is None:
            logger.debug("Cycle cognitif ignore: aucun snapshot camera")
            return

        detections = self._detections_provider()
        entities = self._entities_provider()
        audio_transcript = self._audio_provider() if self._audio_provider else None
        previous_context = self._memory.get_context_summary()

        prompt = self._prompt_manager.build_analysis_prompt(
            detections=detections,
            tracked_entities=entities,
            audio_transcript=audio_transcript,
            previous_context=previous_context,
        )

        t0 = time.perf_counter()
        try:
            raw_response = await self._llm_client.generate(prompt, snapshot)
        except LLMClientError as exc:
            logger.error("Echec appel LLM: %s", exc)
            fallback = ActionResponse(
                action_vocale="Mode veille active. LLM indisponible.",
                niveau_alerte="normal",
                outils_a_lancer=[],
                raw_thinking=None,
                parse_success=False,
                parse_errors=[str(exc)],
                response_time_ms=(time.perf_counter() - t0) * 1000,
            )
            self._event_bus.emit("llm_response", self._to_event_payload(fallback))
            return

        elapsed_ms = (time.perf_counter() - t0) * 1000
        parsed = self._parser.parse(raw_response, response_time_ms=elapsed_ms)

        self._memory.add_exchange(
            context=(audio_transcript or "sans audio"),
            response=parsed.action_vocale,
            alert_level=parsed.niveau_alerte,
        )

        self._event_bus.emit("llm_response", self._to_event_payload(parsed))

    @staticmethod
    def _to_event_payload(response: ActionResponse) -> dict:
        """Convertit ActionResponse en payload serialisable EventBus."""
        return {
            "action_vocale": response.action_vocale,
            "niveau_alerte": response.niveau_alerte,
            "outils_a_lancer": [
                {"tool_name": t.tool_name, "parametres": t.parametres}
                for t in response.outils_a_lancer
            ],
            "raw_thinking": response.raw_thinking,
            "parse_success": response.parse_success,
            "parse_errors": response.parse_errors,
            "response_time_ms": response.response_time_ms,
        }
