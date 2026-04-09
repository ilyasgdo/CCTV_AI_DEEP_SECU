"""Orchestrateur asynchrone du cycle cognitif Sentinel-AI."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Optional

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

        legacy_narration = os.getenv("SENTINEL_DEBUG_SCENE_NARRATION")
        default_narration = legacy_narration if legacy_narration is not None else "true"
        self._scene_narration_enabled = os.getenv(
            "SENTINEL_SCENE_NARRATION", default_narration
        ).lower() in {"1", "true", "yes"}
        try:
            self._snapshot_quality = int(
                os.getenv("SENTINEL_LLM_SNAPSHOT_QUALITY", "55")
            )
        except ValueError:
            self._snapshot_quality = 55
        self._snapshot_quality = max(20, min(95, self._snapshot_quality))

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
        snapshot = None
        if hasattr(self._camera, "get_snapshot"):
            try:
                snapshot = self._camera.get_snapshot(quality=self._snapshot_quality)
            except TypeError:
                snapshot = self._camera.get_snapshot()
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

        logger.info(
            "Cycle IA: envoi snapshot=%dKB q=%d detections=%d entites=%d audio=%s",
            max(1, len(snapshot) // 1024),
            self._snapshot_quality,
            len(detections),
            len(entities),
            "oui" if audio_transcript else "non",
        )

        scene_facts = self._compute_scene_facts(detections, entities)
        scene_summary = self._build_scene_summary(scene_facts)

        t0 = time.perf_counter()
        try:
            raw_response = await self._llm_client.generate(prompt, snapshot)
        except LLMClientError as exc:
            logger.error("Echec appel LLM: %s", exc)
            fallback_action = "Mode veille active. LLM indisponible."
            if self._scene_narration_enabled:
                fallback_action = f"{fallback_action} Ce que je vois: {scene_summary}."
            fallback = ActionResponse(
                action_vocale=fallback_action,
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

        parsed = self._apply_scene_guardrails(parsed, scene_facts, scene_summary)

        logger.info(
            "Cycle IA: reponse alerte=%s tools=%d latence=%.0fms",
            parsed.niveau_alerte,
            len(parsed.outils_a_lancer),
            elapsed_ms,
        )

        self._memory.add_exchange(
            context=(audio_transcript or "sans audio"),
            response=parsed.action_vocale,
            alert_level=parsed.niveau_alerte,
        )

        self._event_bus.emit("llm_response", self._to_event_payload(parsed))

    @staticmethod
    def _compute_scene_facts(detections: list, entities: list) -> dict[str, Any]:
        """Extrait des faits de scene stables a partir des capteurs."""
        person_count = 0
        object_count = 0
        object_labels: dict[str, int] = {}
        danger_labels: list[str] = []
        suspicious_labels: list[str] = []

        danger_keywords = {"knife", "gun", "weapon", "fire", "smoke", "flame", "explosion"}
        suspicious_keywords = {"backpack", "suitcase", "handbag"}

        for det in detections:
            class_name = str(getattr(det, "class_name", "objet") or "objet")
            normalized_name = class_name.lower().strip().replace("_", " ")
            class_id = getattr(det, "class_id", None)
            if class_id == 0 or normalized_name == "person":
                person_count += 1
            else:
                object_count += 1
                object_labels[class_name] = object_labels.get(class_name, 0) + 1
                if any(keyword in normalized_name for keyword in danger_keywords):
                    danger_labels.append(class_name)
                elif any(keyword in normalized_name for keyword in suspicious_keywords):
                    suspicious_labels.append(class_name)

        known_names: list[str] = []
        unknown_persons = 0
        for entity in entities:
            is_person = bool(getattr(entity, "is_person", False))
            if not is_person:
                continue
            face_status = str(getattr(entity, "face_status", "unknown"))
            if face_status == "known":
                face_name = getattr(entity, "face_name", None)
                if isinstance(face_name, str) and face_name.strip():
                    known_names.append(face_name.strip())
            else:
                unknown_persons += 1

        return {
            "person_count": person_count,
            "object_count": object_count,
            "object_labels": object_labels,
            "known_names": known_names,
            "unknown_persons": unknown_persons,
            "danger_labels": danger_labels,
            "suspicious_labels": suspicious_labels,
        }

    @staticmethod
    def _build_scene_summary(scene_facts: dict[str, Any]) -> str:
        """Construit un resume court de la scene observable."""
        person_count = int(scene_facts.get("person_count", 0))
        object_count = int(scene_facts.get("object_count", 0))
        object_labels = dict(scene_facts.get("object_labels", {}))
        known_names = list(scene_facts.get("known_names", []))
        unknown_persons = int(scene_facts.get("unknown_persons", 0))
        danger_labels = list(scene_facts.get("danger_labels", []))
        suspicious_labels = list(scene_facts.get("suspicious_labels", []))

        if person_count == 0 and object_count == 0:
            return "aucun indice visuel clair"

        parts: list[str] = []
        if person_count > 0:
            parts.append(f"{person_count} personne(s)")
        if object_count > 0:
            parts.append(f"{object_count} objet(s)")

        if object_labels:
            top_objects = sorted(
                object_labels.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
            labels = ", ".join(f"{name}:{count}" for name, count in top_objects)
            parts.append(f"classes {labels}")

        if known_names:
            unique_names = sorted(set(known_names))
            parts.append("connus " + ", ".join(unique_names[:3]))

        if unknown_persons > 0:
            parts.append(f"{unknown_persons} non reconnu(s)")

        if danger_labels:
            labels = ", ".join(sorted(set(danger_labels))[:3])
            parts.append(f"danger possible: {labels}")
        elif suspicious_labels:
            labels = ", ".join(sorted(set(suspicious_labels))[:3])
            parts.append(f"objets a surveiller: {labels}")

        return "; ".join(parts)

    @staticmethod
    def _estimate_scene_alert_level(scene_facts: dict[str, Any]) -> str:
        """Estime un niveau d'alerte minimal a partir des faits capteurs."""
        danger_labels = list(scene_facts.get("danger_labels", []))
        unknown_persons = int(scene_facts.get("unknown_persons", 0))
        suspicious_labels = list(scene_facts.get("suspicious_labels", []))

        if danger_labels:
            return "critique"
        if unknown_persons >= 2:
            return "alerte"
        if unknown_persons >= 1 or suspicious_labels:
            return "attention"
        return "normal"

    def _apply_scene_guardrails(
        self,
        parsed: ActionResponse,
        scene_facts: dict[str, Any],
        scene_summary: str,
    ) -> ActionResponse:
        """Ajoute des garde-fous pour garantir narration et vigilance minimales."""
        alert_rank = {"normal": 0, "attention": 1, "alerte": 2, "critique": 3}

        min_alert = self._estimate_scene_alert_level(scene_facts)
        if alert_rank.get(min_alert, 0) > alert_rank.get(parsed.niveau_alerte, 0):
            parsed.niveau_alerte = min_alert

        action = str(parsed.action_vocale or "").strip()
        generic_markers = {
            "surveillance silencieuse",
            "veille active",
            "analyse en cours",
            "ras",
            "r.a.s",
        }
        is_generic = any(marker in action.lower() for marker in generic_markers)

        has_scene_signal = (
            int(scene_facts.get("person_count", 0)) > 0
            or int(scene_facts.get("object_count", 0)) > 0
        )

        if is_generic and has_scene_signal:
            action = "Je surveille la zone."

        if self._scene_narration_enabled:
            if action:
                separator = "" if action.endswith((".", "!", "?")) else "."
                action = f"{action}{separator} Ce que je vois: {scene_summary}."
            else:
                action = f"Ce que je vois: {scene_summary}."

        parsed.action_vocale = action.strip()
        return parsed

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
