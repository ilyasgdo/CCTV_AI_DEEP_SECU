#!/usr/bin/env python3
"""Demo du pipeline cognitif asynchrone (Etape 4)."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cognitive.conversation_memory import ConversationMemory
from src.cognitive.llm_client import LLMClient
from src.cognitive.orchestrator import AnalysisOrchestrator
from src.cognitive.prompt_manager import PromptManager
from src.cognitive.response_parser import ResponseParser
from src.core.camera import Camera
from src.core.config import Config
from src.core.detector import ObjectDetector
from src.core.face_manager import FaceManager
from src.core.tracker import Tracker
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI de la demo."""
    parser = argparse.ArgumentParser(description="Demo pipeline cognitif")
    parser.add_argument("--source", type=str, default=None, help="Source camera")
    parser.add_argument("--duration", type=int, default=20, help="Duree demo en secondes")
    return parser.parse_args()


async def main_async() -> int:
    """Point d'entree asynchrone de la demo cognitive."""
    args = parse_args()

    setup_logging(level="INFO", file_enabled=False)
    logger = get_logger("demo_cognitive")

    config = Config()
    if args.source is not None:
        config.camera.source = args.source

    bus = EventBus()
    bus.subscribe(
        "llm_response",
        lambda d: logger.info(
            "LLM -> niveau=%s vocal='%s' parse=%s",
            d.get("niveau_alerte"),
            d.get("action_vocale"),
            d.get("parse_success"),
        ),
    )

    camera = Camera(config, bus)
    detector = ObjectDetector(config, bus)
    tracker = Tracker(config, bus)
    face_manager = FaceManager(config, bus)

    llm_client = LLMClient(config)
    prompt_manager = PromptManager(config)
    parser = ResponseParser()
    memory = ConversationMemory(max_entries=10)

    latest_detections = []

    def detections_provider() -> list:
        return latest_detections

    def entities_provider() -> list:
        return tracker.active_entities

    orchestrator = AnalysisOrchestrator(
        config=config,
        event_bus=bus,
        camera=camera,
        llm_client=llm_client,
        prompt_manager=prompt_manager,
        parser=parser,
        memory=memory,
        detections_provider=detections_provider,
        entities_provider=entities_provider,
        audio_provider=lambda: None,
    )

    camera.start()
    if not camera.is_connected:
        logger.error("Impossible de connecter la camera")
        return 1

    logger.info("Demo cognitive lancee pour %ss", args.duration)

    loop_task = asyncio.create_task(orchestrator.analysis_loop())
    start = time.time()

    try:
        while time.time() - start < args.duration:
            frame = camera.get_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            latest_detections = detector.detect(frame, frame_id=camera.frame_id)
            tracked = tracker.update(latest_detections)
            tracker.apply_face_recognition(frame, face_manager, camera.frame_id)

            logger.debug(
                "Scene: detections=%s tracked=%s", len(latest_detections), len(tracked)
            )

            await asyncio.sleep(0.02)

    finally:
        orchestrator.stop()
        await asyncio.sleep(config.llm.analysis_interval + 0.2)
        if not loop_task.done():
            loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)
        camera.stop()

    return 0


def main() -> int:
    """Point d'entree synchrone."""
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
