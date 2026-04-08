#!/usr/bin/env python3
"""Demo du pipeline avec reconnaissance faciale et tracking."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.core.camera import Camera
from src.core.config import Config
from src.core.detector import ObjectDetector
from src.core.face_manager import FaceManager
from src.core.tracker import Tracker
from src.core.visualizer import Visualizer
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(description="Demo reconnaissance faciale")
    parser.add_argument("--source", type=str, default=None, help="Source video")
    parser.add_argument("--model", type=str, default=None, help="Modele YOLO")
    parser.add_argument("--headless", action="store_true", help="Sans fenetre")
    return parser.parse_args()


def main() -> int:
    """Point d'entree de la demo reconnaissance faciale."""
    args = parse_args()

    setup_logging(level="INFO", file_enabled=False)
    logger = get_logger("demo_face")

    config = Config()
    if args.source is not None:
        config.camera.source = args.source
    if args.model is not None:
        config.detection.model_path = args.model

    bus = EventBus()
    bus.subscribe("face_recognized", lambda d: logger.info(
        "Reconnu: %s (%.2f) track=%s", d.get("name"), d.get("confidence"), d.get("track_id")
    ))
    bus.subscribe("face_unknown", lambda d: logger.warning(
        "Inconnu track=%s snapshot=%s", d.get("track_id"), d.get("snapshot_path")
    ))
    bus.subscribe("face_uncertain", lambda d: logger.info(
        "Incertain track=%s best=%s conf=%.2f",
        d.get("track_id"), d.get("best_match"), d.get("confidence", 0.0)
    ))

    detector = ObjectDetector(config, bus)
    tracker = Tracker(config, bus)
    face_manager = FaceManager(config, bus)
    visualizer = Visualizer(show_hud=True)
    camera = Camera(config, bus)

    camera.start()
    if not camera.is_connected:
        logger.error("Impossible de connecter la camera.")
        return 1

    logger.info("Pipeline lance. Appuyez sur q pour quitter.")

    frame_count = 0
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            detections = detector.detect(frame, frame_id=camera.frame_id)
            tracked = tracker.update(detections)
            tracked = tracker.apply_face_recognition(frame, face_manager, camera.frame_id)

            annotated = visualizer.draw(
                frame,
                entities=tracked,
                fps=camera.metrics.fps_current,
                extra_info={"known_embeddings": face_manager.known_embeddings_count},
            )

            if not args.headless:
                cv2.imshow("Sentinel-AI - Face Recognition", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            else:
                if frame_count >= 150:
                    break

    except KeyboardInterrupt:
        logger.info("Arret demande.")
    finally:
        camera.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
