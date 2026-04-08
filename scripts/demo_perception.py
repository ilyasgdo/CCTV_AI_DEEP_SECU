#!/usr/bin/env python3
"""
Demo script — Pipeline de Perception Visuelle.

Démontre le fonctionnement complet du pipeline :
Caméra → Détection YOLO → Tracking → Visualisation.

Usage:
    python scripts/demo_perception.py
    python scripts/demo_perception.py --source 0
    python scripts/demo_perception.py --source video.mp4
    python scripts/demo_perception.py --model yolo26n-pose.pt
"""

import argparse
import sys
import time
from pathlib import Path

# Ajouter la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.core.config import Config
from src.core.camera import Camera
from src.core.detector import ObjectDetector
from src.core.tracker import Tracker
from src.core.visualizer import Visualizer
from src.utils.event_bus import EventBus
from src.utils.logger import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Demo Pipeline de Perception Visuelle"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Source vidéo (0=webcam, URL, fichier)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Modèle YOLO (ex: yolo26n.pt, yolo26n-pose.pt)",
    )
    parser.add_argument(
        "--confidence", type=float, default=None,
        help="Seuil de confiance YOLO (0.0-1.0)",
    )
    parser.add_argument(
        "--no-skeleton", action="store_true",
        help="Désactiver l'affichage des squelettes",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Mode sans fenêtre (pour les tests)",
    )
    return parser.parse_args()


def main() -> int:
    """Point d'entrée de la demo perception."""
    args = parse_args()

    # Setup
    setup_logging(level="INFO", file_enabled=False)
    logger = get_logger("demo_perception")

    logger.info("=" * 50)
    logger.info("🔍 Demo Pipeline de Perception Visuelle")
    logger.info("=" * 50)

    # Configuration
    config = Config()

    # Overrides CLI
    if args.source is not None:
        config.camera.source = args.source
    if args.model is not None:
        config.detection.model_path = args.model
    if args.confidence is not None:
        config.detection.confidence = args.confidence

    # Event Bus
    event_bus = EventBus()

    # Listeners pour le debug
    event_bus.subscribe("entity_appeared", lambda d: logger.info(
        f"🆕 Entité #{d['track_id']} apparue: "
        f"{d['class']} (conf={d['confidence']:.2f})"
    ))
    event_bus.subscribe("entity_disappeared", lambda d: logger.info(
        f"👋 Entité #{d['track_id']} disparue "
        f"(durée={d['duration']:.1f}s)"
    ))
    event_bus.subscribe("person_detected", lambda d: logger.debug(
        f"👤 {d['count']} personne(s) détectée(s)"
    ))

    # ── Initialiser les modules ──
    logger.info(
        f"📷 Source: {config.camera.source}"
    )
    logger.info(
        f"🔍 Modèle: {config.detection.model_path}"
    )

    try:
        # Détecteur YOLO
        detector = ObjectDetector(config, event_bus)
        logger.info(f"✅ Détecteur: {detector}")

        # Tracker
        tracker = Tracker(config, event_bus)
        logger.info(f"✅ Tracker: {tracker}")

        # Visualiseur
        visualizer = Visualizer(
            show_skeleton=not args.no_skeleton,
            show_hud=True,
        )
        logger.info(f"✅ Visualiseur: {visualizer}")

        # Caméra
        camera = Camera(config, event_bus)
        camera.start()

        if not camera.is_connected:
            logger.error("❌ Impossible de connecter la caméra")
            return 1

        logger.info("✅ Caméra connectée")
        logger.info("")
        logger.info("🎬 Pipeline démarré — Appuyez sur 'q' pour quitter")
        logger.info("")

    except Exception as e:
        logger.error(f"❌ Erreur d'initialisation: {e}")
        return 1

    # ── Boucle principale ──
    frame_count = 0
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            # Détection
            if detector.is_pose_model:
                detections = detector.detect_persons(
                    frame, frame_id=camera.frame_id
                )
            else:
                detections = detector.detect(
                    frame, frame_id=camera.frame_id
                )

            # Tracking
            tracked = tracker.update(detections)

            # Visualisation
            annotated = visualizer.draw(
                frame,
                entities=tracked,
                fps=camera.metrics.fps_current,
                extra_info={
                    "status": "LIVE",
                },
            )

            # Affichage
            if not args.headless:
                cv2.imshow("Sentinel-AI — Perception Demo", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' ou Escape
                    break
            else:
                # Mode headless : tourner pendant 5 secondes
                if frame_count >= 150:  # ~5s @ 30fps
                    break

            # Log périodique
            if frame_count % 150 == 0:
                logger.info(
                    f"📊 Frame #{frame_count} | "
                    f"FPS={camera.metrics.fps_current:.1f} | "
                    f"Personnes={tracker.person_count} | "
                    f"Entités={tracker.active_count} | "
                    f"Infer={detector.metrics.avg_inference_ms:.0f}ms"
                )

    except KeyboardInterrupt:
        logger.info("\n🛑 Arrêt demandé.")

    finally:
        camera.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    # Stats finales
    logger.info("")
    logger.info("=" * 50)
    logger.info("📊 Statistiques finales")
    logger.info("=" * 50)
    logger.info(f"  Frames traitées: {frame_count}")
    logger.info(f"  FPS moyen: {camera.metrics.fps_average:.1f}")
    logger.info(f"  Détections total: {detector.metrics.total_detections}")
    logger.info(f"  Personnes total: {detector.metrics.total_persons}")
    logger.info(f"  Inference moy: {detector.metrics.avg_inference_ms:.1f}ms")
    logger.info(f"  Entités apparues: {tracker.total_appeared}")
    logger.info(f"  Entités disparues: {tracker.total_disappeared}")
    logger.info(f"  Events émis: {event_bus.total_events}")
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
