#!/usr/bin/env python3
"""Script CLI d'enrollement pour la whitelist faciale."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.camera import Camera
from src.core.config import Config
from src.core.face_manager import EnrollmentError, FaceManager
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(description="Enrollement d'une personne")

    parser.add_argument("--name", required=True, help="Nom complet")
    parser.add_argument("--role", default="Visiteur", help="Role de la personne")
    parser.add_argument("--access-level", default="user", help="Niveau d'acces")
    parser.add_argument("--notes", default="", help="Notes libres")

    parser.add_argument(
        "--mode",
        choices=["photo", "camera"],
        required=True,
        help="Mode d'enrollement",
    )

    parser.add_argument(
        "--photos",
        nargs="*",
        default=[],
        help="Liste de 3 a 5 images en mode photo",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Nombre de captures pour le mode camera",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source camera (0, RTSP, fichier video)",
    )

    return parser.parse_args()


def main() -> int:
    """Point d'entree du script d'enrollement."""
    args = parse_args()

    setup_logging(level="INFO", file_enabled=False)
    logger = get_logger("enroll_person")

    config = Config()
    if args.source is not None:
        config.camera.source = args.source

    event_bus = EventBus()
    manager = FaceManager(config=config, event_bus=event_bus)

    try:
        if args.mode == "photo":
            if len(args.photos) < 3 or len(args.photos) > 5:
                logger.error("Le mode photo exige entre 3 et 5 images.")
                return 1

            person = manager.enroll_from_photos(
                name=args.name,
                role=args.role,
                access_level=args.access_level,
                photo_paths=args.photos,
                notes=args.notes,
            )
        else:
            camera = Camera(config, event_bus)
            camera.start()
            if not camera.is_connected:
                logger.error("Impossible de connecter la camera.")
                return 1

            try:
                person = manager.enroll_from_camera(
                    camera=camera,
                    name=args.name,
                    role=args.role,
                    access_level=args.access_level,
                    notes=args.notes,
                    samples=args.samples,
                )
            finally:
                camera.stop()

        logger.info("Enrollement reussi: %s (%s)", person["name"], person["id"])
        logger.info("Embeddings: %d", len(person["embeddings"]))
        return 0

    except EnrollmentError as exc:
        logger.error("Echec enrollement: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
