#!/usr/bin/env python3
"""Lanceur CLI Sentinel-AI (etape 7)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import RuntimeOptions, SentinelAI, check_all_dependencies


def parse_args() -> argparse.Namespace:
    """Parse les arguments du script de lancement."""
    parser = argparse.ArgumentParser(description="Sentinel-AI launcher")
    parser.add_argument("--config", type=str, default=None, help="Fichier de configuration")
    parser.add_argument("--no-audio", action="store_true", help="Desactiver STT/TTS")
    parser.add_argument("--no-dashboard", action="store_true", help="Desactiver dashboard")
    parser.add_argument("--camera", type=str, default=None, help="Source camera")
    parser.add_argument("--llm-url", type=str, default=None, help="URL du LLM")
    parser.add_argument("--check", action="store_true", help="Verifier les dependances")
    parser.add_argument("--demo", action="store_true", help="Mode demo")
    return parser.parse_args()


async def _run() -> int:
    args = parse_args()

    if args.check:
        return 0 if check_all_dependencies() else 1

    if not check_all_dependencies():
        return 1

    runtime_options = RuntimeOptions(
        no_audio=bool(args.no_audio),
        no_dashboard=bool(args.no_dashboard),
        demo_mode=bool(args.demo),
    )
    app = SentinelAI(config_path=args.config, runtime_options=runtime_options)

    if args.camera:
        app.config.camera.source = args.camera
    if args.llm_url:
        app.config.llm.api_url = args.llm_url

    try:
        await app.run_forever()
        return 0
    except KeyboardInterrupt:
        await app.shutdown()
        return 0


def main() -> int:
    """Point d'entree synchrone du lanceur CLI."""
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
