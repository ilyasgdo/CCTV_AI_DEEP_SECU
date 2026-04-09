#!/usr/bin/env python3
"""Sentinel-AI - point d'entree principal (integration complete etape 7)."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from werkzeug.serving import make_server

from src.audio import STTEngine, TTSEngine
from src.cognitive import (
    AnalysisOrchestrator,
    ConversationMemory,
    LLMClient,
    PromptManager,
    ResponseParser,
)
from src.core.camera import Camera
from src.core.alert_clip_recorder import AlertClipRecorder
from src.core.config import Config
from src.core.detector import Detection, ObjectDetector
from src.core.face_manager import FaceManager
from src.core.surveillance_zones import SurveillanceZoneManager
from src.core.tracker import Tracker, TrackedEntity
from src.core.visualizer import Visualizer
from src.dashboard.app import create_app
from src.effector import ToolExecutor
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger, setup_logging
from src.utils.system_monitor import SystemMonitor

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VERSION = "0.7.0"
APP_NAME = "Sentinel-AI"


@dataclass
class RuntimeOptions:
    """Options runtime du systeme integre."""

    no_audio: bool = False
    no_dashboard: bool = False
    demo_mode: bool = False


class DashboardServer:
    """Serveur dashboard dans un thread, arretable proprement."""

    def __init__(self, config: Config, app: Any) -> None:
        self._config = config
        self._app = app
        self._server: Any = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Demarre le serveur HTTP dashboard en arriere-plan."""
        if self._thread and self._thread.is_alive():
            return

        host = self._config.dashboard.host
        port = self._config.dashboard.port
        self._server = make_server(host, port, self._app, threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Arrete le serveur dashboard."""
        if self._server is not None:
            self._server.shutdown()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)


class SentinelAI:
    """Orchestrateur principal de tous les sous-systemes Sentinel-AI."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        runtime_options: Optional[RuntimeOptions] = None,
    ) -> None:
        self.runtime_options = runtime_options or RuntimeOptions()

        self.config = Config(config_path)
        self.event_bus = EventBus()

        setup_logging(
            level=self.config.logging.level,
            log_dir=self.config.logging.log_dir,
            file_enabled=self.config.logging.file_enabled,
            max_bytes=self.config.logging.max_file_size_mb * 1024 * 1024,
            backup_count=self.config.logging.backup_count,
        )
        self.logger = get_logger("sentinel")

        self.camera = Camera(self.config, self.event_bus)
        self.detector: Optional[ObjectDetector] = None
        self.tracker = Tracker(self.config, self.event_bus)
        self.face_manager: Optional[FaceManager] = None
        self.visualizer: Optional[Visualizer] = None
        self.clip_recorder = AlertClipRecorder(self.config)
        self.zone_manager = SurveillanceZoneManager(self.config)

        self.llm_client = LLMClient(self.config)
        self.prompt_manager = PromptManager(self.config)
        self.response_parser = ResponseParser()
        self.memory = ConversationMemory(max_entries=12)

        self.tts: Optional[TTSEngine] = None
        self.stt: Optional[STTEngine] = None
        self.tool_executor: Optional[ToolExecutor] = None

        self.dashboard_server: Optional[DashboardServer] = None
        self.dashboard_enabled = not self.runtime_options.no_dashboard
        self.monitor = SystemMonitor(
            event_bus=self.event_bus,
            llm_client=self.llm_client,
            fps_provider=self._current_fps,
            person_count_provider=self._current_person_count,
        )

        self._running = threading.Event()
        self._state_lock = threading.Lock()
        self._perception_thread: Optional[threading.Thread] = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._orchestrator_task: Optional[asyncio.Task[None]] = None

        self._frame_id = 0
        self._latest_detections: list[Detection] = []
        self._latest_entities: list[TrackedEntity] = []

        self._degraded_llm = False
        self._degraded_audio = False

        self._init_modules()
        self._wire_events()

    def _init_modules(self) -> None:
        """Initialise les modules en mode nominal ou degrade."""
        try:
            self.detector = ObjectDetector(self.config, self.event_bus)
        except Exception as exc:
            self.logger.error("Detecteur indisponible: %s", exc, exc_info=True)
            self.detector = None

        try:
            self.face_manager = FaceManager(self.config, self.event_bus)
        except Exception as exc:
            self.logger.warning("FaceManager degrade: %s", exc)
            self.face_manager = None

        try:
            self.visualizer = Visualizer(show_skeleton=True, show_hud=False)
        except Exception as exc:
            self.logger.warning("Visualizer indisponible: %s", exc)
            self.visualizer = None

        if not self.runtime_options.no_audio:
            try:
                self.tts = TTSEngine(self.config, self.event_bus)
                self._degraded_audio = False
            except Exception as exc:
                self.logger.warning("TTS indisponible, mode texte uniquement: %s", exc)
                self._degraded_audio = True
                self.tts = None

            try:
                self.stt = STTEngine(self.config, self.event_bus)
            except Exception as exc:
                self.logger.warning("STT indisponible, ecoute vocale desactivee: %s", exc)
                self.stt = None
        else:
            self._degraded_audio = True

        self.tool_executor = ToolExecutor(
            self.config,
            event_bus=self.event_bus,
            camera=self.camera,
            tts_engine=self.tts,
        )

        if self.dashboard_enabled:
            app = create_app(
                config=self.config,
                event_bus=self.event_bus,
                camera=self.camera,
                tracker=self.tracker,
                visualizer=self.visualizer,
                llm_client=self.llm_client,
                monitor=self.monitor,
                zone_manager=self.zone_manager,
            )
            self.dashboard_server = DashboardServer(self.config, app)

    def _current_fps(self) -> float:
        """Retourne le FPS courant camera pour le monitoring."""
        if hasattr(self.camera, "metrics") and hasattr(self.camera.metrics, "fps_current"):
            return float(getattr(self.camera.metrics, "fps_current", 0.0))
        return 0.0

    def _current_person_count(self) -> int:
        """Retourne le nombre de personnes trackees."""
        if hasattr(self.tracker, "person_count"):
            return int(self.tracker.person_count)
        return 0

    def _wire_events(self) -> None:
        """Connecte les handlers d'evenements inter-modules."""
        self.event_bus.subscribe("llm_response", self._on_llm_response)
        self.event_bus.subscribe("camera_disconnected", self._on_camera_disconnected)
        self.event_bus.subscribe("face_unknown", self._on_alert_event)
        self.event_bus.subscribe("entity_lingering", self._on_alert_event)
        self.event_bus.subscribe("zone_intrusion", self._on_alert_event)

    def _on_camera_disconnected(self, data: dict[str, Any]) -> None:
        """Reaction au signal camera indisponible."""
        self.logger.warning("Camera deconnectee: %s", data)

    def _on_llm_response(self, data: dict[str, Any]) -> None:
        """Planifie l'execution TTS + outils a reception LLM."""
        loop = self._async_loop
        if loop is None or loop.is_closed():
            return

        if str(data.get("niveau_alerte", "normal")) in {"alerte", "critique"}:
            self.clip_recorder.on_alert("llm_alert")

        future = asyncio.run_coroutine_threadsafe(self._execute_actions(data), loop)
        future.add_done_callback(self._log_action_execution_result)

    def _log_action_execution_result(self, future: concurrent.futures.Future[Any]) -> None:
        """Journalise toute erreur sur l'execution asynchrone des actions LLM."""
        try:
            error = future.exception()
        except concurrent.futures.CancelledError:
            return
        except Exception as exc:  # pragma: no cover - garde-fou
            self.logger.error("Impossible de lire le resultat des actions LLM: %s", exc)
            return

        if error is None:
            return

        self.logger.error(
            "Erreur execution actions LLM: %s",
            error,
            exc_info=(type(error), error, error.__traceback__),
        )

    def _on_alert_event(self, _: dict[str, Any]) -> None:
        """Declenche l'enregistrement de clip sur evenement alerte."""
        self.clip_recorder.on_alert("intrusion")

    async def _execute_actions(self, payload: dict[str, Any]) -> None:
        """Execute la phase 6.7 (TTS + tools + event log)."""
        action_vocale = str(payload.get("action_vocale") or "").strip()
        tools = payload.get("outils_a_lancer") or []

        if action_vocale:
            self.logger.info("Action IA: %s", action_vocale)

        if action_vocale and self.tts is not None and not self._degraded_audio:
            await self.tts.speak(action_vocale)
            self.logger.info("Action vocale envoyee au moteur TTS")
        elif action_vocale:
            self.logger.info("Action vocale (mode texte): %s", action_vocale)

        if self.tool_executor is not None:
            await self.tool_executor.execute(tools)

    def _get_last_audio_transcript(self) -> Optional[str]:
        """Retourne la derniere transcription STT."""
        if self.stt is None:
            return None
        return self.stt.get_transcript()

    def _get_latest_detections(self) -> list[Detection]:
        """Provider detections pour l'orchestrateur."""
        with self._state_lock:
            return list(self._latest_detections)

    def _get_latest_entities(self) -> list[TrackedEntity]:
        """Provider entites pour l'orchestrateur."""
        with self._state_lock:
            return list(self._latest_entities)

    def _perception_loop(self) -> None:
        """Boucle continue perception CPU: capture + detection + tracking + face."""
        target_fps = max(1, int(self.config.camera.fps))
        sleep_time = 1.0 / float(target_fps)

        while self._running.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            self.clip_recorder.add_frame(frame)

            self._frame_id += 1
            detections = self.detector.detect(frame, self._frame_id) if self.detector else []
            entities = self.tracker.update(detections)

            if self.face_manager is not None:
                entities = self.tracker.apply_face_recognition(
                    frame=frame,
                    face_manager=self.face_manager,
                    frame_id=self._frame_id,
                )

            with self._state_lock:
                self._latest_detections = detections
                self._latest_entities = entities

            intrusions = self.zone_manager.check_intrusions(entities)
            for hit in intrusions:
                self.event_bus.emit("zone_intrusion", hit)

            time.sleep(sleep_time)

    async def start(self) -> None:
        """Demarre tous les sous-systemes dans le bon ordre."""
        self.logger.info("Demarrage Sentinel-AI integration complete")

        self._async_loop = asyncio.get_running_loop()
        self._running.set()

        self.camera.start()

        self._perception_thread = threading.Thread(target=self._perception_loop, daemon=True)
        self._perception_thread.start()

        llm_ok = await self.llm_client.health_check()
        self._degraded_llm = not llm_ok
        if not llm_ok:
            self.logger.warning("LLM indisponible: surveillance silencieuse active")

        if self.stt is not None:
            self.stt.start_listening()

        if self.dashboard_server is not None:
            self.dashboard_server.start()
            self.logger.info(
                "Dashboard disponible sur http://%s:%s",
                self.config.dashboard.host,
                self.config.dashboard.port,
            )

        if not self._degraded_llm:
            orchestrator = AnalysisOrchestrator(
                config=self.config,
                event_bus=self.event_bus,
                camera=self.camera,
                llm_client=self.llm_client,
                prompt_manager=self.prompt_manager,
                parser=self.response_parser,
                memory=self.memory,
                detections_provider=self._get_latest_detections,
                entities_provider=self._get_latest_entities,
                audio_provider=self._get_last_audio_transcript,
            )
            self._orchestrator_task = asyncio.create_task(orchestrator.analysis_loop())
            self._orchestrator = orchestrator
        else:
            self._orchestrator = None

    async def run_forever(self) -> None:
        """Lance le systeme et le maintient actif jusqu'a interruption."""
        await self.start()
        try:
            while self._running.is_set():
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Arret propre de tous les sous-systemes."""
        if not self._running.is_set():
            return

        self.logger.info("Arret Sentinel-AI en cours")
        self._running.clear()

        if getattr(self, "_orchestrator", None) is not None:
            self._orchestrator.stop()
        if self._orchestrator_task is not None:
            await asyncio.gather(self._orchestrator_task, return_exceptions=True)

        if self.stt is not None:
            self.stt.stop_listening()

        if self.tts is not None:
            await self.tts.speak("Systeme Sentinel en cours d'arret. Au revoir.")
            await self.tts.shutdown()

        if self.dashboard_server is not None:
            self.dashboard_server.stop()

        self.camera.stop()

        if self._perception_thread and self._perception_thread.is_alive():
            self._perception_thread.join(timeout=3)

        self._save_state()
        self.logger.info("Sentinel-AI arrete proprement")

    def _save_state(self) -> None:
        """Sauvegarde un etat runtime minimal."""
        out = self.config.project_root / "data" / "reports" / "runtime_state.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        with self._state_lock:
            payload = {
                "timestamp": int(time.time()),
                "frame_id": self._frame_id,
                "tracked_entities": len(self._latest_entities),
                "detections": len(self._latest_detections),
                "degraded": {
                    "llm": self._degraded_llm,
                    "audio": self._degraded_audio,
                },
                "monitoring": self.monitor.get_metrics(),
                "zones": len(self.zone_manager.list_zones()),
            }

        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def check_dependency(module_name: str, display_name: str, required: bool = True) -> bool:
    """Verifie la presence d'une dependance Python."""
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "?")
        marker = "OK"
        print(f"  [{marker}] {display_name:<24s} v{version}")
        return True
    except ImportError:
        marker = "KO" if required else "OPT"
        print(f"  [{marker}] {display_name:<24s}")
        return not required


def check_all_dependencies() -> bool:
    """Verifie les dependances principales du projet."""
    print("\nVerification des dependances Sentinel-AI\n")
    required = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("requests", "Requests"),
        ("flask", "Flask"),
    ]
    optional = [
        ("ultralytics", "Ultralytics"),
        ("insightface", "InsightFace"),
        ("edge_tts", "edge-tts"),
        ("faster_whisper", "faster-whisper"),
        ("flask_socketio", "Flask-SocketIO"),
    ]

    ok = all(check_dependency(m, n, required=True) for m, n in required)
    for module_name, display_name in optional:
        check_dependency(module_name, display_name, required=False)
    return ok


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI du point d'entree principal."""
    parser = argparse.ArgumentParser(description="Sentinel-AI integration complete")
    parser.add_argument("--check", action="store_true", help="Verifier les dependances")
    parser.add_argument("--version", action="store_true", help="Afficher la version")
    parser.add_argument("--config", type=str, default=None, help="Chemin config YAML")
    parser.add_argument("--no-audio", action="store_true", help="Desactiver audio")
    parser.add_argument("--no-dashboard", action="store_true", help="Desactiver dashboard")
    parser.add_argument("--camera", type=str, default=None, help="Override camera.source")
    parser.add_argument("--llm-url", type=str, default=None, help="Override llm.api_url")
    parser.add_argument("--demo", action="store_true", help="Mode demo")
    return parser.parse_args()


async def _run_from_args(args: argparse.Namespace) -> int:
    """Execute l'application avec les options CLI."""
    if args.version:
        print(f"{APP_NAME} v{VERSION}")
        return 0

    if args.check:
        return 0 if check_all_dependencies() else 1

    if not check_all_dependencies():
        return 1

    options = RuntimeOptions(
        no_audio=bool(args.no_audio),
        no_dashboard=bool(args.no_dashboard),
        demo_mode=bool(args.demo),
    )
    app = SentinelAI(config_path=args.config, runtime_options=options)

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
    """Point d'entree synchrone pour execution standard."""
    args = parse_args()
    return asyncio.run(_run_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
