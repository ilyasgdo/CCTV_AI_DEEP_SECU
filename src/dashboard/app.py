"""Serveur Flask du dashboard web Sentinel-AI (Etape 6)."""

from __future__ import annotations

import asyncio
import base64
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify, render_template, request

try:
    from flask_socketio import SocketIO  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    SocketIO = None  # type: ignore[assignment]

from src.cognitive.llm_client import LLMClient
from src.core.config import Config
from src.core.face_manager import WhitelistRepository
from src.core.report_generator import ReportGenerator
from src.core.surveillance_zones import SurveillanceZoneManager
from src.utils.system_monitor import SystemMonitor
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger

logger = get_logger(__name__)

EVENT_LOG_FILE = "data/event_log.json"
SNAPSHOTS_DIR = "data/snapshots"
SETTINGS_FILE = "config/settings.yaml"


@dataclass
class DashboardState:
    """Etat partage entre les routes du dashboard."""

    config: Config
    event_bus: Optional[EventBus]
    camera: Optional[object]
    tracker: Optional[object]
    visualizer: Optional[object]
    llm_client: Optional[LLMClient]
    whitelist_repo: WhitelistRepository
    monitor: Optional[SystemMonitor]
    zone_manager: Optional[SurveillanceZoneManager]
    report_generator: ReportGenerator
    last_ai_action: str = "Aucune action."
    last_ai_alert_level: str = "normal"
    last_ai_response_ms: float = 0.0


def _resolve_path(config: Config, value: str) -> Path:
    """Resolve un chemin relatif a la racine du projet."""
    path = Path(value)
    if path.is_absolute():
        return path
    return config.project_root / path


def _default_placeholder_frame() -> np.ndarray:
    """Construit une frame de secours pour le stream MJPEG."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (18, 18, 26)
    cv2.putText(
        frame,
        "Sentinel-AI Dashboard Stream",
        (60, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 212, 255),
        2,
    )
    cv2.putText(
        frame,
        "Camera indisponible - placeholder actif",
        (60, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (220, 220, 220),
        2,
    )
    return frame


def _load_event_log(path: Path) -> list[dict[str, Any]]:
    """Charge le journal d'evenements JSON."""
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.error("Impossible de lire event_log: %s", exc, exc_info=True)
    return []


def _system_metrics() -> dict[str, Any]:
    """Retourne des metriques systeme de base."""
    cpu = 0.0
    ram = 0.0
    try:
        import psutil  # type: ignore[import-not-found]

        cpu = float(psutil.cpu_percent(interval=0.0))
        ram = float(psutil.virtual_memory().percent)
    except Exception:
        pass

    return {"cpu_percent": cpu, "ram_percent": ram}


def create_app(
    config: Optional[Config] = None,
    event_bus: Optional[EventBus] = None,
    camera: Optional[object] = None,
    tracker: Optional[object] = None,
    visualizer: Optional[object] = None,
    llm_client: Optional[LLMClient] = None,
    monitor: Optional[SystemMonitor] = None,
    zone_manager: Optional[SurveillanceZoneManager] = None,
) -> Flask:
    """Construit et configure l'application Flask du dashboard.

    Args:
        config: Configuration Sentinel-AI.
        event_bus: Event bus global.
        camera: Module camera.
        tracker: Module tracker.
        visualizer: Module visualizer.
        llm_client: Client LLM optionnel.

    Returns:
        Application Flask configuree.
    """
    cfg = config or Config()

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    socketio: Any = None
    if SocketIO is not None:
        socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
        app.extensions["socketio"] = socketio

    whitelist_dir = _resolve_path(cfg, cfg.face.whitelist_dir)
    state = DashboardState(
        config=cfg,
        event_bus=event_bus,
        camera=camera,
        tracker=tracker,
        visualizer=visualizer,
        llm_client=llm_client,
        whitelist_repo=WhitelistRepository(whitelist_dir),
        monitor=monitor,
        zone_manager=zone_manager or SurveillanceZoneManager(cfg),
        report_generator=ReportGenerator(cfg),
    )
    app.config["dashboard_state"] = state
    state_lock = threading.Lock()

    if state.event_bus is not None:
        def _remember_llm_response(payload: dict[str, Any]) -> None:
            action = str(payload.get("action_vocale") or "").strip()
            if not action:
                return
            alert = str(payload.get("niveau_alerte") or "normal")
            response_ms = payload.get("response_time_ms", 0.0)
            try:
                response_ms_value = float(response_ms)
            except (TypeError, ValueError):
                response_ms_value = 0.0

            with state_lock:
                state.last_ai_action = action
                state.last_ai_alert_level = alert
                state.last_ai_response_ms = response_ms_value

        state.event_bus.subscribe("llm_response", _remember_llm_response)

    auth_enabled = os.getenv("DASHBOARD_AUTH_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    disable_background = os.getenv("DASHBOARD_DISABLE_BACKGROUND", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    auth_user = os.getenv("DASHBOARD_USERNAME", "admin")
    auth_pass = os.getenv("DASHBOARD_PASSWORD", "sentinel")
    rate_limit_enabled = os.getenv("DASHBOARD_RATE_LIMIT_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    rate_limit_max = max(5, int(os.getenv("DASHBOARD_RATE_LIMIT_PER_MIN", "120")))
    request_history: dict[str, list[float]] = {}
    external_api_key = os.getenv("SENTINEL_EXTERNAL_API_KEY", "").strip()

    def _auth_ok() -> bool:
        if not auth_enabled:
            return True

        auth = request.authorization
        if auth and auth.username == auth_user and auth.password == auth_pass:
            return True

        header_pw = request.headers.get("X-Dashboard-Password")
        if header_pw and header_pw == auth_pass:
            return True

        return False

    def _unauthorized() -> Response:
        return Response(
            "Authentication required",
            401,
            {"WWW-Authenticate": 'Basic realm="Sentinel Dashboard"'},
        )

    def _validate_whitelist_payload(payload: Any) -> tuple[bool, str]:
        """Valide le payload whitelist avant ecriture."""
        if not isinstance(payload, dict):
            return False, "Invalid payload"

        name = payload.get("name", "Unknown")
        role = payload.get("role", "Visitor")
        access_level = payload.get("access_level", "user")
        notes = payload.get("notes", "")

        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            return False, "Invalid field: name"
        if not isinstance(role, str) or len(role) > 80:
            return False, "Invalid field: role"
        if access_level not in {"guest", "user", "admin", "staff", "security"}:
            return False, "Invalid field: access_level"
        if not isinstance(notes, str) or len(notes) > 400:
            return False, "Invalid field: notes"

        return True, ""

    @app.before_request
    def _access_log_and_rate_limit() -> Optional[Response]:
        """Journalise les acces dashboard et applique un rate limit API."""
        logger.info("dashboard_access method=%s path=%s remote=%s",
                    request.method, request.path, request.remote_addr)

        if not rate_limit_enabled:
            return None
        if not request.path.startswith("/api/"):
            return None

        now = time.time()
        key = f"{request.remote_addr or 'unknown'}:{request.path}"
        timestamps = [ts for ts in request_history.get(key, []) if now - ts < 60.0]
        if len(timestamps) >= rate_limit_max:
            return jsonify({"error": "Rate limit exceeded"}), 429

        timestamps.append(now)
        request_history[key] = timestamps
        return None

    def require_auth(func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorateur d'authentification basique."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _auth_ok():
                return _unauthorized()
            return func(*args, **kwargs)

        return wrapper

    def require_external_api_key(func: Callable[..., Any]) -> Callable[..., Any]:
        """Protege les endpoints API v1 via cle simple."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not external_api_key:
                return jsonify({"error": "External API key not configured"}), 503
            key = request.headers.get("X-API-Key", "")
            if key != external_api_key:
                return jsonify({"error": "Unauthorized"}), 401
            return func(*args, **kwargs)

        return wrapper

    @app.route("/")
    @require_auth
    def live_view() -> str:
        """Page principale live view."""
        return render_template("live.html", title="Live View")

    @app.route("/whitelist")
    @require_auth
    def whitelist_page() -> str:
        """Page de gestion whitelist."""
        return render_template("whitelist.html", title="Whitelist")

    @app.route("/events")
    @require_auth
    def events_page() -> str:
        """Page journal des evenements."""
        return render_template("events.html", title="Events")

    @app.route("/settings")
    @require_auth
    def settings_page() -> str:
        """Page de configuration systeme."""
        return render_template("settings.html", title="Settings")

    @app.route("/snapshots")
    @require_auth
    def snapshots_page() -> str:
        """Page galerie snapshots."""
        return render_template("snapshots.html", title="Snapshots")

    @app.route("/api/stream")
    @require_auth
    def video_stream() -> Response:
        """Flux MJPEG temps reel pour affichage live."""

        def generate() -> Any:
            placeholder = _default_placeholder_frame()
            while True:
                frame: Optional[np.ndarray] = None
                if state.camera is not None and hasattr(state.camera, "get_frame"):
                    frame = state.camera.get_frame()

                if frame is None:
                    frame = placeholder.copy()

                if (
                    state.visualizer is not None
                    and state.tracker is not None
                    and hasattr(state.visualizer, "draw")
                    and hasattr(state.tracker, "active_entities")
                ):
                    try:
                        frame = state.visualizer.draw(
                            frame,
                            entities=state.tracker.active_entities,
                        )
                    except Exception as exc:
                        logger.debug("Visualizer stream error: %s", exc)

                ok, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                if not ok:
                    time.sleep(0.05)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
                time.sleep(0.04)

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/events")
    @require_auth
    def api_events() -> Response:
        """Retourne le journal d'evenements pagine."""
        page = max(1, int(request.args.get("page", 1)))
        per_page = max(1, min(100, int(request.args.get("per_page", 20))))

        events = _load_event_log(_resolve_path(cfg, EVENT_LOG_FILE))
        events = list(reversed(events))

        start = (page - 1) * per_page
        end = start + per_page
        paged = events[start:end]

        return jsonify(
            {
                "items": paged,
                "page": page,
                "per_page": per_page,
                "total": len(events),
            }
        )

    @app.route("/api/events/<int:event_id>")
    @require_auth
    def api_event_detail(event_id: int) -> Response:
        """Retourne le detail d'un evenement par index."""
        events = _load_event_log(_resolve_path(cfg, EVENT_LOG_FILE))
        events = list(reversed(events))

        if event_id < 0 or event_id >= len(events):
            return jsonify({"error": "Event not found"}), 404

        return jsonify(events[event_id])

    @app.route("/api/whitelist", methods=["GET"])
    @require_auth
    def api_whitelist_get() -> Response:
        """Liste les personnes de la whitelist."""
        return jsonify({"items": state.whitelist_repo.list_persons()})

    @app.route("/api/whitelist", methods=["POST"])
    @require_auth
    def api_whitelist_create() -> Response:
        """Ajoute une personne dans la whitelist."""
        payload = request.get_json(silent=True) or {}
        is_valid, error_msg = _validate_whitelist_payload(payload)
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        now_iso = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

        person = {
            "id": payload.get("id") or f"person_{int(time.time() * 1000)}",
            "name": payload.get("name") or "Unknown",
            "role": payload.get("role") or "Visitor",
            "access_level": payload.get("access_level") or "user",
            "embeddings": payload.get("embeddings") or [],
            "photos": payload.get("photos") or [],
            "enrolled_at": payload.get("enrolled_at") or now_iso,
            "last_seen": payload.get("last_seen"),
            "notes": payload.get("notes") or "",
        }

        try:
            state.whitelist_repo.add_person(person)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(person), 201

    @app.route("/api/whitelist/<person_id>", methods=["PUT"])
    @require_auth
    def api_whitelist_update(person_id: str) -> Response:
        """Met a jour une personne de la whitelist."""
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid payload"}), 400

        allowed_fields = {"name", "role", "access_level", "notes", "last_seen"}
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in allowed_fields:
                continue
            if key in {"name", "role", "notes", "last_seen"} and not isinstance(value, str):
                return jsonify({"error": f"Invalid field type: {key}"}), 400
            sanitized[key] = value

        if "access_level" in sanitized and sanitized["access_level"] not in {
            "guest", "user", "admin", "staff", "security"
        }:
            return jsonify({"error": "Invalid field: access_level"}), 400

        ok = state.whitelist_repo.update_person(person_id, sanitized)
        if not ok:
            return jsonify({"error": "Person not found"}), 404
        return jsonify({"updated": True})

    @app.route("/api/whitelist/<person_id>", methods=["DELETE"])
    @require_auth
    def api_whitelist_delete(person_id: str) -> Response:
        """Supprime une personne de la whitelist."""
        ok = state.whitelist_repo.remove_person(person_id)
        if not ok:
            return jsonify({"error": "Person not found"}), 404
        return jsonify({"deleted": True})

    @app.route("/api/settings", methods=["GET"])
    @require_auth
    def api_settings_get() -> Response:
        """Retourne la configuration actuelle."""
        return jsonify(
            {
                "camera": asdict(cfg.camera),
                "llm": asdict(cfg.llm),
                "detection": asdict(cfg.detection),
                "face": asdict(cfg.face),
                "audio": asdict(cfg.audio),
                "dashboard": asdict(cfg.dashboard),
                "alerts": asdict(cfg.alerts),
                "logging": asdict(cfg.logging),
            }
        )

    @app.route("/api/settings", methods=["PUT"])
    @require_auth
    def api_settings_put() -> Response:
        """Met a jour et persiste la configuration."""
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid payload"}), 400

        allowed_sections = {
            "camera", "llm", "detection", "face", "audio", "dashboard", "alerts", "logging"
        }
        if any(section not in allowed_sections for section in payload):
            return jsonify({"error": "Invalid section in payload"}), 400

        for section_name, section_values in payload.items():
            target = getattr(cfg, section_name, None)
            if target is None or not isinstance(section_values, dict):
                continue
            for key, value in section_values.items():
                if hasattr(target, key):
                    setattr(target, key, value)

        settings_path = _resolve_path(cfg, SETTINGS_FILE)
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        file_data: dict[str, Any] = {}
        if settings_path.exists():
            loaded = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                file_data = loaded

        for sec in [
            "camera",
            "llm",
            "detection",
            "face",
            "audio",
            "dashboard",
            "alerts",
            "logging",
        ]:
            file_data[sec] = asdict(getattr(cfg, sec))

        settings_path.write_text(
            yaml.safe_dump(file_data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        return jsonify({"updated": True})

    @app.route("/api/settings/llm/test", methods=["POST"])
    @require_auth
    def api_settings_llm_test() -> Response:
        """Teste la connectivite LLM actuelle."""
        client = state.llm_client or LLMClient(cfg)
        ok = asyncio.run(client.health_check())
        return jsonify({"ok": ok, "api_url": cfg.llm.api_url, "model": cfg.llm.model_name})

    @app.route("/api/status")
    @require_auth
    def api_status() -> Response:
        """Expose le statut runtime du systeme."""
        return jsonify(_collect_status())

    @app.route("/api/monitoring")
    @require_auth
    def api_monitoring() -> Response:
        """Expose les metriques detaillees du module de monitoring."""
        if state.monitor is None:
            fallback = _collect_status()
            fallback["monitoring_mode"] = "fallback"
            return jsonify(fallback)

        return jsonify(state.monitor.get_metrics())

    def _collect_status() -> dict[str, Any]:
        """Construit le payload statut sans contexte request."""
        metrics = _system_metrics()
        fps = 0.0
        if state.camera is not None and hasattr(state.camera, "metrics"):
            fps = float(getattr(state.camera.metrics, "fps_current", 0.0))

        person_count = 0
        if state.tracker is not None and hasattr(state.tracker, "person_count"):
            person_count = int(state.tracker.person_count)

        llm_ok = False
        try:
            llm_ok = asyncio.run((state.llm_client or LLMClient(cfg)).health_check())
        except Exception:
            llm_ok = False

        with state_lock:
            last_ai_action = state.last_ai_action
            last_ai_alert_level = state.last_ai_alert_level
            last_ai_response_ms = state.last_ai_response_ms

        return {
            "online": True,
            "fps": fps,
            "persons": person_count,
            "llm_connected": llm_ok,
            "cpu_percent": metrics["cpu_percent"],
            "ram_percent": metrics["ram_percent"],
            "last_ai_action": last_ai_action,
            "last_ai_alert_level": last_ai_alert_level,
            "last_ai_response_ms": last_ai_response_ms,
            "timestamp": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        }

    @app.route("/api/snapshots")
    @require_auth
    def api_snapshots() -> Response:
        """Retourne la liste des snapshots disponibles."""
        snapshots_root = _resolve_path(cfg, SNAPSHOTS_DIR)
        snapshots_root.mkdir(parents=True, exist_ok=True)

        files = []
        for path in sorted(snapshots_root.rglob("*.jpg"), reverse=True):
            try:
                rel = path.relative_to(cfg.project_root).as_posix()
            except ValueError:
                rel = path.as_posix()
            files.append(
                {
                    "path": rel,
                    "name": path.name,
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime)
                    .isoformat(),
                }
            )

        return jsonify({"items": files})

    @app.route("/api/stats")
    @require_auth
    def api_stats() -> Response:
        """Calcule des statistiques simples du systeme."""
        events = _load_event_log(_resolve_path(cfg, EVENT_LOG_FILE))
        unknown_count = sum(1 for e in events if e.get("event_type") == "unknown_face")
        alert_count = sum(
            1
            for e in events
            if str(e.get("event_type", "")).lower() in {"alert", "security", "critical"}
        )

        return jsonify(
            {
                "events_total": len(events),
                "alerts_total": alert_count,
                "unknown_faces_total": unknown_count,
                "whitelist_count": len(state.whitelist_repo.list_persons()),
            }
        )

    @app.route("/api/zones", methods=["GET"])
    @require_auth
    def api_zones_get() -> Response:
        """Retourne les zones de surveillance configurees."""
        if state.zone_manager is None:
            return jsonify({"items": []})
        return jsonify({"items": state.zone_manager.list_zones()})

    @app.route("/api/zones", methods=["PUT"])
    @require_auth
    def api_zones_put() -> Response:
        """Met a jour les zones de surveillance."""
        if state.zone_manager is None:
            return jsonify({"error": "Zone manager unavailable"}), 503

        payload = request.get_json(silent=True) or {}
        zones = payload.get("zones") if isinstance(payload, dict) else None
        if not isinstance(zones, list):
            return jsonify({"error": "Invalid payload: zones[] required"}), 400

        items = state.zone_manager.set_zones(zones)
        return jsonify({"updated": True, "items": items})

    @app.route("/api/report/daily")
    @require_auth
    def api_report_daily() -> Response:
        """Genere un rapport PDF journalier et retourne son chemin."""
        date_str = str(request.args.get("date") or datetime.now().strftime("%Y-%m-%d"))
        try:
            date_value = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "Invalid date format, expected YYYY-MM-DD"}), 400

        path = state.report_generator.generate_daily_report(date_value)
        return jsonify({"report_path": path, "date": date_str})

    @app.route("/api/v1/webhook/alert", methods=["POST"])
    @require_external_api_key
    def api_v1_webhook_alert() -> Response:
        """Endpoint externe pour recevoir une alerte tierce."""
        payload = request.get_json(silent=True) or {}
        if state.event_bus is not None:
            state.event_bus.emit(
                "external_alert",
                {
                    "source": payload.get("source", "external"),
                    "message": payload.get("message", ""),
                    "severity": payload.get("severity", "normal"),
                },
            )
        return jsonify({"accepted": True})

    @app.route("/api/v1/cameras")
    @require_external_api_key
    def api_v1_cameras() -> Response:
        """Liste les cameras connues (MVP: camera unique)."""
        camera_source = cfg.camera.source
        return jsonify({"items": [{"id": "cam_0", "name": "Main Camera", "source": camera_source}]})

    @app.route("/api/v1/stream/<cam_id>")
    @require_external_api_key
    def api_v1_stream(cam_id: str) -> Response:
        """Expose le stream video pour une camera (cam_0 seulement)."""
        if cam_id != "cam_0":
            return jsonify({"error": "Camera not found"}), 404
        return video_stream()

    @app.route("/api/v1/report/daily")
    @require_external_api_key
    def api_v1_report_daily() -> Response:
        """Expose le rapport journalier via API externe securisee."""
        return api_report_daily()

    def _emit_system_status() -> None:
        """Emet periodiquement le status systeme sur websocket."""
        if socketio is None:
            return
        while True:
            try:
                status = _collect_status()
                socketio.emit("system_status", status)
            except Exception as exc:
                logger.debug("Status socket emit error: %s", exc)
            time.sleep(2.0)

    if socketio is not None and not disable_background:
        thread = threading.Thread(target=_emit_system_status, daemon=True)
        thread.start()

        if state.event_bus is not None:
            def forward_event(name: str) -> Callable[[dict[str, Any]], None]:
                def callback(data: dict[str, Any]) -> None:
                    socketio.emit(name, data)
                    if name in {"face_unknown", "entity_lingering"}:
                        socketio.emit("alert", {"type": name, "data": data})
                    if name in {"face_unknown", "face_recognized", "tool_executed"}:
                        socketio.emit("new_event", {"type": name, "payload": data})

                return callback

            for evt in [
                "person_detected",
                "face_recognized",
                "face_unknown",
                "entity_lingering",
                "llm_response",
                "tool_executed",
            ]:
                state.event_bus.subscribe(evt, forward_event(evt))

    return app


def main() -> int:
    """Point d'entree local: demarre le dashboard Flask."""
    cfg = Config()
    app = create_app(cfg)
    socketio = app.extensions.get("socketio")

    host = cfg.dashboard.host
    port = cfg.dashboard.port

    if socketio is not None:
        socketio.run(app, host=host, port=port, debug=cfg.dashboard.debug)
    else:
        app.run(host=host, port=port, debug=cfg.dashboard.debug)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
