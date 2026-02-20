"""
Dashboard Web — Serveur Flask pour le CCTV AI DEEP SECU.

Endpoints :
  GET /                → Page d'accueil (dashboard HTML)
  GET /api/stats       → Stats système (JSON)
  GET /api/persons     → Stats par personne (JSON)
  GET /api/counter     → Compteur entrées/sorties (JSON)
  GET /api/alerts      → Alertes récentes (JSON)
  GET /video_feed      → Flux MJPEG live
"""
import cv2
import time
import json
import threading
import numpy as np
from flask import Flask, Response, jsonify, render_template
from pathlib import Path


def create_dashboard(shared_state: dict) -> Flask:
    """
    Crée l'application Flask avec accès à l'état partagé du système.
    
    Args:
        shared_state: Dict partagé avec le thread principal contenant :
            - frame: dernière frame BGR
            - fps, detections_count, person_stats, counter_stats, etc.
    """
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"
    
    app = Flask(__name__,
                template_folder=str(template_dir),
                static_folder=str(static_dir))

    # Supprimer les logs Flask (trop verbeux)
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/stats")
    def api_stats():
        """Stats système globales."""
        return jsonify({
            "fps": round(shared_state.get("fps", 0), 1),
            "detections": shared_state.get("detections_count", 0),
            "counter": shared_state.get("counter_stats", {}),
            "heatmap": shared_state.get("heatmap_stats", {}),
            "timestamp": time.time(),
        })

    @app.route("/api/persons")
    def api_persons():
        """Stats détaillées par personne."""
        ps = shared_state.get("person_stats", {})
        # Convertir les clés int en str pour JSON
        result = {}
        for tid, stats in ps.items():
            result[str(tid)] = {
                "name": stats.get("name", "INCONNU"),
                "presence_time": round(stats.get("presence_time", 0), 1),
                "current_action": stats.get("current_action", "N/A"),
                "action_durations": {
                    k: round(v, 1) for k, v in stats.get("action_durations", {}).items()
                    if v > 0.5
                },
                "top_action": stats.get("top_action"),
                "objects": stats.get("pose_objects", []),
            }
        return jsonify(result)

    @app.route("/api/alerts")
    def api_alerts():
        """Alertes récentes depuis la BDD."""
        try:
            db_stats = shared_state.get("db_stats", {})
            return jsonify({
                "total_alerts": db_stats.get("total_alerts", 0),
                "currently_present": db_stats.get("currently_present", 0),
                "total_records": db_stats.get("total_records", 0),
            })
        except Exception as e:
            return jsonify({"error": str(e)})

    @app.route("/api/counter")
    def api_counter():
        """Compteur entrées/sorties."""
        cs = shared_state.get("counter_stats", {})
        return jsonify({
            "entries": cs.get("total_entries", 0),
            "exits": cs.get("total_exits", 0),
            "present": cs.get("present", 0),
            "recent_events": cs.get("recent_events", [])[-20:],
        })

    def generate_mjpeg():
        """Générateur de frames MJPEG."""
        while True:
            frame = shared_state.get("frame")
            if frame is not None:
                # Réduire la qualité pour la bande passante
                ret, buffer = cv2.imencode('.jpg', frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')
            time.sleep(0.05)  # ~20 FPS max

    @app.route("/video_feed")
    def video_feed():
        """Flux vidéo MJPEG."""
        return Response(
            generate_mjpeg(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    return app


def run_dashboard(app: Flask, port: int = 5555):
    """Lance le serveur Flask dans un thread."""
    print(f"[DASHBOARD] Démarrage sur le port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
