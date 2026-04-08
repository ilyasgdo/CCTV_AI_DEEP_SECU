"""
Module de gestion de caméra pour Sentinel-AI.

Gestionnaire de caméra polyvalent supportant webcam, flux RTSP/HTTP,
DroidCam et fichiers vidéo. La capture tourne dans un thread dédié
avec buffer circulaire et auto-reconnexion.

Usage:
    from src.core.camera import Camera
    from src.core.config import Config

    config = Config()
    camera = Camera(config)
    camera.start()

    frame = camera.get_frame()       # Dernière frame BGR
    jpeg = camera.get_snapshot()     # Frame encodée en JPEG

    camera.stop()
"""

import base64
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.core.config import Config
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────

class CameraError(Exception):
    """Erreur liée à la caméra."""
    pass


class CameraConnectionError(CameraError):
    """Impossible de se connecter à la source vidéo."""
    pass


# ──────────────────────────────────────────────
# Métriques
# ──────────────────────────────────────────────

@dataclass
class CameraMetrics:
    """Métriques en temps réel de la caméra."""
    fps_current: float = 0.0
    fps_average: float = 0.0
    frames_captured: int = 0
    frames_dropped: int = 0
    latency_ms: float = 0.0
    is_connected: bool = False
    reconnect_count: int = 0
    uptime_seconds: float = 0.0


# ──────────────────────────────────────────────
# Camera Manager
# ──────────────────────────────────────────────

class Camera:
    """
    Gestionnaire de caméra multi-source avec capture threadée.

    Supporte : webcam locale, flux RTSP, flux HTTP/IP,
    DroidCam, fichiers vidéo (.mp4, .avi, etc.).

    La capture tourne dans un thread dédié pour ne jamais
    bloquer le thread principal. Un buffer circulaire garde
    les N dernières frames.

    Args:
        config: Configuration du projet.
        event_bus: Bus d'événements (optionnel).

    Example:
        >>> camera = Camera(config)
        >>> camera.start()
        >>> frame = camera.get_frame()
        >>> camera.stop()
    """

    def __init__(
        self,
        config: Config,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._config = config.camera
        self._event_bus = event_bus

        # Source vidéo
        self._source = self._parse_source(self._config.source)
        self._cap: Optional[cv2.VideoCapture] = None

        # Buffer circulaire thread-safe
        self._buffer: deque[np.ndarray] = deque(
            maxlen=self._config.buffer_size
        )
        self._buffer_lock = threading.Lock()

        # Thread de capture
        self._capture_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._connected = threading.Event()

        # Métriques
        self._metrics = CameraMetrics()
        self._fps_timestamps: deque[float] = deque(maxlen=60)
        self._start_time: float = 0.0

        # Frame counter
        self._frame_id: int = 0

    @staticmethod
    def _parse_source(source: str) -> int | str:
        """Parse la source vidéo.

        Args:
            source: Identifiant de la source ("0", URL, chemin).

        Returns:
            int pour webcam, str pour URL/fichier.
        """
        try:
            return int(source)
        except ValueError:
            return source

    def start(self) -> None:
        """Démarre la capture vidéo dans un thread dédié.

        Raises:
            CameraError: Si la caméra est déjà démarrée.
        """
        if self._running.is_set():
            logger.warning("La caméra est déjà en cours de capture.")
            return

        logger.info(f"📷 Démarrage caméra: source={self._source}")
        self._start_time = time.time()
        self._running.set()

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCaptureThread",
            daemon=True,
        )
        self._capture_thread.start()

        # Attendre la connexion (max 10 secondes)
        if self._connected.wait(timeout=10.0):
            logger.info("📷 Caméra connectée et prête.")
        else:
            logger.warning(
                "📷 Timeout de connexion caméra. "
                "La capture continue en arrière-plan."
            )

    def stop(self) -> None:
        """Arrête la capture et libère les ressources."""
        logger.info("📷 Arrêt de la caméra...")
        self._running.clear()

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)

        self._release_capture()
        self._connected.clear()
        logger.info("📷 Caméra arrêtée.")

    def _connect(self) -> bool:
        """Tente de se connecter à la source vidéo.

        Returns:
            True si la connexion est établie.
        """
        try:
            self._release_capture()

            self._cap = cv2.VideoCapture(self._source)

            if not self._cap.isOpened():
                logger.error(
                    f"Impossible d'ouvrir la source: {self._source}"
                )
                return False

            # Configurer la résolution et le FPS si webcam/IP
            if isinstance(self._source, int):
                self._cap.set(
                    cv2.CAP_PROP_FRAME_WIDTH, self._config.width
                )
                self._cap.set(
                    cv2.CAP_PROP_FRAME_HEIGHT, self._config.height
                )
                self._cap.set(
                    cv2.CAP_PROP_FPS, self._config.fps
                )

            # Lire les propriétés réelles
            actual_w = int(
                self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )
            actual_h = int(
                self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

            logger.info(
                f"📷 Connecté: {actual_w}x{actual_h} @ "
                f"{actual_fps:.1f} FPS"
            )

            self._connected.set()
            self._metrics.is_connected = True
            self._metrics.reconnect_count += 1

            if self._event_bus:
                self._event_bus.emit("camera_connected", {
                    "source": str(self._source),
                    "resolution": f"{actual_w}x{actual_h}",
                    "fps": actual_fps,
                })

            return True

        except Exception as e:
            logger.error(f"Erreur de connexion caméra: {e}")
            if self._event_bus:
                self._event_bus.emit("camera_error", {
                    "source": str(self._source),
                    "error": str(e),
                })
            return False

    def _release_capture(self) -> None:
        """Libère l'objet VideoCapture."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _capture_loop(self) -> None:
        """Boucle principale de capture (tourne dans un thread).

        Gère la connexion, la reconnexion automatique avec backoff
        exponentiel, et le remplissage du buffer circulaire.
        """
        reconnect_attempt = 0

        while self._running.is_set():
            # ── Connexion / Reconnexion ──
            if self._cap is None or not self._cap.isOpened():
                self._connected.clear()
                self._metrics.is_connected = False

                if reconnect_attempt > 0:
                    # Backoff exponentiel
                    delay = min(
                        self._config.reconnect_delay * (
                            2 ** (reconnect_attempt - 1)
                        ),
                        30.0,  # Max 30 secondes
                    )
                    logger.warning(
                        f"📷 Reconnexion dans {delay:.1f}s "
                        f"(tentative {reconnect_attempt}/"
                        f"{self._config.max_reconnect_attempts})"
                    )
                    # Dormir par petits bouts pour pouvoir arrêter
                    deadline = time.time() + delay
                    while (time.time() < deadline
                           and self._running.is_set()):
                        time.sleep(0.1)
                    if not self._running.is_set():
                        break

                if reconnect_attempt >= \
                        self._config.max_reconnect_attempts:
                    logger.error(
                        "📷 Nombre max de reconnexions atteint."
                    )
                    if self._event_bus:
                        self._event_bus.emit(
                            "camera_disconnected",
                            {
                                "source": str(self._source),
                                "reason": "max_reconnect_reached",
                            },
                        )
                    break

                if self._connect():
                    reconnect_attempt = 0
                else:
                    reconnect_attempt += 1
                    continue

            # ── Lecture de frame ──
            try:
                t_start = time.perf_counter()
                ret, frame = self._cap.read()
                t_end = time.perf_counter()

                if not ret or frame is None:
                    self._metrics.frames_dropped += 1
                    logger.debug("Frame dropped (ret=False)")

                    # Trop de drops consécutifs = déconnexion
                    if self._metrics.frames_dropped % 30 == 0:
                        logger.warning(
                            f"📷 {self._metrics.frames_dropped} "
                            f"frames perdues. Reconnexion..."
                        )
                        self._release_capture()
                        reconnect_attempt += 1
                    continue

                # Ajouter au buffer
                with self._buffer_lock:
                    self._buffer.append(frame)

                # Métriques
                self._frame_id += 1
                self._metrics.frames_captured += 1
                self._metrics.latency_ms = (
                    (t_end - t_start) * 1000
                )

                now = time.time()
                self._fps_timestamps.append(now)
                if len(self._fps_timestamps) >= 2:
                    elapsed = (
                        self._fps_timestamps[-1]
                        - self._fps_timestamps[0]
                    )
                    if elapsed > 0:
                        self._metrics.fps_current = (
                            (len(self._fps_timestamps) - 1) / elapsed
                        )

                if self._start_time > 0:
                    self._metrics.uptime_seconds = (
                        now - self._start_time
                    )
                    total_time = self._metrics.uptime_seconds
                    if total_time > 0:
                        self._metrics.fps_average = (
                            self._metrics.frames_captured / total_time
                        )

            except Exception as e:
                logger.error(f"Erreur de capture: {e}")
                self._release_capture()
                reconnect_attempt += 1

        logger.debug("Boucle de capture terminée.")

    def get_frame(self) -> Optional[np.ndarray]:
        """Retourne la dernière frame capturée (thread-safe).

        Returns:
            Frame BGR en numpy array, ou None si aucune frame
            n'est disponible.
        """
        with self._buffer_lock:
            if self._buffer:
                return self._buffer[-1].copy()
        return None

    def get_snapshot(
        self, quality: int = 80
    ) -> Optional[bytes]:
        """Retourne la dernière frame encodée en JPEG.

        Utile pour l'envoi au LLM (base64) ou l'affichage web.

        Args:
            quality: Qualité JPEG (1-100).

        Returns:
            Bytes JPEG ou None si pas de frame.
        """
        frame = self.get_frame()
        if frame is None:
            return None

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)

        if success:
            return buffer.tobytes()
        return None

    def get_snapshot_base64(self, quality: int = 80) -> Optional[str]:
        """Retourne la dernière frame en JPEG encodé en base64.

        Args:
            quality: Qualité JPEG (1-100).

        Returns:
            String base64 ou None.
        """
        jpeg = self.get_snapshot(quality)
        if jpeg:
            return base64.b64encode(jpeg).decode("utf-8")
        return None

    def get_buffer_frames(
        self, count: Optional[int] = None
    ) -> list[np.ndarray]:
        """Retourne les N dernières frames du buffer.

        Args:
            count: Nombre de frames (None = tout le buffer).

        Returns:
            Liste de frames (les plus anciennes en premier).
        """
        with self._buffer_lock:
            frames = list(self._buffer)
        if count is not None:
            return frames[-count:]
        return frames

    @property
    def frame_id(self) -> int:
        """ID de la dernière frame capturée."""
        return self._frame_id

    @property
    def metrics(self) -> CameraMetrics:
        """Retourne les métriques en temps réel."""
        return self._metrics

    @property
    def is_connected(self) -> bool:
        """Indique si la caméra est connectée."""
        return self._connected.is_set()

    @property
    def is_running(self) -> bool:
        """Indique si la boucle de capture tourne."""
        return self._running.is_set()

    def release(self) -> None:
        """Alias pour stop() — compatibilité."""
        self.stop()

    def __repr__(self) -> str:
        status = "connectée" if self.is_connected else "déconnectée"
        return (
            f"Camera(source={self._source}, "
            f"status={status}, "
            f"fps={self._metrics.fps_current:.1f}, "
            f"frames={self._metrics.frames_captured})"
        )

    def __del__(self) -> None:
        """Nettoyage à la destruction."""
        if self._running.is_set():
            self.stop()
