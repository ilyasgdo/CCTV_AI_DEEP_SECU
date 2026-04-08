"""Enregistreur de clips video sur alerte (Etape 9)."""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.core.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlertClipRecorder:
    """Enregistre un clip MP4 avec buffer pre/post alerte.

    Le module conserve un buffer circulaire des frames recentes (pre-alert).
    Lors d'une alerte, il collecte aussi les frames post-alert puis ecrit un MP4.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.alerts
        self._project_root = config.project_root
        self._fps = max(1, int(config.camera.fps))

        self._pre_frames = max(1, int(self._cfg.clip_pre_seconds) * self._fps)
        self._post_frames = max(1, int(self._cfg.clip_post_seconds) * self._fps)

        self._buffer: deque[np.ndarray] = deque(maxlen=self._pre_frames)
        self._lock = threading.Lock()

        self._active = False
        self._post_counter = 0
        self._pending_frames: list[np.ndarray] = []
        self._pending_reason = "alert"

    def add_frame(self, frame: Optional[np.ndarray]) -> None:
        """Ajoute une frame au buffer et complete un enregistrement en cours."""
        if frame is None:
            return

        frame_copy = frame.copy()
        clip_to_flush: Optional[tuple[list[np.ndarray], str]] = None

        with self._lock:
            self._buffer.append(frame_copy)
            if self._active:
                self._pending_frames.append(frame_copy)
                self._post_counter += 1
                if self._post_counter >= self._post_frames:
                    clip_to_flush = (list(self._pending_frames), self._pending_reason)
                    self._active = False
                    self._post_counter = 0
                    self._pending_frames = []
                    self._pending_reason = "alert"

        if clip_to_flush is not None:
            frames, reason = clip_to_flush
            threading.Thread(
                target=self._write_clip,
                args=(frames, reason),
                daemon=True,
            ).start()

    def on_alert(self, reason: str = "alert") -> None:
        """Declenche un enregistrement de clip a partir du buffer courant."""
        with self._lock:
            if self._active:
                return

            self._pending_frames = list(self._buffer)
            self._pending_reason = (reason or "alert").strip().replace(" ", "_")[:48]
            self._post_counter = 0
            self._active = True

    def _write_clip(self, frames: list[np.ndarray], reason: str) -> None:
        """Ecrit un MP4 dans data/clips a partir des frames capturees."""
        if not frames:
            return

        clips_dir = self._project_root / "data" / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"ALERT_{stamp}_{reason}.mp4"
        out_path = clips_dir / filename

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self._fps),
            (w, h),
        )

        try:
            for frame in frames:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()

        logger.info("Clip alerte sauvegarde: %s (%s frames)", out_path, len(frames))
