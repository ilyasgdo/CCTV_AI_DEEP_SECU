"""Tests pour l'enregistreur de clips d'alerte (Etape 9)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.core.alert_clip_recorder import AlertClipRecorder
from src.core.config import Config


def test_alert_clip_recorder_writes_mp4(tmp_path: Path) -> None:
    """Un clip MP4 est ecrit apres alerte + frames post-alert."""
    cfg = Config()
    cfg.alerts.clip_pre_seconds = 1
    cfg.alerts.clip_post_seconds = 1
    cfg.camera.fps = 5

    recorder = AlertClipRecorder(cfg)
    recorder._project_root = tmp_path

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    for _ in range(5):
        recorder.add_frame(frame)

    recorder.on_alert("intrusion")
    for _ in range(8):
        recorder.add_frame(frame)

    time.sleep(0.3)
    clips = list((tmp_path / "data" / "clips").glob("*.mp4"))
    assert len(clips) >= 1
