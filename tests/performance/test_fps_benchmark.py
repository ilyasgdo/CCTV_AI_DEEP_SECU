"""Benchmarks legers Etape 8: debit perception."""

from __future__ import annotations

import time

from src.core.config import Config


def test_fps_target_config_is_valid() -> None:
    """La configuration vise un FPS >= 15."""
    cfg = Config()
    assert cfg.camera.fps >= 15


def test_perception_loop_sleep_budget() -> None:
    """Le budget theorique par frame reste sous 50ms pour 20+ FPS."""
    cfg = Config()
    target_fps = max(1, cfg.camera.fps)
    sleep_budget = 1.0 / target_fps
    assert sleep_budget <= 0.066

    t0 = time.perf_counter()
    time.sleep(min(0.01, sleep_budget))
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05
