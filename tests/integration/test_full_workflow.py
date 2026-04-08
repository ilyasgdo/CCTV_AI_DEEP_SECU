"""Tests integration pipeline global Etape 8."""

from __future__ import annotations

import asyncio

import main as sentinel_main


async def _start_then_stop(app: sentinel_main.SentinelAI) -> None:
    await app.start()
    await asyncio.sleep(0.15)
    await app.shutdown()


def test_full_workflow_cycle_5s_smoke(monkeypatch) -> None:
    """Smoke test workflow complet (startup -> cycle -> shutdown)."""
    monkeypatch.setenv("DASHBOARD_DISABLE_BACKGROUND", "true")
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "false")
    app = sentinel_main.SentinelAI(runtime_options=sentinel_main.RuntimeOptions(no_dashboard=True))
    asyncio.run(_start_then_stop(app))

    assert app._frame_id >= 0
    assert app.monitor.get_metrics()["uptime_seconds"] >= 0
