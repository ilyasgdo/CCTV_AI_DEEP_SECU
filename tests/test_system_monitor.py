"""Tests unitaires du module SystemMonitor (Etape 8)."""

from __future__ import annotations

from src.cognitive.llm_client import LLMClient
from src.core.config import Config
from src.utils.event_bus import EventBus
from src.utils.system_monitor import SystemMonitor


class FakeLLMClient(LLMClient):
    """Client LLM avec metriques controlees pour tests."""

    def __init__(self) -> None:
        cfg = Config()
        super().__init__(cfg)
        self._metrics.total_requests = 10
        self._metrics.failed_requests = 2
        self._metrics.last_response_ms = 820.0
        self._metrics.avg_response_ms = 640.0


def test_system_monitor_metrics_payload() -> None:
    """Le monitor expose les metriques attendues."""
    bus = EventBus()
    bus.emit("test", {})

    llm = FakeLLMClient()
    monitor = SystemMonitor(
        event_bus=bus,
        llm_client=llm,
        fps_provider=lambda: 18.5,
        person_count_provider=lambda: 3,
    )

    metrics = monitor.get_metrics()

    assert "cpu_percent" in metrics
    assert "ram_percent" in metrics
    assert metrics["fps_current"] == 18.5
    assert metrics["persons_detected"] == 3
    assert metrics["llm_latency_ms"] == 820.0
    assert metrics["llm_error_rate"] == 0.2
    assert metrics["events_total"] >= 1
    assert metrics["uptime_seconds"] >= 0
