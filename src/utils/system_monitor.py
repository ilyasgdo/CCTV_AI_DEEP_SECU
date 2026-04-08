"""Monitoring runtime centralise pour Sentinel-AI (Etape 8)."""

from __future__ import annotations

import time
from typing import Callable, Optional

from src.cognitive.llm_client import LLMClient
from src.utils.event_bus import EventBus


class SystemMonitor:
    """Expose les metriques systeme et applicatives en continu.

    Args:
        event_bus: Bus d'evenements pour comptage global.
        llm_client: Client LLM pour metriques latence/erreurs.
        fps_provider: Callback retournant le FPS courant.
        person_count_provider: Callback retournant le nombre de personnes trackees.
    """

    def __init__(
        self,
        event_bus: EventBus,
        llm_client: LLMClient,
        fps_provider: Callable[[], float],
        person_count_provider: Callable[[], int],
        ram_alert_threshold: float = 80.0,
    ) -> None:
        self._event_bus = event_bus
        self._llm_client = llm_client
        self._fps_provider = fps_provider
        self._person_count_provider = person_count_provider
        self._start_time = time.time()
        self._ram_alert_threshold = float(ram_alert_threshold)

    @staticmethod
    def _safe_system_metrics() -> tuple[float, float, float]:
        """Retourne CPU%, RAM% et RAM RSS MB (fallback 0 si indisponible)."""
        cpu = 0.0
        ram_pct = 0.0
        rss_mb = 0.0
        try:
            import psutil  # type: ignore[import-not-found]

            cpu = float(psutil.cpu_percent(interval=0.0))
            vm = psutil.virtual_memory()
            ram_pct = float(vm.percent)
            rss_mb = float(psutil.Process().memory_info().rss / (1024 * 1024))
        except Exception:
            pass

        return cpu, ram_pct, rss_mb

    def get_metrics(self) -> dict[str, float | int | bool]:
        """Construit le snapshot courant des metriques."""
        cpu_percent, ram_percent, ram_used_mb = self._safe_system_metrics()
        llm_metrics = getattr(self._llm_client, "metrics", None)

        if llm_metrics is None:
            class _FallbackMetrics:
                total_requests = 0
                failed_requests = 0
                last_response_ms = 0.0
                avg_response_ms = 0.0

            llm_metrics = _FallbackMetrics()

        total_requests = llm_metrics.total_requests
        llm_error_rate = 0.0
        if total_requests > 0:
            llm_error_rate = float(llm_metrics.failed_requests / total_requests)

        metrics: dict[str, float | int | bool] = {
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "ram_used_mb": ram_used_mb,
            "ram_alert": ram_percent >= self._ram_alert_threshold,
            "fps_current": float(self._fps_provider()),
            "persons_detected": int(self._person_count_provider()),
            "llm_latency_ms": float(llm_metrics.last_response_ms),
            "llm_avg_latency_ms": float(llm_metrics.avg_response_ms),
            "llm_error_rate": llm_error_rate,
            "events_total": int(self._event_bus.total_events),
            "uptime_seconds": int(time.time() - self._start_time),
        }

        return metrics
