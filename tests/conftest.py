"""Fixtures partagees Etape 8."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _test_defaults(monkeypatch: pytest.MonkeyPatch):
    """Parametres stables pour tests dashboard/monitoring."""
    monkeypatch.setenv("DASHBOARD_DISABLE_BACKGROUND", "true")
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "false")
    monkeypatch.setenv("DASHBOARD_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("DASHBOARD_RATE_LIMIT_PER_MIN", "120")
    yield


@pytest.fixture
def disable_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Desactive explicitement le rate limiting pour un test."""
    monkeypatch.setenv("DASHBOARD_RATE_LIMIT_ENABLED", "false")
