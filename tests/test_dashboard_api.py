"""Tests d'integration API dashboard (Etape 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.config import Config
from src.dashboard import app as dashboard_app
from src.dashboard.app import create_app


class FakeLLMClient:
    """Client LLM factice pour tests dashboard."""

    async def health_check(self) -> bool:
        return True


class FakeCamera:
    """Camera factice retournant une frame noire."""

    class Metrics:
        fps_current = 27.5

    metrics = Metrics()

    def get_frame(self):
        return np.zeros((240, 320, 3), dtype=np.uint8)


class FakeTracker:
    """Tracker factice."""

    person_count = 2
    active_entities = []


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Client Flask de test configure avec chemins temporaires."""
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "false")
    monkeypatch.setenv("DASHBOARD_DISABLE_BACKGROUND", "true")

    event_log = tmp_path / "event_log.json"
    event_log.write_text("[]", encoding="utf-8")

    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("{}", encoding="utf-8")

    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(dashboard_app, "EVENT_LOG_FILE", str(event_log))
    monkeypatch.setattr(dashboard_app, "SETTINGS_FILE", str(settings_file))
    monkeypatch.setattr(dashboard_app, "SNAPSHOTS_DIR", str(snapshots_dir))

    cfg = Config()
    cfg.face.whitelist_dir = str(tmp_path / "whitelist")

    app = create_app(
        config=cfg,
        camera=FakeCamera(),
        tracker=FakeTracker(),
        llm_client=FakeLLMClient(),
    )
    app.testing = True

    with app.test_client() as test_client:
        yield test_client


def test_dashboard_pages_load(client) -> None:
    """Les pages principales du dashboard sont accessibles."""
    for route in ["/", "/whitelist", "/events", "/settings", "/snapshots"]:
        response = client.get(route)
        assert response.status_code == 200


def test_stream_endpoint(client) -> None:
    """Le flux MJPEG est expose sur /api/stream."""
    response = client.get("/api/stream")
    assert response.status_code == 200
    assert "multipart/x-mixed-replace" in response.mimetype


def test_whitelist_crud(client) -> None:
    """CRUD whitelist via API dashboard."""
    create = client.post(
        "/api/whitelist",
        json={"name": "Ilyas", "role": "Owner", "access_level": "admin"},
    )
    assert create.status_code == 201
    person_id = create.get_json()["id"]

    listing = client.get("/api/whitelist")
    assert listing.status_code == 200
    assert len(listing.get_json()["items"]) == 1

    update = client.put(f"/api/whitelist/{person_id}", json={"role": "Admin"})
    assert update.status_code == 200

    delete = client.delete(f"/api/whitelist/{person_id}")
    assert delete.status_code == 200


def test_settings_get_put_and_llm_test(client) -> None:
    """Lecture/ecriture settings et test de connexion LLM."""
    get_resp = client.get("/api/settings")
    assert get_resp.status_code == 200

    put_resp = client.put("/api/settings", json={"llm": {"api_url": "http://localhost:11434"}})
    assert put_resp.status_code == 200
    assert put_resp.get_json()["updated"] is True

    llm_test = client.post("/api/settings/llm/test")
    assert llm_test.status_code == 200
    assert llm_test.get_json()["ok"] is True


def test_events_and_stats_endpoints(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Endpoints events, event detail, stats et status."""
    event_log = Path(dashboard_app.EVENT_LOG_FILE)
    event_log.write_text(
        '[{"timestamp":"2026-04-08T12:00:00Z","event_type":"security","message":"intrusion"}]',
        encoding="utf-8",
    )

    events = client.get("/api/events")
    assert events.status_code == 200
    assert events.get_json()["total"] == 1

    detail = client.get("/api/events/0")
    assert detail.status_code == 200

    stats = client.get("/api/stats")
    assert stats.status_code == 200
    assert stats.get_json()["events_total"] == 1

    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.get_json()["persons"] == 2


def test_snapshots_endpoint(client) -> None:
    """Liste snapshots disponible via API."""
    snapshots_dir = Path(dashboard_app.SNAPSHOTS_DIR)
    (snapshots_dir / "a.jpg").write_bytes(b"test")

    response = client.get("/api/snapshots")
    assert response.status_code == 200
    assert len(response.get_json()["items"]) >= 1


def test_monitoring_endpoint(client) -> None:
    """L'endpoint monitoring est expose."""
    response = client.get("/api/monitoring")
    assert response.status_code == 200
    payload = response.get_json()
    assert "cpu_percent" in payload


def test_whitelist_validation_rejects_invalid_payload(client) -> None:
    """Validation API: payload invalide retourne 400."""
    response = client.post(
        "/api/whitelist",
        json={"name": "", "access_level": "root"},
    )
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_settings_validation_rejects_unknown_section(client) -> None:
    """Validation API: section de config inconnue rejetee."""
    response = client.put("/api/settings", json={"invalid_section": {"x": 1}})
    assert response.status_code == 400


def test_auth_required_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Securite dashboard: 401 sans credentials quand auth active."""
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "true")
    monkeypatch.setenv("DASHBOARD_DISABLE_BACKGROUND", "true")
    monkeypatch.setenv("DASHBOARD_USERNAME", "admin")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret")

    event_log = tmp_path / "event_log.json"
    event_log.write_text("[]", encoding="utf-8")
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("{}", encoding="utf-8")
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(dashboard_app, "EVENT_LOG_FILE", str(event_log))
    monkeypatch.setattr(dashboard_app, "SETTINGS_FILE", str(settings_file))
    monkeypatch.setattr(dashboard_app, "SNAPSHOTS_DIR", str(snapshots_dir))

    cfg = Config()
    cfg.face.whitelist_dir = str(tmp_path / "whitelist")
    app = create_app(config=cfg, camera=FakeCamera(), tracker=FakeTracker(), llm_client=FakeLLMClient())
    app.testing = True

    with app.test_client() as auth_client:
        denied = auth_client.get("/api/status")
        assert denied.status_code == 401

        allowed = auth_client.get("/api/status", headers={"X-Dashboard-Password": "secret"})
        assert allowed.status_code == 200


def test_rate_limit_blocks_excess_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Securite API: rate limit actif retourne 429."""
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "false")
    monkeypatch.setenv("DASHBOARD_DISABLE_BACKGROUND", "true")
    monkeypatch.setenv("DASHBOARD_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("DASHBOARD_RATE_LIMIT_PER_MIN", "5")

    event_log = tmp_path / "event_log.json"
    event_log.write_text("[]", encoding="utf-8")
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("{}", encoding="utf-8")
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(dashboard_app, "EVENT_LOG_FILE", str(event_log))
    monkeypatch.setattr(dashboard_app, "SETTINGS_FILE", str(settings_file))
    monkeypatch.setattr(dashboard_app, "SNAPSHOTS_DIR", str(snapshots_dir))

    cfg = Config()
    cfg.face.whitelist_dir = str(tmp_path / "whitelist")
    app = create_app(config=cfg, camera=FakeCamera(), tracker=FakeTracker(), llm_client=FakeLLMClient())
    app.testing = True

    with app.test_client() as rate_client:
        for _ in range(5):
            assert rate_client.get("/api/status").status_code == 200
        limited = rate_client.get("/api/status")
        assert limited.status_code == 429
