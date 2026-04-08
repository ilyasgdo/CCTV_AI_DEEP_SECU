"""Tests du gestionnaire de zones de surveillance (Etape 9)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.config import Config
from src.core.surveillance_zones import SurveillanceZoneManager


@dataclass
class FakeEntity:
    """Entite minimale pour test zones."""

    track_id: int
    last_center: tuple[int, int] | None


def test_zone_manager_set_and_detect_intrusion(tmp_path: Path) -> None:
    """Detection d'intrusion si centre entite dans un polygone."""
    cfg = Config()
    manager = SurveillanceZoneManager(cfg, zones_file=str(tmp_path / "zones.json"))

    manager.set_zones(
        [
            {
                "id": "z1",
                "name": "Porte",
                "zone_type": "intrusion",
                "points": [[0, 0], [200, 0], [200, 200], [0, 200]],
                "enabled": True,
            }
        ]
    )

    hits = manager.check_intrusions([FakeEntity(track_id=1, last_center=(120, 120))])
    assert len(hits) == 1
    assert hits[0]["zone_id"] == "z1"

    hits_outside = manager.check_intrusions([FakeEntity(track_id=2, last_center=(260, 260))])
    assert hits_outside == []
