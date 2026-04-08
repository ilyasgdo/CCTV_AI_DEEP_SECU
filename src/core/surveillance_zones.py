"""Gestionnaire de zones de surveillance configurables (Etape 9)."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from src.core.config import Config
from src.core.tracker import TrackedEntity


@dataclass
class Zone:
    """Zone polygonale de surveillance."""

    id: str
    name: str
    zone_type: str
    points: list[list[int]]
    enabled: bool = True


class SurveillanceZoneManager:
    """Stocke, charge et evalue des zones polygonales."""

    def __init__(self, config: Config, zones_file: Optional[str] = None) -> None:
        self._config = config
        self._zones_path = (
            Path(zones_file)
            if zones_file
            else config.project_root / "data" / "reports" / "zones.json"
        )
        self._zones: list[Zone] = []
        self.load()

    def load(self) -> None:
        """Charge les zones depuis disque."""
        if not self._zones_path.exists():
            self._zones = []
            return

        data = json.loads(self._zones_path.read_text(encoding="utf-8"))
        zones: list[Zone] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    zones.append(
                        Zone(
                            id=str(item.get("id") or "zone"),
                            name=str(item.get("name") or "Zone"),
                            zone_type=str(item.get("zone_type") or "intrusion"),
                            points=[list(map(int, p)) for p in item.get("points", [])],
                            enabled=bool(item.get("enabled", True)),
                        )
                    )
                except Exception:
                    continue
        self._zones = zones

    def save(self) -> None:
        """Persiste les zones sur disque."""
        self._zones_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(zone) for zone in self._zones]
        self._zones_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def set_zones(self, zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remplace les zones existantes par une nouvelle configuration."""
        parsed: list[Zone] = []
        for idx, item in enumerate(zones):
            if not isinstance(item, dict):
                continue
            points = item.get("points") or []
            if not isinstance(points, list) or len(points) < 3:
                continue
            parsed.append(
                Zone(
                    id=str(item.get("id") or f"zone_{idx+1}"),
                    name=str(item.get("name") or f"Zone {idx+1}"),
                    zone_type=str(item.get("zone_type") or "intrusion"),
                    points=[list(map(int, p)) for p in points],
                    enabled=bool(item.get("enabled", True)),
                )
            )

        self._zones = parsed
        self.save()
        return [asdict(zone) for zone in self._zones]

    def list_zones(self) -> list[dict[str, Any]]:
        """Retourne la configuration courante des zones."""
        return [asdict(zone) for zone in self._zones]

    @staticmethod
    def _point_in_polygon(point: tuple[int, int], polygon: list[list[int]]) -> bool:
        """Test simple point-dans-polygone (ray casting)."""
        x, y = point
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / max(1e-6, (yj - yi)) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def check_intrusions(self, entities: list[TrackedEntity]) -> list[dict[str, Any]]:
        """Retourne les intrusions detectees pour les zones type intrusion."""
        hits: list[dict[str, Any]] = []
        active_intrusion_zones = [z for z in self._zones if z.enabled and z.zone_type == "intrusion"]
        if not active_intrusion_zones:
            return hits

        for entity in entities:
            center = getattr(entity, "last_center", None)
            if center is None:
                continue
            for zone in active_intrusion_zones:
                if self._point_in_polygon(center, zone.points):
                    hits.append(
                        {
                            "zone_id": zone.id,
                            "zone_name": zone.name,
                            "track_id": entity.track_id,
                            "center": {"x": center[0], "y": center[1]},
                        }
                    )
        return hits
