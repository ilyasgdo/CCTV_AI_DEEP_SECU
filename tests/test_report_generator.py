"""Tests du generateur de rapports PDF (Etape 9)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.core.config import Config
from src.core.report_generator import ReportGenerator


def test_generate_daily_report(tmp_path: Path) -> None:
    """Le rapport PDF journalier est genere sur disque."""
    cfg = Config()
    gen = ReportGenerator(cfg)

    gen._reports_dir = tmp_path
    gen._event_log = tmp_path / "event_log.json"

    gen._event_log.write_text(
        json.dumps(
            [
                {"timestamp": "2026-04-08T10:00:00Z", "event_type": "face_recognized", "message": "ok"},
                {"timestamp": "2026-04-08T10:10:00Z", "event_type": "zone_intrusion", "message": "zone"},
            ]
        ),
        encoding="utf-8",
    )

    out = gen.generate_daily_report(datetime(2026, 4, 8))
    assert Path(out).exists()
    assert out.endswith(".pdf")
