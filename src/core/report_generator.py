"""Generation de rapports PDF automatiques (Etape 9)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fpdf import FPDF  # type: ignore[import-not-found]

from src.core.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Genere des rapports PDF journaliers a partir des events."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._reports_dir = config.project_root / "data" / "reports"
        self._event_log = config.project_root / "data" / "event_log.json"
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_events(self) -> list[dict[str, Any]]:
        """Charge les evenements depuis event_log.json."""
        if not self._event_log.exists():
            return []
        try:
            data = json.loads(self._event_log.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.error("Lecture event_log impossible: %s", exc)
        return []

    def _events_for_date(self, date_value: datetime) -> list[dict[str, Any]]:
        """Filtre les evenements pour une date (YYYY-MM-DD)."""
        day_prefix = date_value.strftime("%Y-%m-%d")
        return [e for e in self._load_events() if str(e.get("timestamp", "")).startswith(day_prefix)]

    @staticmethod
    def _safe_pdf_text(value: str, max_len: int = 120) -> str:
        """Normalise un texte pour PDF (core fonts FPDF)."""
        text = (value or "").replace("\n", " ").replace("\r", " ").strip()
        text = " ".join(text.split())
        text = text[:max_len]
        # Core fonts FPDF acceptent latin-1; on degrade proprement si besoin.
        return text.encode("latin-1", "replace").decode("latin-1")

    def generate_daily_report(self, date_value: datetime) -> str:
        """Genere le rapport PDF journalier et retourne son chemin."""
        events = self._events_for_date(date_value)
        known_count = sum(1 for e in events if e.get("event_type") == "face_recognized")
        unknown_count = sum(1 for e in events if e.get("event_type") == "face_unknown")
        alert_count = sum(
            1
            for e in events
            if str(e.get("event_type", "")).lower() in {"alert", "security", "critical", "zone_intrusion"}
        )

        title_date = date_value.strftime("%Y-%m-%d")
        out_path = self._reports_dir / f"REPORT_{title_date}.pdf"

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"Sentinel-AI Daily Report - {title_date}")
        pdf.ln(10)

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Total events: {len(events)}")
        pdf.ln(8)
        pdf.cell(0, 8, f"Known persons events: {known_count}")
        pdf.ln(8)
        pdf.cell(0, 8, f"Unknown persons events: {unknown_count}")
        pdf.ln(8)
        pdf.cell(0, 8, f"Alert events: {alert_count}")
        pdf.ln(8)

        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Recent events (up to 25)")
        pdf.ln(8)

        pdf.set_font("Helvetica", "", 9)
        for event in events[-25:]:
            ts = self._safe_pdf_text(str(event.get("timestamp", ""))[:19], max_len=24)
            ev_type = self._safe_pdf_text(str(event.get("event_type", "generic")), max_len=30)
            msg = self._safe_pdf_text(str(event.get("message", "")), max_len=90)
            line = f"- {ts} | {ev_type} | {msg}"
            pdf.set_x(10)
            pdf.multi_cell(190, 5, line)

        pdf.output(str(out_path))
        return str(out_path)
