"""Outil d'ecriture d'evenements dans un journal JSON."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.config import Config
from src.effector.base_tool import BaseTool, ToolResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EventLogTool(BaseTool):
    """Ajoute des evenements applicatifs dans data/event_log.json."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._log_file = self._config.project_root / "data" / "event_log.json"

    @property
    def name(self) -> str:
        return "log_event"

    @property
    def description(self) -> str:
        return "Journalise un evenement dans data/event_log.json"

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            await asyncio.to_thread(self._append_event, params)
            return ToolResult(
                tool_name=self.name,
                success=True,
                message="Evenement journalise",
                data={"event_type": str(params.get("event_type") or "generic")},
            )
        except Exception as exc:
            logger.error("EventLogTool failure: %s", exc, exc_info=True)
            return ToolResult(
                tool_name=self.name,
                success=False,
                message="Echec journalisation",
                error=str(exc),
            )

    def _append_event(self, params: dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "event_type": str(params.get("event_type") or "generic"),
            "message": str(params.get("message") or ""),
            "details": params.get("details") or {},
        }

        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if self._log_file.exists():
                data = json.loads(self._log_file.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    data = []
            else:
                data = []

            data.append(event)
            self._log_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
