"""Outil de declenchement d'alarme sonore locale."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from src.effector.base_tool import BaseTool, ToolResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

AlarmPlayFn = Callable[[str, int], Awaitable[None]]


class AlarmTool(BaseTool):
    """Declenche une alarme sonore locale pour une duree donnee."""

    def __init__(self, play_fn: Optional[AlarmPlayFn] = None) -> None:
        self._play_fn = play_fn

    @property
    def name(self) -> str:
        return "trigger_alarm"

    @property
    def description(self) -> str:
        return "Declenchement d'une alarme sonore"

    async def run(self, params: dict[str, Any]) -> ToolResult:
        alarm_type = str(params.get("type") or "intrusion")
        duration = int(params.get("duration") or 10)
        duration = max(1, min(duration, 120))

        try:
            if self._play_fn is not None:
                await self._play_fn(alarm_type, duration)
            else:
                # Fallback non-bloquant: simuler la duree d'alarme.
                await asyncio.sleep(min(duration, 2))

            return ToolResult(
                tool_name=self.name,
                success=True,
                message="Alarme declenchee",
                data={"type": alarm_type, "duration": duration},
            )
        except Exception as exc:
            logger.error("AlarmTool failure: %s", exc, exc_info=True)
            return ToolResult(
                tool_name=self.name,
                success=False,
                message="Echec declenchement alarme",
                error=str(exc),
            )
