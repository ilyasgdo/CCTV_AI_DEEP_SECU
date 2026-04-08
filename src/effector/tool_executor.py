"""Dispatcher central d'execution des outils effecteurs."""

from __future__ import annotations

from typing import Any, Optional

from src.cognitive.response_parser import ToolAction
from src.core.config import Config
from src.effector.alarm_tool import AlarmTool
from src.effector.base_tool import BaseTool, ToolResult
from src.effector.email_tool import EmailTool
from src.effector.event_log_tool import EventLogTool
from src.effector.snapshot_tool import SnapshotTool
from src.effector.telegram_tool import TelegramTool
from src.utils.event_bus import EventBus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnnounceTool(BaseTool):
    """Outil d'annonce vocale via TTSEngine."""

    def __init__(self, tts_engine: Optional[object]) -> None:
        self._tts_engine = tts_engine

    @property
    def name(self) -> str:
        return "announce"

    @property
    def description(self) -> str:
        return "Prononce un message via le moteur TTS"

    async def run(self, params: dict[str, Any]) -> ToolResult:
        text = str(params.get("text") or "")
        if not text:
            return ToolResult(
                tool_name=self.name,
                success=False,
                message="Annonce vide",
                error="missing text",
            )

        if self._tts_engine is None or not hasattr(self._tts_engine, "speak"):
            return ToolResult(
                tool_name=self.name,
                success=False,
                message="TTS non disponible",
                error="tts unavailable",
            )

        await self._tts_engine.speak(text)
        return ToolResult(
            tool_name=self.name,
            success=True,
            message="Annonce en file d'attente",
            data={"text": text},
        )


class ToolExecutor:
    """Execute les actions d'outils demandees par le LLM."""

    def __init__(
        self,
        config: Config,
        event_bus: Optional[EventBus] = None,
        camera: Optional[object] = None,
        tts_engine: Optional[object] = None,
    ) -> None:
        self._event_bus = event_bus
        self.tools: dict[str, BaseTool] = {}
        self._register_tools(config, camera, tts_engine)

    def _register_tools(
        self,
        config: Config,
        camera: Optional[object],
        tts_engine: Optional[object],
    ) -> None:
        """Enregistre tous les outils disponibles."""
        self.tools["send_email"] = EmailTool(config)
        self.tools["trigger_alarm"] = AlarmTool()
        self.tools["save_snapshot"] = SnapshotTool(config, camera=camera)
        self.tools["log_event"] = EventLogTool(config)
        self.tools["send_telegram"] = TelegramTool()
        self.tools["announce"] = AnnounceTool(tts_engine)

    async def execute(self, actions: list[ToolAction | dict[str, Any]]) -> list[ToolResult]:
        """Execute des actions de maniere sequentielle et robuste."""
        results: list[ToolResult] = []

        for action in actions:
            if isinstance(action, dict):
                tool_name = str(action.get("tool_name") or "")
                params = action.get("parametres") or {}
            else:
                tool_name = action.tool_name
                params = action.parametres

            tool = self.tools.get(tool_name)
            if tool is None:
                result = ToolResult(
                    tool_name=tool_name,
                    success=False,
                    message="Outil inconnu",
                    error=f"Unknown tool: {tool_name}",
                )
                logger.error("Tool inconnu: %s", tool_name)
            else:
                try:
                    result = await tool.run(params)
                except Exception as exc:
                    logger.error("Tool execution failure (%s): %s", tool_name, exc, exc_info=True)
                    result = ToolResult(
                        tool_name=tool_name,
                        success=False,
                        message="Execution outil echouee",
                        error=str(exc),
                    )

            results.append(result)
            if self._event_bus:
                self._event_bus.emit("tool_executed", result.to_dict())

        return results
