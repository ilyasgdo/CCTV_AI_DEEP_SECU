"""Notification Telegram pour alertes Sentinel-AI (Etape 9)."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import requests

from src.effector.base_tool import BaseTool, ToolResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TelegramTool(BaseTool):
    """Envoie un message Telegram via Bot API."""

    def __init__(self, request_fn: Any = requests.post) -> None:
        self._request_fn = request_fn

    @property
    def name(self) -> str:
        return "send_telegram"

    @property
    def description(self) -> str:
        return "Envoie une notification Telegram"

    async def run(self, params: dict[str, Any]) -> ToolResult:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            return ToolResult(
                tool_name=self.name,
                success=False,
                message="Telegram non configure",
                error="missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID",
            )

        text = str(params.get("message") or params.get("text") or "Alerte Sentinel-AI")
        endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}

        try:
            response = await asyncio.to_thread(self._request_fn, endpoint, json=payload, timeout=8)
            response.raise_for_status()
            return ToolResult(
                tool_name=self.name,
                success=True,
                message="Notification Telegram envoyee",
                data={"chat_id": chat_id},
            )
        except Exception as exc:
            logger.error("TelegramTool failure: %s", exc, exc_info=True)
            return ToolResult(
                tool_name=self.name,
                success=False,
                message="Echec envoi Telegram",
                error=str(exc),
            )
