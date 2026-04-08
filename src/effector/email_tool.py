"""Outil d'envoi d'emails d'alerte."""

from __future__ import annotations

import asyncio
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Optional

from src.core.config import Config
from src.effector.base_tool import BaseTool, ToolResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

SMTPFactory = Callable[..., smtplib.SMTP]


class EmailTool(BaseTool):
    """Envoie un email d'alerte avec piece jointe optionnelle."""

    def __init__(self, config: Config, smtp_factory: Optional[SMTPFactory] = None) -> None:
        self._config = config
        self._smtp_factory = smtp_factory or smtplib.SMTP

    @property
    def name(self) -> str:
        return "send_email"

    @property
    def description(self) -> str:
        return "Envoi d'un email d'alerte securite"

    async def run(self, params: dict[str, Any]) -> ToolResult:
        """Execute l'envoi d'email de maniere asynchrone."""
        try:
            return await asyncio.to_thread(self._send_email, params)
        except Exception as exc:
            logger.error("EmailTool failure: %s", exc, exc_info=True)
            return ToolResult(
                tool_name=self.name,
                success=False,
                message="Envoi email echoue",
                error=str(exc),
            )

    def _send_email(self, params: dict[str, Any]) -> ToolResult:
        subject = str(params.get("sujet") or "Alerte Sentinel-AI")
        urgency = str(params.get("urgence") or "moyenne")
        description = str(params.get("description") or "")
        attach_snapshot = bool(params.get("attach_snapshot", True))
        snapshot_path = params.get("snapshot_path")

        smtp_host = os.getenv("SMTP_HOST", "")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        to_addr = os.getenv("ALERT_EMAIL_TO", "")

        if not all([smtp_host, smtp_user, smtp_password, to_addr]):
            raise RuntimeError("SMTP non configure (variables .env manquantes)")

        msg = EmailMessage()
        msg["From"] = smtp_user
        msg["To"] = to_addr
        msg["Subject"] = f"[{urgency.upper()}] {subject}"

        body = [
            "Alerte Sentinel-AI",
            f"Urgence: {urgency}",
            "",
            description or "Aucune description fournie.",
        ]
        msg.set_content("\n".join(body))

        if attach_snapshot and snapshot_path:
            path = Path(str(snapshot_path))
            if path.exists():
                data = path.read_bytes()
                msg.add_attachment(
                    data,
                    maintype="image",
                    subtype="jpeg",
                    filename=path.name,
                )

        with self._smtp_factory(smtp_host, smtp_port, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(smtp_user, smtp_password)
            smtp.send_message(msg)

        return ToolResult(
            tool_name=self.name,
            success=True,
            message="Email envoye",
            data={"to": to_addr, "subject": subject, "urgence": urgency},
        )
