"""Outil de capture snapshot depuis la camera courante."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2

from src.core.config import Config
from src.effector.base_tool import BaseTool, ToolResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SnapshotTool(BaseTool):
    """Sauvegarde un snapshot horodate dans data/snapshots/."""

    def __init__(self, config: Config, camera: Optional[object] = None) -> None:
        self._config = config
        self._camera = camera

    @property
    def name(self) -> str:
        return "save_snapshot"

    @property
    def description(self) -> str:
        return "Capture un snapshot de la camera"

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            path = await asyncio.to_thread(self._save_snapshot, params)
            return ToolResult(
                tool_name=self.name,
                success=True,
                message="Snapshot sauvegarde",
                data={"snapshot_path": str(path)},
            )
        except Exception as exc:
            logger.error("SnapshotTool failure: %s", exc, exc_info=True)
            return ToolResult(
                tool_name=self.name,
                success=False,
                message="Echec capture snapshot",
                error=str(exc),
            )

    def _save_snapshot(self, params: dict[str, Any]) -> Path:
        frame = params.get("frame")
        if frame is None and self._camera and hasattr(self._camera, "get_frame"):
            frame = self._camera.get_frame()

        if frame is None:
            raise RuntimeError("Aucune frame disponible pour snapshot")

        snapshots_dir = self._config.project_root / "data" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        reason = str(params.get("reason") or "event")
        filename = f"snapshot_{reason}_{stamp}.jpg"
        path = snapshots_dir / filename

        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise RuntimeError("Ecriture snapshot impossible")

        return path
