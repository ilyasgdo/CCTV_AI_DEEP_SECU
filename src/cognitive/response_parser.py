"""Parseur robuste de reponses LLM (support des balises <think>)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

ALLOWED_ALERT_LEVELS = {"normal", "attention", "alerte", "critique"}


@dataclass
class ToolAction:
    """Action outil demandee par le LLM."""

    tool_name: str
    parametres: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResponse:
    """Reponse normalisee du pipeline cognitif."""

    action_vocale: str
    niveau_alerte: str
    outils_a_lancer: list[ToolAction]
    raw_thinking: Optional[str]
    parse_success: bool
    parse_errors: list[str]
    response_time_ms: float


class ResponseParser:
    """Parse la reponse brute du LLM en ActionResponse valide."""

    def parse(self, raw_response: str, response_time_ms: float = 0.0) -> ActionResponse:
        """Extrait la pensee interne et le JSON d'action.

        Args:
            raw_response: Reponse brute retournee par le LLM.
            response_time_ms: Latence de la requete LLM.

        Returns:
            ActionResponse validee et normalisee.
        """
        errors: list[str] = []
        thinking, cleaned = self._extract_thinking(raw_response)

        json_blocks = self._extract_json_blocks(cleaned)
        if not json_blocks:
            errors.append("Aucun bloc JSON detecte")
            return self._fallback_response(thinking, errors, response_time_ms)

        payload: dict[str, Any] = {}
        parsed = False

        for block in reversed(json_blocks):
            try:
                payload = json.loads(block)
                parsed = True
                break
            except json.JSONDecodeError as exc:
                errors.append(f"JSON invalide: {exc}")

        if not parsed:
            return self._fallback_response(thinking, errors, response_time_ms)

        action_vocale = payload.get("action_vocale")
        if not isinstance(action_vocale, str) or not action_vocale.strip():
            errors.append("Champ action_vocale manquant ou invalide")
            action_vocale = "Mode veille active. Analyse en cours."

        niveau = payload.get("niveau_alerte", "normal")
        if not isinstance(niveau, str) or niveau not in ALLOWED_ALERT_LEVELS:
            errors.append("Champ niveau_alerte invalide")
            niveau = "normal"

        tools = self._parse_tools(payload.get("outils_a_lancer"), errors)

        return ActionResponse(
            action_vocale=action_vocale.strip(),
            niveau_alerte=niveau,
            outils_a_lancer=tools,
            raw_thinking=thinking,
            parse_success=len(errors) == 0,
            parse_errors=errors,
            response_time_ms=response_time_ms,
        )

    def _parse_tools(self, raw_tools: Any, errors: list[str]) -> list[ToolAction]:
        """Valide et convertit la liste des actions outils."""
        if raw_tools is None:
            return []

        if not isinstance(raw_tools, list):
            errors.append("outils_a_lancer doit etre une liste")
            return []

        tools: list[ToolAction] = []
        for index, item in enumerate(raw_tools):
            if not isinstance(item, dict):
                errors.append(f"tool[{index}] invalide (dict attendu)")
                continue

            name = item.get("tool_name")
            params = item.get("parametres", {})
            if not isinstance(name, str) or not name.strip():
                errors.append(f"tool[{index}] sans tool_name valide")
                continue

            if not isinstance(params, dict):
                errors.append(f"tool[{index}] parametres invalides")
                params = {}

            tools.append(ToolAction(tool_name=name.strip(), parametres=params))

        return tools

    def _extract_thinking(self, text: str) -> tuple[Optional[str], str]:
        """Extrait tous les blocs <think>...</think> meme imparfaits."""
        if not text:
            return None, ""

        # Cas simple et frequent: extraction regex non-gourmande de blocs valides.
        blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Gestion de balises partiellement malformees.
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")

        if not blocks:
            return None, cleaned.strip()

        thinking = "\n\n".join(block.strip() for block in blocks if block.strip())
        return thinking or None, cleaned.strip()

    def _extract_json_blocks(self, text: str) -> list[str]:
        """Extrait les blocs JSON en analysant les accolades equilibrees."""
        blocks: list[str] = []
        in_string = False
        escape = False
        depth = 0
        start: Optional[int] = None

        for index, char in enumerate(text):
            if escape:
                escape = False
                continue

            if char == "\\":
                escape = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        blocks.append(text[start:index + 1])
                        start = None

        return blocks

    def _fallback_response(
        self,
        thinking: Optional[str],
        errors: list[str],
        response_time_ms: float,
    ) -> ActionResponse:
        """Construit une reponse de securite si le parsing echoue."""
        return ActionResponse(
            action_vocale="Je reste en surveillance silencieuse.",
            niveau_alerte="normal",
            outils_a_lancer=[],
            raw_thinking=thinking,
            parse_success=False,
            parse_errors=errors,
            response_time_ms=response_time_ms,
        )
