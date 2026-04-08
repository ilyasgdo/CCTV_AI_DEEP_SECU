"""Memoire conversationnelle glissante pour le pipeline cognitif."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque


@dataclass
class MemoryEntry:
    """Echange conserve en memoire conversationnelle.

    Attributes:
        timestamp: Horodatage ISO8601 UTC.
        context: Resume du contexte capteurs envoye au LLM.
        response: Resume de reponse du LLM.
        alert_level: Niveau d'alerte associe.
    """

    timestamp: str
    context: str
    response: str
    alert_level: str


class ConversationMemory:
    """Memoire glissante des interactions recentes.

    Args:
        max_entries: Nombre maximal d'entrees conservees.
    """

    def __init__(self, max_entries: int = 10) -> None:
        self._history: Deque[MemoryEntry] = deque(maxlen=max_entries)

    @staticmethod
    def _now_iso() -> str:
        """Retourne l'horodatage courant au format ISO UTC."""
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    def add_exchange(self, context: str, response: str, alert_level: str) -> None:
        """Ajoute un echange contexte/reponse en memoire.

        Args:
            context: Contexte envoye au LLM.
            response: Reponse vocale ou synthese d'action.
            alert_level: Niveau d'alerte (normal, attention, alerte, critique).
        """
        self._history.append(
            MemoryEntry(
                timestamp=self._now_iso(),
                context=context.strip(),
                response=response.strip(),
                alert_level=alert_level.strip() or "normal",
            )
        )

    def get_context_summary(self) -> str:
        """Construit un resume texte des interactions recentes.

        Returns:
            Bloc texte adapte au prompt LLM.
        """
        if not self._history:
            return "Aucun historique recent."

        lines: list[str] = []
        for index, entry in enumerate(self._history, start=1):
            lines.append(
                f"{index}. [{entry.timestamp}] Niveau={entry.alert_level} | "
                f"Contexte={entry.context[:160]} | Reponse={entry.response[:160]}"
            )

        return "\n".join(lines)

    def clear(self) -> None:
        """Vide l'historique conversationnel."""
        self._history.clear()

    @property
    def size(self) -> int:
        """Nombre d'entrees presentes en memoire."""
        return len(self._history)
