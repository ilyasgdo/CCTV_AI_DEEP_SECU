"""Interfaces et structures de base pour les outils effecteurs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Resultat standard d'execution d'un outil.

    Attributes:
        tool_name: Nom de l'outil execute.
        success: Indique si l'execution a reussi.
        message: Message humain resumant le resultat.
        data: Donnees complementaires de sortie.
        error: Description d'erreur en cas d'echec.
    """

    tool_name: str
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convertit le resultat en dictionnaire serialisable."""
        return asdict(self)


class BaseTool(ABC):
    """Interface de base pour tous les outils Sentinel-AI."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom unique de l'outil."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description fonctionnelle de l'outil."""

    @abstractmethod
    async def run(self, params: dict[str, Any]) -> ToolResult:
        """Execute l'outil avec ses parametres."""
