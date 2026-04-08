"""Gestion du system prompt et construction du contexte d'analyse."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.core.config import Config
from src.core.detector import Detection
from src.core.tracker import TrackedEntity

DEFAULT_PROMPT_PATH = Path("config") / "prompts" / "sentinel_system.txt"


class PromptManager:
    """Construit les prompts cognitifs pour Sentinel-AI.

    Args:
        config: Configuration globale Sentinel-AI.
        system_prompt_path: Chemin optionnel du system prompt.
    """

    def __init__(
        self,
        config: Config,
        system_prompt_path: Optional[str | Path] = None,
    ) -> None:
        self._config = config
        project_root = config.project_root
        path = Path(system_prompt_path) if system_prompt_path else DEFAULT_PROMPT_PATH
        self._system_prompt_path = path if path.is_absolute() else project_root / path

    def load_system_prompt(self) -> str:
        """Charge le system prompt depuis le fichier configure."""
        if not self._system_prompt_path.exists():
            return "Tu es Sentinel-AI, agent de securite autonome."

        return self._system_prompt_path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _format_detected_objects(detections: list[Detection]) -> str:
        """Formate le resume des objets detectes."""
        if not detections:
            return "- Aucun objet detecte"

        class_counter = Counter(d.class_name for d in detections)
        lines = [f"- {name}: {count}" for name, count in class_counter.items()]
        return "\n".join(lines)

    @staticmethod
    def _format_identity_block(entities: list[TrackedEntity]) -> str:
        """Formate la section d'identification des personnes trackees."""
        if not entities:
            return "- Aucune entite trackee"

        lines: list[str] = []
        for entity in entities:
            if not entity.is_person:
                continue

            if entity.face_status == "known":
                lines.append(
                    f"- Personne #{entity.track_id}: CONNU - {entity.face_name} "
                    f"(score={entity.face_confidence:.2f})"
                )
            elif entity.face_status == "uncertain":
                lines.append(
                    f"- Personne #{entity.track_id}: INCERTAIN - {entity.face_name or 'N/A'} "
                    f"(score={entity.face_confidence:.2f})"
                )
            else:
                lines.append(
                    f"- Personne #{entity.track_id}: INCONNU - "
                    f"presence={entity.duration_seconds:.1f}s"
                )

        return "\n".join(lines) if lines else "- Aucune personne detectee"

    def build_analysis_prompt(
        self,
        detections: list[Detection],
        tracked_entities: list[TrackedEntity],
        audio_transcript: Optional[str],
        previous_context: Optional[str],
    ) -> str:
        """Construit le prompt complet a envoyer au LLM.

        Args:
            detections: Objets detectes dans la scene.
            tracked_entities: Entites trackees enrichies (face status).
            audio_transcript: Transcription audio optionnelle.
            previous_context: Resume conversationnel precedent.

        Returns:
            Prompt final (system prompt + rapport capteurs).
        """
        system_prompt = self.load_system_prompt()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sensor_report = [
            f"[RAPPORT CAPTEURS - {now}]",
            "",
            "OBJETS DETECTES:",
            self._format_detected_objects(detections),
            "",
            "IDENTIFICATION:",
            self._format_identity_block(tracked_entities),
            "",
            "CONTEXTE AUDITIF:",
            (audio_transcript or "Aucune transcription audio."),
            "",
            "CONTEXTE PRECEDENT:",
            (previous_context or "Aucun contexte precedent."),
        ]

        return (
            f"{system_prompt}\n\n"
            + "\n".join(sensor_report).strip()
        )
