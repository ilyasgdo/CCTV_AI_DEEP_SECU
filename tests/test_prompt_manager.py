"""Tests unitaires pour le gestionnaire de prompts cognitifs."""

from __future__ import annotations

import time
from pathlib import Path

from src.cognitive.prompt_manager import PromptManager
from src.core.config import Config
from src.core.detector import Detection
from src.core.tracker import TrackedEntity


def make_detection(class_name: str, class_id: int) -> Detection:
    """Construit une detection de test."""
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=0.9,
        bbox=(10, 10, 100, 100),
        center=(55, 55),
        frame_id=1,
        timestamp=time.time(),
    )


def make_entity_known(track_id: int) -> TrackedEntity:
    """Construit une entite connue pour test prompt."""
    entity = TrackedEntity(track_id=track_id)
    entity.add_detection(make_detection("person", 0))
    entity.face_status = "known"
    entity.face_name = "Ilyas"
    entity.face_confidence = 0.87
    return entity


def test_load_system_prompt_from_file(config_tmp: Path) -> None:
    """Le prompt manager charge bien le fichier de system prompt."""
    cfg = Config()
    prompt_file = config_tmp / "sentinel_system.txt"
    prompt_file.write_text("SYSTEM TEST", encoding="utf-8")

    manager = PromptManager(cfg, system_prompt_path=prompt_file)

    assert manager.load_system_prompt() == "SYSTEM TEST"


def test_build_analysis_prompt_contains_sections(tmp_path: Path) -> None:
    """Le prompt final contient les sections structurantes attendues."""
    cfg = Config()
    prompt_file = tmp_path / "sentinel_system.txt"
    prompt_file.write_text("SYSTEM HEADER", encoding="utf-8")

    manager = PromptManager(cfg, system_prompt_path=prompt_file)

    dets = [make_detection("person", 0), make_detection("backpack", 24)]
    entities = [make_entity_known(1)]

    text = manager.build_analysis_prompt(
        detections=dets,
        tracked_entities=entities,
        audio_transcript="Bonjour je suis livreur",
        previous_context="Interaction precedente courte",
    )

    assert "SYSTEM HEADER" in text
    assert "OBJETS DETECTES" in text
    assert "IDENTIFICATION" in text
    assert "CONTEXTE AUDITIF" in text
    assert "CONTEXTE PRECEDENT" in text
    assert "Ilyas" in text


def test_build_analysis_prompt_defaults_when_empty(tmp_path: Path) -> None:
    """Le prompt gere correctement les champs absents/vides."""
    cfg = Config()
    prompt_file = tmp_path / "sentinel_system.txt"
    prompt_file.write_text("SYSTEM", encoding="utf-8")
    manager = PromptManager(cfg, system_prompt_path=prompt_file)

    text = manager.build_analysis_prompt([], [], None, None)

    assert "Aucun objet detecte" in text
    assert "Aucun contexte precedent" in text


# Fixture locale pour eviter impact global sur config project
import pytest


@pytest.fixture
def config_tmp(tmp_path: Path) -> Path:
    return tmp_path
