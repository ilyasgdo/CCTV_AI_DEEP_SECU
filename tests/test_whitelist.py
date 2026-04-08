"""Tests unitaires CRUD pour la whitelist faciale."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.face_manager import WhitelistRepository


def test_repository_initializes_files(tmp_path: Path) -> None:
    """Le repository cree la structure minimale au demarrage."""
    repo = WhitelistRepository(tmp_path / "whitelist")

    assert repo.registry_path.exists()
    assert repo.embeddings_dir.exists()
    assert repo.photos_dir.exists()
    assert repo.list_persons() == []


def test_add_and_get_person(tmp_path: Path) -> None:
    """Ajout et lecture d'une personne du registre."""
    repo = WhitelistRepository(tmp_path / "whitelist")

    person = {
        "id": "person_001",
        "name": "Ilyas",
        "role": "Owner",
        "access_level": "admin",
        "embeddings": ["ilyas_001.npy"],
        "photos": ["ilyas_ref_1.jpg"],
        "enrolled_at": "2026-04-08T12:00:00Z",
        "last_seen": None,
        "notes": "test",
    }

    repo.add_person(person)
    loaded = repo.get_person("person_001")

    assert loaded is not None
    assert loaded["name"] == "Ilyas"


def test_update_person(tmp_path: Path) -> None:
    """Mise a jour d'une personne existante."""
    repo = WhitelistRepository(tmp_path / "whitelist")
    repo.add_person(
        {
            "id": "person_002",
            "name": "Marie",
            "role": "Guest",
            "access_level": "user",
            "embeddings": [],
            "photos": [],
            "enrolled_at": "2026-04-08T12:00:00Z",
            "last_seen": None,
            "notes": "",
        }
    )

    ok = repo.update_person("person_002", {"last_seen": "2026-04-08T13:00:00Z"})

    assert ok is True
    assert repo.get_person("person_002")["last_seen"] == "2026-04-08T13:00:00Z"


def test_remove_person_also_removes_files(tmp_path: Path) -> None:
    """Suppression d'une personne avec nettoyage des fichiers."""
    repo = WhitelistRepository(tmp_path / "whitelist")

    emb = np.zeros((512,), dtype=np.float32)
    repo.save_embedding("person_003_001.npy", emb)
    photo = np.zeros((100, 100, 3), dtype=np.uint8)
    repo.save_photo("person_003_ref_1.jpg", photo)

    repo.add_person(
        {
            "id": "person_003",
            "name": "John",
            "role": "Guest",
            "access_level": "user",
            "embeddings": ["person_003_001.npy"],
            "photos": ["person_003_ref_1.jpg"],
            "enrolled_at": "2026-04-08T12:00:00Z",
            "last_seen": None,
            "notes": "",
        }
    )

    removed = repo.remove_person("person_003")

    assert removed is True
    assert repo.get_person("person_003") is None
    assert not (repo.embeddings_dir / "person_003_001.npy").exists()
    assert not (repo.photos_dir / "person_003_ref_1.jpg").exists()
