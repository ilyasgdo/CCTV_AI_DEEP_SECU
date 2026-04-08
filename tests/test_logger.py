"""
Tests unitaires pour src/utils/logger.py

Vérifie le système de logging centralisé : format, niveaux,
rotation de fichiers, et get_logger().
"""

import logging
from pathlib import Path

import pytest

from src.utils.logger import (
    get_logger,
    setup_logging,
    reset_logging,
    LOG_FORMAT,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_logging() -> None:
    """Réinitialise le logging avant chaque test."""
    reset_logging()
    yield
    reset_logging()


# ──────────────────────────────────────────────
# Tests — Initialisation
# ──────────────────────────────────────────────

class TestLoggingSetup:
    """Tests de l'initialisation du logging."""

    def test_setup_creates_handlers(self) -> None:
        """setup_logging crée les handlers console et fichier."""
        setup_logging(level="INFO", file_enabled=False)
        logger = logging.getLogger("sentinel")
        assert len(logger.handlers) >= 1  # Au moins la console

    def test_setup_with_file(self, tmp_path: Path) -> None:
        """setup_logging crée un fichier log si file_enabled=True."""
        setup_logging(
            level="INFO",
            log_dir=str(tmp_path),
            file_enabled=True,
        )
        logger = logging.getLogger("sentinel")
        assert len(logger.handlers) >= 2  # Console + fichier

    def test_setup_idempotent(self) -> None:
        """setup_logging appelé 2 fois ne duplique pas les handlers."""
        setup_logging(level="INFO", file_enabled=False)
        count1 = len(logging.getLogger("sentinel").handlers)

        setup_logging(level="DEBUG", file_enabled=False)
        count2 = len(logging.getLogger("sentinel").handlers)

        assert count1 == count2

    def test_setup_log_level(self) -> None:
        """Le niveau de log est correctement appliqué."""
        setup_logging(level="WARNING", file_enabled=False)
        logger = logging.getLogger("sentinel")
        assert logger.level == logging.WARNING


# ──────────────────────────────────────────────
# Tests — get_logger
# ──────────────────────────────────────────────

class TestGetLogger:
    """Tests de la fonction get_logger."""

    def test_returns_logger(self) -> None:
        """get_logger retourne une instance logging.Logger."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_cleaned(self) -> None:
        """Le préfixe 'src.' est retiré du nom du logger."""
        logger = get_logger("src.core.detector")
        assert "sentinel.core.detector" in logger.name

    def test_logger_name_without_prefix(self) -> None:
        """Un nom sans préfixe est gardé tel quel."""
        logger = get_logger("my_module")
        assert "sentinel.my_module" in logger.name

    def test_auto_init(self) -> None:
        """get_logger initialise le logging si pas encore fait."""
        logger = get_logger("auto_init_test")
        assert logger is not None
        # Doit avoir au moins un handler via le parent
        root = logging.getLogger("sentinel")
        assert len(root.handlers) >= 1

    def test_logger_can_log(self, capfd: pytest.CaptureFixture) -> None:
        """Le logger peut écrire des messages."""
        setup_logging(level="DEBUG", file_enabled=False)
        logger = get_logger("test_output")
        logger.info("Message de test pour la capture")
        captured = capfd.readouterr()
        assert "Message de test" in captured.out


# ──────────────────────────────────────────────
# Tests — Fichier log
# ──────────────────────────────────────────────

class TestLogFile:
    """Tests de l'écriture dans un fichier log."""

    def test_log_file_created(self, tmp_path: Path) -> None:
        """Un fichier log est créé dans le répertoire spécifié."""
        setup_logging(
            level="INFO",
            log_dir=str(tmp_path),
            log_file="test.log",
            file_enabled=True,
        )
        logger = get_logger("file_test")
        logger.info("Test d'écriture fichier")

        log_file = tmp_path / "test.log"
        assert log_file.exists()

    def test_log_file_content(self, tmp_path: Path) -> None:
        """Le fichier log contient les messages formatés."""
        setup_logging(
            level="INFO",
            log_dir=str(tmp_path),
            log_file="content_test.log",
            file_enabled=True,
        )
        logger = get_logger("content_module")
        logger.info("Contenu vérifiable")

        log_file = tmp_path / "content_test.log"
        content = log_file.read_text(encoding="utf-8")
        assert "Contenu vérifiable" in content
        assert "content_module" in content
        assert "INFO" in content


# ──────────────────────────────────────────────
# Tests — reset_logging
# ──────────────────────────────────────────────

class TestResetLogging:
    """Tests de la réinitialisation du logging."""

    def test_reset_clears_handlers(self) -> None:
        """reset_logging supprime tous les handlers."""
        setup_logging(level="INFO", file_enabled=False)
        root = logging.getLogger("sentinel")
        assert len(root.handlers) > 0

        reset_logging()
        assert len(root.handlers) == 0

    def test_reset_allows_reinit(self) -> None:
        """Après reset, on peut réinitialiser avec de nouvelles
        valeurs."""
        setup_logging(level="INFO", file_enabled=False)
        reset_logging()
        setup_logging(level="DEBUG", file_enabled=False)

        root = logging.getLogger("sentinel")
        assert root.level == logging.DEBUG
