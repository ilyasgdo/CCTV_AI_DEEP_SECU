"""
Module de logging centralisé pour Sentinel-AI.

Fournit un système de logging configurable avec :
- Double sortie : console (colorée) + fichier.
- Rotation automatique des fichiers (max 10MB, 5 backups).
- Format structuré : [TIMESTAMP] [MODULE] [LEVEL] Message.
- Un logger par module via get_logger(__name__).

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Démarrage du module")
    logger.warning("Attention: FPS bas")
    logger.error("Erreur de connexion", exc_info=True)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────

DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
DEFAULT_LOG_FILE = "sentinel.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5
DEFAULT_LEVEL = logging.INFO

# Format du log
LOG_FORMAT = (
    "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Couleurs ANSI pour la console
COLORS = {
    "DEBUG": "\033[36m",       # Cyan
    "INFO": "\033[32m",        # Vert
    "WARNING": "\033[33m",     # Jaune
    "ERROR": "\033[31m",       # Rouge
    "CRITICAL": "\033[41m",    # Fond rouge
    "RESET": "\033[0m",        # Reset
    "BOLD": "\033[1m",         # Gras
    "DIM": "\033[2m",          # Atténué
}


# ──────────────────────────────────────────────
# Custom Formatter avec couleurs console
# ──────────────────────────────────────────────

class ColoredFormatter(logging.Formatter):
    """Formatter qui ajoute des couleurs ANSI pour la console."""

    def __init__(self, fmt: str, datefmt: str) -> None:
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Formate un log record avec des couleurs ANSI.

        Args:
            record: Le log record à formater.

        Returns:
            Le message formaté avec couleurs.
        """
        # Sauvegarder les valeurs originales
        original_levelname = record.levelname
        original_name = record.name

        # Appliquer les couleurs
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        dim = COLORS["DIM"]

        record.levelname = f"{color}{record.levelname:>8}{reset}"
        record.name = f"{dim}{record.name}{reset}"

        # Formater
        result = super().format(record)

        # Restaurer les valeurs originales
        record.levelname = original_levelname
        record.name = original_name

        return result


# ──────────────────────────────────────────────
# Configuration root logger (appelée une seule fois)
# ──────────────────────────────────────────────

_initialized = False


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: str = DEFAULT_LOG_FILE,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    file_enabled: bool = True,
) -> None:
    """Configure le système de logging global.

    Cette fonction doit être appelée UNE SEULE FOIS au démarrage
    de l'application (dans main.py).

    Args:
        level: Niveau de log ("DEBUG", "INFO", "WARNING", "ERROR",
               "CRITICAL").
        log_dir: Répertoire pour les fichiers log. Si None, utilise
                 PROJECT_ROOT/logs.
        log_file: Nom du fichier log.
        max_bytes: Taille max d'un fichier log avant rotation.
        backup_count: Nombre de fichiers de backup à garder.
        file_enabled: Si True, écrit aussi dans un fichier.
    """
    global _initialized
    if _initialized:
        return

    # Niveau de log
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configurer le root logger
    root_logger = logging.getLogger("sentinel")
    root_logger.setLevel(log_level)

    # Éviter la propagation vers le root logger de Python
    root_logger.propagate = False

    # ── Handler Console (coloré) ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = ColoredFormatter(LOG_FORMAT, LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # ── Handler Fichier (avec rotation) ──
    if file_enabled:
        log_path = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        log_path.mkdir(parents=True, exist_ok=True)
        file_path = log_path / log_file

        file_handler = RotatingFileHandler(
            filename=str(file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Obtient un logger pour un module spécifique.

    Si le système de logging n'a pas été initialisé via
    setup_logging(), une configuration minimale par défaut
    est appliquée automatiquement.

    Args:
        name: Nom du module (utiliser __name__). Le préfixe 'src.'
              est automatiquement retiré pour plus de lisibilité.

    Returns:
        Instance logging.Logger configurée.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module démarré")
        [2026-04-08 12:00:00] [core.detector] [INFO] Module démarré
    """
    global _initialized
    if not _initialized:
        # Configuration minimale par défaut (console uniquement)
        setup_logging(level="INFO", file_enabled=False)

    # Nettoyer le nom du module pour la lisibilité
    clean_name = name.replace("src.", "").replace("__", "")
    if not clean_name:
        clean_name = "root"

    return logging.getLogger(f"sentinel.{clean_name}")


def reset_logging() -> None:
    """Réinitialise le système de logging.

    Utile principalement pour les tests unitaires.
    """
    global _initialized
    root_logger = logging.getLogger("sentinel")
    root_logger.handlers.clear()
    _initialized = False
