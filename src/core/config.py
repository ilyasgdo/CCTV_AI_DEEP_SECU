"""
Module de configuration centralisée pour Sentinel-AI.

Charge les paramètres depuis config/settings.yaml et les variables
d'environnement (.env). Fournit un accès typé à toute la configuration
via des objets imbriqués (dot-notation).

Usage:
    from src.core.config import Config
    config = Config()
    print(config.camera.source)    # "0"
    print(config.llm.api_url)     # "http://localhost:11434"
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ──────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────

class ConfigError(Exception):
    """Erreur liée à la configuration."""
    pass


class ConfigFileNotFoundError(ConfigError):
    """Fichier de configuration introuvable."""
    pass


class ConfigValidationError(ConfigError):
    """Erreur de validation d'un paramètre."""
    pass


# ──────────────────────────────────────────────
# Dataclasses de configuration
# ──────────────────────────────────────────────

@dataclass
class CameraConfig:
    """Configuration de la source vidéo."""
    source: str = "0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 30
    reconnect_delay: float = 2.0
    max_reconnect_attempts: int = 10


@dataclass
class LLMConfig:
    """Configuration du modèle de langage (Ollama / API)."""
    api_url: str = "http://localhost:11434"
    model_name: str = "gemma4"
    timeout: int = 30
    analysis_interval: int = 5
    max_retries: int = 3
    temperature: float = 0.3
    max_tokens: int = 1024
    num_ctx: int = 4096
    keep_alive: str = "15m"


@dataclass
class DetectionConfig:
    """Configuration de la détection d'objets YOLO."""
    model_path: str = "yolo26n.pt"
    confidence: float = 0.5
    iou_threshold: float = 0.45
    input_size: int = 640
    device: str = "auto"
    skip_frames: int = 1


@dataclass
class FaceConfig:
    """Configuration de la reconnaissance faciale."""
    similarity_threshold: float = 0.6
    uncertain_threshold: float = 0.4
    recalculate_interval: int = 30
    margin_percent: float = 0.2
    whitelist_dir: str = "data/whitelist"


@dataclass
class AudioConfig:
    """Configuration audio (TTS + STT)."""
    tts_enabled: bool = True
    stt_enabled: bool = True
    tts_voice: str = "fr-FR-HenriNeural"
    stt_model: str = "base"
    stt_language: str = "fr"
    vad_enabled: bool = True
    buffer_seconds: int = 10


@dataclass
class DashboardConfig:
    """Configuration du dashboard web."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    secret_key: str = "sentinel-ai-secret-change-me"


@dataclass
class AlertConfig:
    """Configuration des alertes."""
    email_enabled: bool = False
    email_to: str = ""
    lingering_threshold: int = 120
    clip_pre_seconds: int = 10
    clip_post_seconds: int = 15


@dataclass
class LoggingConfig:
    """Configuration du logging."""
    level: str = "INFO"
    file_enabled: bool = True
    log_dir: str = "logs"
    max_file_size_mb: int = 10
    backup_count: int = 5


# ──────────────────────────────────────────────
# Configuration principale
# ──────────────────────────────────────────────

# Chemin racine du projet (2 niveaux au-dessus de ce fichier)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class Config:
    """
    Configuration centralisée de Sentinel-AI.

    Charge la configuration depuis un fichier YAML et l'expose
    sous forme d'objets typés avec notation pointée.

    Args:
        config_path: Chemin vers le fichier settings.yaml.
                     Si None, utilise le chemin par défaut.

    Raises:
        ConfigFileNotFoundError: Si le fichier YAML n'existe pas.
        ConfigValidationError: Si un paramètre est invalide.

    Example:
        >>> config = Config()
        >>> config.camera.source
        '0'
        >>> config.llm.api_url
        'http://localhost:11434'
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._raw: dict[str, Any] = {}

        # Initialiser les sous-configs avec les valeurs par défaut
        self.camera = CameraConfig()
        self.llm = LLMConfig()
        self.detection = DetectionConfig()
        self.face = FaceConfig()
        self.audio = AudioConfig()
        self.dashboard = DashboardConfig()
        self.alerts = AlertConfig()
        self.logging = LoggingConfig()

        # Charger la configuration depuis le fichier YAML
        self._load()
        self._load_env_overrides()
        self._validate()

    def _load(self) -> None:
        """Charge la configuration depuis le fichier YAML."""
        if not self._config_path.exists():
            # Utiliser les valeurs par défaut si le fichier n'existe pas
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(
                f"Erreur de parsing YAML dans {self._config_path}: {e}"
            ) from e

        # Appliquer les valeurs du YAML aux dataclasses
        self._apply_section("camera", self.camera)
        self._apply_section("llm", self.llm)
        self._apply_section("detection", self.detection)
        self._apply_section("face", self.face)
        self._apply_section("audio", self.audio)
        self._apply_section("dashboard", self.dashboard)
        self._apply_section("alerts", self.alerts)
        self._apply_section("logging", self.logging)

    def _apply_section(
        self, section_name: str, target: object
    ) -> None:
        """Applique les valeurs d'une section YAML à un dataclass.

        Args:
            section_name: Nom de la section dans le YAML.
            target: Instance du dataclass cible.
        """
        section_data = self._raw.get(section_name, {})
        if not isinstance(section_data, dict):
            return

        for key, value in section_data.items():
            if hasattr(target, key):
                setattr(target, key, value)

    def _load_env_overrides(self) -> None:
        """Charge les surcharges depuis les variables d'environnement.

        Les variables d'environnement suivent le pattern:
        SENTINEL_<SECTION>_<KEY> (ex: SENTINEL_LLM_API_URL)
        """
        env_prefix = "SENTINEL_"

        env_mapping: dict[str, tuple[object, str]] = {
            f"{env_prefix}CAMERA_SOURCE": (self.camera, "source"),
            f"{env_prefix}LLM_API_URL": (self.llm, "api_url"),
            f"{env_prefix}LLM_MODEL_NAME": (self.llm, "model_name"),
            f"{env_prefix}LLM_TIMEOUT": (self.llm, "timeout"),
            f"{env_prefix}LLM_ANALYSIS_INTERVAL": (
                self.llm, "analysis_interval"
            ),
            f"{env_prefix}LLM_MAX_RETRIES": (self.llm, "max_retries"),
            f"{env_prefix}LLM_MAX_TOKENS": (self.llm, "max_tokens"),
            f"{env_prefix}LLM_NUM_CTX": (self.llm, "num_ctx"),
            f"{env_prefix}LLM_KEEP_ALIVE": (self.llm, "keep_alive"),
            f"{env_prefix}DETECTION_MODEL_PATH": (
                self.detection, "model_path"
            ),
            f"{env_prefix}DETECTION_CONFIDENCE": (
                self.detection, "confidence"
            ),
            f"{env_prefix}DETECTION_IOU_THRESHOLD": (
                self.detection, "iou_threshold"
            ),
            f"{env_prefix}DETECTION_INPUT_SIZE": (
                self.detection, "input_size"
            ),
            f"{env_prefix}DETECTION_SKIP_FRAMES": (
                self.detection, "skip_frames"
            ),
            f"{env_prefix}DASHBOARD_HOST": (self.dashboard, "host"),
            f"{env_prefix}DASHBOARD_PORT": (self.dashboard, "port"),
            f"{env_prefix}AUDIO_TTS_VOICE": (self.audio, "tts_voice"),
            f"{env_prefix}LOG_LEVEL": (self.logging, "level"),
        }

        for env_var, (target, attr) in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convertir au type attendu
                current = getattr(target, attr)
                try:
                    if isinstance(current, bool):
                        converted = value.lower() in ("true", "1", "yes")
                    elif isinstance(current, int):
                        converted = int(value)
                    elif isinstance(current, float):
                        converted = float(value)
                    else:
                        converted = value
                    setattr(target, attr, converted)
                except (ValueError, TypeError):
                    pass  # Garder la valeur par défaut si conversion échoue

    def _validate(self) -> None:
        """Valide les paramètres critiques.

        Raises:
            ConfigValidationError: Si un paramètre est invalide.
        """
        if self.camera.fps < 1 or self.camera.fps > 120:
            raise ConfigValidationError(
                f"camera.fps doit être entre 1 et 120, "
                f"reçu: {self.camera.fps}"
            )

        if self.detection.confidence < 0.0 or self.detection.confidence > 1.0:
            raise ConfigValidationError(
                f"detection.confidence doit être entre 0.0 et 1.0, "
                f"reçu: {self.detection.confidence}"
            )

        if self.face.similarity_threshold < 0.0 or \
                self.face.similarity_threshold > 1.0:
            raise ConfigValidationError(
                f"face.similarity_threshold doit être entre 0.0 et 1.0, "
                f"reçu: {self.face.similarity_threshold}"
            )

        if self.llm.timeout < 1:
            raise ConfigValidationError(
                f"llm.timeout doit être >= 1, reçu: {self.llm.timeout}"
            )

        if self.llm.analysis_interval < 1:
            raise ConfigValidationError(
                "llm.analysis_interval doit être >= 1, "
                f"reçu: {self.llm.analysis_interval}"
            )

        if self.llm.max_retries < 1:
            raise ConfigValidationError(
                f"llm.max_retries doit être >= 1, reçu: {self.llm.max_retries}"
            )

        if self.llm.max_tokens < 1:
            raise ConfigValidationError(
                f"llm.max_tokens doit être >= 1, reçu: {self.llm.max_tokens}"
            )

        if self.llm.num_ctx < 512:
            raise ConfigValidationError(
                f"llm.num_ctx doit être >= 512, reçu: {self.llm.num_ctx}"
            )

        if self.detection.input_size < 160:
            raise ConfigValidationError(
                "detection.input_size doit être >= 160, "
                f"reçu: {self.detection.input_size}"
            )

        if self.detection.skip_frames < 1:
            raise ConfigValidationError(
                "detection.skip_frames doit être >= 1, "
                f"reçu: {self.detection.skip_frames}"
            )

        if self.dashboard.port < 1 or self.dashboard.port > 65535:
            raise ConfigValidationError(
                f"dashboard.port doit être entre 1 et 65535, "
                f"reçu: {self.dashboard.port}"
            )

    @property
    def project_root(self) -> Path:
        """Retourne le chemin racine du projet."""
        return PROJECT_ROOT

    def get_raw(self, key: str, default: Any = None) -> Any:
        """Accède à une valeur brute du YAML par clé dotée.

        Args:
            key: Clé en notation pointée (ex: "camera.source").
            default: Valeur par défaut si la clé n'existe pas.

        Returns:
            La valeur trouvée ou la valeur par défaut.
        """
        keys = key.split(".")
        current = self._raw
        for k in keys:
            if isinstance(current, dict):
                current = current.get(k, default)
            else:
                return default
        return current

    def __repr__(self) -> str:
        return (
            f"Config("
            f"camera={self.camera}, "
            f"llm={self.llm}, "
            f"detection={self.detection})"
        )
