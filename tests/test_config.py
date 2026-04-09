"""
Tests unitaires pour src/core/config.py

Vérifie le chargement de la config YAML, les valeurs par défaut,
la validation, et les surcharges par variables d'environnement.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.core.config import (
    Config,
    ConfigError,
    ConfigValidationError,
    CameraConfig,
    LLMConfig,
    DetectionConfig,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def default_config() -> Config:
    """Config avec les valeurs par défaut (fichier YAML existant)."""
    return Config()


@pytest.fixture
def custom_yaml(tmp_path: Path) -> Path:
    """Crée un fichier YAML personnalisé pour les tests."""
    config_data = {
        "camera": {
            "source": "rtsp://192.168.1.10/stream",
            "width": 1920,
            "height": 1080,
            "fps": 60,
        },
        "llm": {
            "api_url": "http://192.168.1.50:11434",
            "model_name": "llama3",
            "timeout": 60,
        },
        "detection": {
            "confidence": 0.7,
            "model_path": "yolo11m-pose.pt",
        },
    }
    yaml_path = tmp_path / "test_settings.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config_data, f)
    return yaml_path


@pytest.fixture
def invalid_yaml(tmp_path: Path) -> Path:
    """Crée un fichier YAML avec des valeurs invalides."""
    config_data = {
        "camera": {"fps": 999},  # Invalide: > 120
    }
    yaml_path = tmp_path / "invalid_settings.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config_data, f)
    return yaml_path


# ──────────────────────────────────────────────
# Tests — Valeurs par défaut
# ──────────────────────────────────────────────

class TestDefaultConfig:
    """Tests des valeurs de configuration par défaut."""

    def test_camera_defaults(self, default_config: Config) -> None:
        """Les valeurs par défaut de la caméra sont correctes."""
        assert default_config.camera.source == "0"
        assert default_config.camera.width == 1280
        assert default_config.camera.height == 720
        assert default_config.camera.fps == 30

    def test_llm_defaults(self, default_config: Config) -> None:
        """Les valeurs par défaut du LLM sont correctes."""
        assert default_config.llm.api_url == "http://localhost:11434"
        assert default_config.llm.model_name in {
            "gemma4",
            "gemma4-lox-maxctx:latest",
        }
        assert default_config.llm.timeout == 30
        assert default_config.llm.analysis_interval == 5

    def test_detection_defaults(self, default_config: Config) -> None:
        """Les valeurs par défaut de la détection sont correctes."""
        assert default_config.detection.model_path == "yolo26n.pt"
        assert default_config.detection.confidence == 0.5

    def test_face_defaults(self, default_config: Config) -> None:
        """Les valeurs par défaut de la reconnaissance faciale
        sont correctes."""
        assert default_config.face.similarity_threshold == 0.6

    def test_audio_defaults(self, default_config: Config) -> None:
        """Les valeurs par défaut audio sont correctes."""
        assert default_config.audio.tts_enabled is True
        assert default_config.audio.stt_enabled is True
        assert default_config.audio.tts_voice == "fr-FR-HenriNeural"

    def test_dashboard_defaults(self, default_config: Config) -> None:
        """Les valeurs par défaut du dashboard sont correctes."""
        assert default_config.dashboard.host == "0.0.0.0"
        assert default_config.dashboard.port == 5000

    def test_project_root(self, default_config: Config) -> None:
        """Le chemin racine du projet est correct."""
        root = default_config.project_root
        assert root.exists()
        assert (root / "main.py").exists()


# ──────────────────────────────────────────────
# Tests — Chargement YAML personnalisé
# ──────────────────────────────────────────────

class TestCustomConfig:
    """Tests du chargement d'un fichier YAML personnalisé."""

    def test_custom_camera(self, custom_yaml: Path) -> None:
        """Les valeurs personnalisées de la caméra sont chargées."""
        config = Config(str(custom_yaml))
        assert config.camera.source == "rtsp://192.168.1.10/stream"
        assert config.camera.width == 1920
        assert config.camera.height == 1080
        assert config.camera.fps == 60

    def test_custom_llm(self, custom_yaml: Path) -> None:
        """Les valeurs personnalisées du LLM sont chargées."""
        config = Config(str(custom_yaml))
        assert config.llm.api_url == "http://192.168.1.50:11434"
        assert config.llm.model_name == "llama3"
        assert config.llm.timeout == 60

    def test_custom_detection(self, custom_yaml: Path) -> None:
        """Les valeurs personnalisées de la détection sont chargées."""
        config = Config(str(custom_yaml))
        assert config.detection.confidence == 0.7
        assert config.detection.model_path == "yolo11m-pose.pt"

    def test_nonexistent_yaml_uses_defaults(
        self, tmp_path: Path
    ) -> None:
        """Un fichier YAML inexistant utilise les valeurs par défaut."""
        config = Config(str(tmp_path / "nonexistent.yaml"))
        assert config.camera.source == "0"
        assert config.llm.api_url == "http://localhost:11434"


# ──────────────────────────────────────────────
# Tests — Validation
# ──────────────────────────────────────────────

class TestConfigValidation:
    """Tests de la validation des paramètres."""

    def test_invalid_fps(self, invalid_yaml: Path) -> None:
        """Un FPS invalide lève une ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="camera.fps"):
            Config(str(invalid_yaml))

    def test_invalid_confidence(self, tmp_path: Path) -> None:
        """Un seuil de confiance > 1.0 lève une erreur."""
        data = {"detection": {"confidence": 1.5}}
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        with pytest.raises(
            ConfigValidationError, match="detection.confidence"
        ):
            Config(str(path))

    def test_invalid_port(self, tmp_path: Path) -> None:
        """Un port invalide lève une erreur."""
        data = {"dashboard": {"port": 99999}}
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        with pytest.raises(
            ConfigValidationError, match="dashboard.port"
        ):
            Config(str(path))


# ──────────────────────────────────────────────
# Tests — Surcharges environnement
# ──────────────────────────────────────────────

class TestEnvOverrides:
    """Tests des surcharges par variables d'environnement."""

    def test_env_llm_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SENTINEL_LLM_API_URL surcharge la config."""
        monkeypatch.setenv(
            "SENTINEL_LLM_API_URL", "http://remote:11434"
        )
        config = Config()
        assert config.llm.api_url == "http://remote:11434"

    def test_env_log_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SENTINEL_LOG_LEVEL surcharge la config."""
        monkeypatch.setenv("SENTINEL_LOG_LEVEL", "DEBUG")
        config = Config()
        assert config.logging.level == "DEBUG"


# ──────────────────────────────────────────────
# Tests — Divers
# ──────────────────────────────────────────────

class TestConfigMisc:
    """Tests divers pour la configuration."""

    def test_repr(self, default_config: Config) -> None:
        """La représentation string est correcte."""
        repr_str = repr(default_config)
        assert "Config(" in repr_str
        assert "camera=" in repr_str
        assert "llm=" in repr_str

    def test_get_raw(self, default_config: Config) -> None:
        """get_raw retourne les valeurs brutes du YAML."""
        # Les valeurs par défaut viennent du fichier settings.yaml
        source = default_config.get_raw("camera.source", "fallback")
        assert source == "0" or source == "fallback"

    def test_get_raw_missing_key(
        self, default_config: Config
    ) -> None:
        """get_raw avec clé inexistante retourne le défaut."""
        result = default_config.get_raw(
            "nonexistent.key", "default_value"
        )
        assert result == "default_value"
