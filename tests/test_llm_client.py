"""Tests unitaires pour le client LLM agnostique."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.cognitive.llm_client import LLMClient
from src.core.config import Config


class DummyResponse:
    """Reponse HTTP factice pour les tests."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture
def config() -> Config:
    return Config()


def test_health_check_true(config: Config) -> None:
    """health_check retourne True si /api/tags repond."""

    def request_fn(method: str, url: str, json: Any, timeout: int) -> DummyResponse:
        assert method == "GET"
        assert url.endswith("/api/tags")
        return DummyResponse({"models": [{"name": "gemma4"}]})

    client = LLMClient(config, request_fn=request_fn)
    ok = asyncio.run(client.health_check())

    assert ok is True


def test_generate_text(config: Config) -> None:
    """generate retourne le texte brut de l'API."""

    def request_fn(method: str, url: str, json: Any, timeout: int) -> DummyResponse:
        assert method == "POST"
        assert url.endswith("/api/generate")
        assert json["prompt"].startswith("bonjour")
        return DummyResponse({"response": "ok"})

    client = LLMClient(config, request_fn=request_fn)
    response = asyncio.run(client.generate("bonjour test"))

    assert response == "ok"


def test_generate_multimodal_adds_images(config: Config) -> None:
    """generate ajoute le champ images pour une requete multimodale."""

    seen_payload: dict[str, Any] = {}

    def request_fn(method: str, url: str, json: Any, timeout: int) -> DummyResponse:
        nonlocal seen_payload
        seen_payload = json
        return DummyResponse({"response": "scene analysee"})

    client = LLMClient(config, request_fn=request_fn)
    result = asyncio.run(client.generate("analyse", image=b"jpeg-bytes"))

    assert result == "scene analysee"
    assert "images" in seen_payload
    assert isinstance(seen_payload["images"][0], str)


def test_retry_logic_then_success(config: Config) -> None:
    """Le client retry puis reussit a la seconde tentative."""
    calls = {"n": 0}

    def request_fn(method: str, url: str, json: Any, timeout: int) -> DummyResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("network down")
        return DummyResponse({"response": "ok-after-retry"})

    client = LLMClient(config, request_fn=request_fn)
    text = asyncio.run(client.generate("prompt"))

    assert text == "ok-after-retry"
    assert calls["n"] == 2


def test_list_models(config: Config) -> None:
    """list_models retourne la liste des noms exposes."""

    def request_fn(method: str, url: str, json: Any, timeout: int) -> DummyResponse:
        return DummyResponse({"models": [{"name": "gemma4"}, {"name": "llama3"}]})

    client = LLMClient(config, request_fn=request_fn)
    models = asyncio.run(client.list_models())

    assert models == ["gemma4", "llama3"]
