"""Client LLM agnostique pour Sentinel-AI (Ollama/API compatible)."""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests

from src.core.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

RequestFn = Callable[..., requests.Response]


class LLMClientError(Exception):
    """Erreur de communication avec le service LLM."""


@dataclass
class LLMMetrics:
    """Metriques de fonctionnement du client LLM."""

    total_requests: int = 0
    success_requests: int = 0
    failed_requests: int = 0
    avg_response_ms: float = 0.0
    last_response_ms: float = 0.0
    last_error: str = ""


class LLMClient:
    """Client HTTP asynchrone agnostique pour API LLM.

    Args:
        config: Configuration globale Sentinel-AI.
        request_fn: Fonction HTTP injectable pour les tests.
    """

    def __init__(self, config: Config, request_fn: Optional[RequestFn] = None) -> None:
        self.api_url = config.llm.api_url.rstrip("/")
        self.model = config.llm.model_name
        self.timeout = int(config.llm.timeout)
        self.max_retries = int(config.llm.max_retries)
        self.temperature = float(config.llm.temperature)
        self.max_tokens = int(config.llm.max_tokens)

        self._request_fn = request_fn or requests.request
        self._metrics = LLMMetrics()
        self._response_times: list[float] = []

    async def _request_json(
        self,
        method: str,
        endpoint: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute une requete HTTP JSON avec retries/backoff."""
        url = f"{self.api_url}{endpoint}"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            self._metrics.total_requests += 1
            t0 = time.perf_counter()

            try:
                logger.debug("Requete LLM %s %s (attempt=%s)", method, url, attempt)
                response = await asyncio.to_thread(
                    self._request_fn,
                    method,
                    url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._record_success(elapsed_ms)
                return data

            except Exception as exc:
                last_exc = exc
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._record_failure(elapsed_ms, str(exc))

                logger.warning(
                    "Erreur LLM attempt=%s/%s endpoint=%s err=%s",
                    attempt,
                    self.max_retries,
                    endpoint,
                    exc,
                )

                if attempt < self.max_retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 5))

        raise LLMClientError(f"Echec requete LLM sur {endpoint}: {last_exc}") from last_exc

    def _record_success(self, elapsed_ms: float) -> None:
        """Met a jour les metriques de succes."""
        self._metrics.success_requests += 1
        self._metrics.last_response_ms = elapsed_ms
        self._response_times.append(elapsed_ms)
        if len(self._response_times) > 100:
            self._response_times = self._response_times[-100:]
        self._metrics.avg_response_ms = sum(self._response_times) / len(self._response_times)

    def _record_failure(self, elapsed_ms: float, error: str) -> None:
        """Met a jour les metriques d'erreur."""
        self._metrics.failed_requests += 1
        self._metrics.last_response_ms = elapsed_ms
        self._metrics.last_error = error

    async def generate(self, prompt: str, image: Optional[bytes] = None) -> str:
        """Envoie un prompt au LLM et retourne la reponse brute.

        Args:
            prompt: Prompt complet (system + contexte).
            image: Image JPEG optionnelle (multimodal).

        Returns:
            Texte brut retourne par l'API.

        Raises:
            LLMClientError: En cas d'echec total apres retries.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_p": 0.9,
            },
        }

        if image is not None:
            payload["images"] = [base64.b64encode(image).decode("utf-8")]

        data = await self._request_json("POST", "/api/generate", payload)

        if isinstance(data.get("response"), str):
            return data["response"]

        if isinstance(data.get("message"), dict):
            content = data["message"].get("content", "")
            if isinstance(content, str):
                return content

        raise LLMClientError("Reponse LLM invalide: champ texte introuvable")

    async def health_check(self) -> bool:
        """Verifie que l'API LLM est accessible."""
        try:
            await self._request_json("GET", "/api/tags", None)
            return True
        except LLMClientError:
            return False

    async def list_models(self) -> list[str]:
        """Liste les modeles disponibles via /api/tags."""
        data = await self._request_json("GET", "/api/tags", None)
        models = data.get("models", [])

        names: list[str] = []
        if isinstance(models, list):
            for model in models:
                if isinstance(model, dict) and isinstance(model.get("name"), str):
                    names.append(model["name"])

        return names

    async def show_model(self, model_name: Optional[str] = None) -> dict[str, Any]:
        """Retourne les details d'un modele via /api/show."""
        payload = {"name": model_name or self.model}
        return await self._request_json("POST", "/api/show", payload)

    @property
    def metrics(self) -> LLMMetrics:
        """Expose les metriques du client LLM."""
        return self._metrics
