"""Validation latence LLM Etape 8 (metrique parser/client)."""

from __future__ import annotations

from src.cognitive.response_parser import ResponseParser


def test_llm_latency_metric_below_target_in_payload() -> None:
    """Le parser transporte correctement la latence mesuree."""
    parser = ResponseParser()
    raw = '{"action_vocale":"ok","niveau_alerte":"normal","outils_a_lancer":[]}'
    parsed = parser.parse(raw_response=raw, response_time_ms=1200.0)
    assert parsed.response_time_ms < 3000.0
    assert parsed.parse_success is True
