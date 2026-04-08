"""Tests unitaires pour le parseur de reponses LLM."""

from __future__ import annotations

from src.cognitive.response_parser import ResponseParser


def test_parse_thinking_and_json() -> None:
    """Le parser extrait <think> et le JSON principal."""
    parser = ResponseParser()
    raw = (
        "<think>analyse interne</think>\n"
        "{\"action_vocale\":\"Bonjour\",\"niveau_alerte\":\"attention\","
        "\"outils_a_lancer\":[{\"tool_name\":\"alarm\",\"parametres\":{\"x\":1}}]}"
    )

    result = parser.parse(raw, response_time_ms=12.5)

    assert result.raw_thinking == "analyse interne"
    assert result.action_vocale == "Bonjour"
    assert result.niveau_alerte == "attention"
    assert len(result.outils_a_lancer) == 1
    assert result.outils_a_lancer[0].tool_name == "alarm"


def test_parse_without_thinking() -> None:
    """Le parser fonctionne sans balises think."""
    parser = ResponseParser()
    raw = '{"action_vocale":"RAS","niveau_alerte":"normal","outils_a_lancer":[]}'

    result = parser.parse(raw)

    assert result.raw_thinking is None
    assert result.parse_success is True


def test_parse_malformed_json_returns_fallback() -> None:
    """JSON casse -> fallback securite + parse_success False."""
    parser = ResponseParser()
    raw = "<think>abc</think>{action_vocale: Bonjour}"

    result = parser.parse(raw)

    assert result.parse_success is False
    assert result.action_vocale.startswith("Je reste")
    assert len(result.parse_errors) >= 1


def test_parse_multiple_json_blocks_keeps_last() -> None:
    """Si plusieurs JSON existent, le dernier est retenu."""
    parser = ResponseParser()
    raw = (
        '{"action_vocale":"ancien","niveau_alerte":"normal","outils_a_lancer":[]}'
        " bruit "
        '{"action_vocale":"nouveau","niveau_alerte":"alerte","outils_a_lancer":[]}'
    )

    result = parser.parse(raw)

    assert result.action_vocale == "nouveau"
    assert result.niveau_alerte == "alerte"


def test_parse_missing_fields_uses_defaults() -> None:
    """Champs manquants -> valeurs par defaut + erreurs collecte."""
    parser = ResponseParser()
    raw = '{"niveau_alerte":"invalide"}'

    result = parser.parse(raw)

    assert result.action_vocale
    assert result.niveau_alerte == "normal"
    assert result.parse_success is False
    assert len(result.parse_errors) >= 1
