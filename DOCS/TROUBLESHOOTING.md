# Troubleshooting - Sentinel-AI

## Camera indisponible

Symptomes:
- stream vide
- `camera_disconnected` dans les logs

Actions:
1. Verifier `camera.source` dans `config/settings.yaml`
2. Tester la source avec OpenCV localement
3. Verifier droits d'acces webcam

## LLM indisponible

Symptomes:
- `llm_connected=false`
- mode degrade actif

Actions:
1. Verifier `llm.api_url`
2. Verifier que le modele est present dans Ollama
3. Lancer `python scripts/run.py --check`

## Dashboard inaccessible

Actions:
1. Verifier `dashboard.host` et `dashboard.port`
2. Verifier credentials dashboard
3. Verifier pare-feu local

## Erreurs 429 API

Cause: rate limiting atteint.

Actions:
1. Ralentir le polling client
2. Ajuster `DASHBOARD_RATE_LIMIT_PER_MIN`

## Audio non fonctionnel

Actions:
1. Lancer en mode sans audio: `--no-audio`
2. Verifier dependances `edge-tts` / micro
