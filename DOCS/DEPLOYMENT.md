# Deployment Guide - Sentinel-AI

## Prerequis

- Python 3.10+
- Dependances installees: `pip install -r requirements.txt`
- Ollama lance localement ou endpoint distant accessible

## Lancement

```bash
python scripts/run.py
```

Options utiles:

```bash
python scripts/run.py --check
python scripts/run.py --no-audio
python scripts/run.py --no-dashboard
python scripts/run.py --camera 0
python scripts/run.py --llm-url http://localhost:11434
```

## Variables d'environnement importantes

- `DASHBOARD_AUTH_ENABLED`
- `DASHBOARD_USERNAME`
- `DASHBOARD_PASSWORD`
- `DASHBOARD_RATE_LIMIT_ENABLED`
- `DASHBOARD_RATE_LIMIT_PER_MIN`

## Recommandations securite production

- Activer HTTPS via reverse proxy (Nginx/Caddy)
- Isoler le dashboard sur reseau interne
- Utiliser des secrets forts dans `.env`
- Activer rotation des logs et sauvegardes regulieres
