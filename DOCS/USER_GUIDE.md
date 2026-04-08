# User Guide - Sentinel-AI

## 1. Installation

### Windows

1. Ouvrir PowerShell dans le projet.
2. Lancer:
   - `scripts\install.ps1`
3. Activer l'environnement:
   - `.\venv\Scripts\Activate.ps1`

### Linux/Mac

1. Ouvrir un terminal dans le projet.
2. Rendre le script executable:
   - `chmod +x scripts/install.sh`
3. Lancer:
   - `./scripts/install.sh`

## 2. Premier demarrage

1. Copier et configurer `.env` (si absent, scripts d'install le creent).
2. Verifier les dependances:
   - `python scripts/run.py --check`
3. Demarrer:
   - `python scripts/run.py`
4. Ouvrir le dashboard:
   - `http://localhost:5000`

## 3. Enrolement whitelist

1. Utiliser l'API dashboard `/api/whitelist` (interface Whitelist).
2. Ajouter une personne avec nom/role/access.
3. Verifier la presence dans la liste.

## 4. Dashboard

- Live View: stream camera et evenements temps reel.
- Whitelist: gestion des personnes connues.
- Events: historique d'evenements.
- Settings: configuration runtime.
- Snapshots: captures inconnus.

Fonctions avancees:

- Zones: `GET/PUT /api/zones`
- Rapport journalier: `GET /api/report/daily`

## 5. Configuration

- Parametres systeme: `config/settings.yaml`
- Secrets et API keys: `.env`
- Reference complete: `DOCS/CONFIGURATION.md`

## 6. Depannage

- Voir `DOCS/TROUBLESHOOTING.md`
- Verifier les logs dans `logs/`
- Tester l'etat systeme via `/api/status` et `/api/monitoring`

## 7. FAQ

Q: Le dashboard retourne 401 ?
A: Verifier `DASHBOARD_AUTH_ENABLED`, `DASHBOARD_USERNAME`, `DASHBOARD_PASSWORD`.

Q: Le LLM est indisponible ?
A: Verifier Ollama et `llm.api_url`.

Q: Telegram ne fonctionne pas ?
A: Verifier `TELEGRAM_BOT_TOKEN` et `TELEGRAM_CHAT_ID`.
