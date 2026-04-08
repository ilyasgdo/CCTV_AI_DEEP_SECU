# API Reference - Dashboard Sentinel-AI

## Pages

- `GET /`
- `GET /whitelist`
- `GET /events`
- `GET /settings`
- `GET /snapshots`

## API REST

- `GET /api/stream`: flux MJPEG live
- `GET /api/events`: liste paginee des evenements
- `GET /api/events/<id>`: detail evenement
- `GET /api/whitelist`: liste whitelist
- `POST /api/whitelist`: ajout whitelist
- `PUT /api/whitelist/<person_id>`: mise a jour whitelist
- `DELETE /api/whitelist/<person_id>`: suppression whitelist
- `GET /api/settings`: lecture configuration
- `PUT /api/settings`: mise a jour configuration
- `POST /api/settings/llm/test`: test connectivite LLM
- `GET /api/status`: etat runtime
- `GET /api/monitoring`: metriques detaillees
- `GET /api/snapshots`: galerie snapshots
- `GET /api/stats`: stats globales

## Securite

- Authentification basique activee par defaut (`DASHBOARD_AUTH_ENABLED=true`)
- Mot de passe via `DASHBOARD_PASSWORD`
- Rate limiting API configurable:
  - `DASHBOARD_RATE_LIMIT_ENABLED`
  - `DASHBOARD_RATE_LIMIT_PER_MIN`

## Erreurs communes

- `401`: non authentifie
- `400`: payload invalide
- `404`: ressource introuvable
- `429`: rate limit depasse
