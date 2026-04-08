# Fonctionnalites Avancees - Etape 09

## Scope retenu (sprint Etape 9)

Conformement au plan, 4 fonctionnalites avancees ont ete implementees sans
sur-implementation:

1. Enregistrement de clips d'alerte
2. Rapports PDF journaliers
3. Zones de surveillance configurables
4. Notifications Telegram

Une API externe v1 minimale securisee a egalement ete ajoutee.

## 1) Clips d'alerte

Module: `src/core/alert_clip_recorder.py`

- Buffer circulaire des frames pre-alerte
- Capture des frames post-alerte
- Encodage MP4 asynchrone en thread dedie
- Fichiers sauvegardes dans `data/clips/`

Trigger: evenements `face_unknown`, `entity_lingering`, `zone_intrusion`,
et alertes LLM de niveau `alerte|critique`.

## 2) Rapports PDF

Module: `src/core/report_generator.py`

- Generation d'un PDF journalier a partir de `data/event_log.json`
- Statistiques: total events, connus/inconnus, alertes
- Liste des evenements recents
- Sortie dans `data/reports/REPORT_YYYY-MM-DD.pdf`

Endpoint dashboard:

- `GET /api/report/daily?date=YYYY-MM-DD`

## 3) Zones de surveillance

Module: `src/core/surveillance_zones.py`

- Stockage/chargement de zones polygonales JSON
- Types supportes pour le sprint: `intrusion`
- Detection par point-in-polygon sur centre des entites trackees
- Emission d'evenements `zone_intrusion`

Endpoints dashboard:

- `GET /api/zones`
- `PUT /api/zones`

## 4) Notifications Telegram

Module: `src/effector/telegram_tool.py`

- Outil `send_telegram` enregistre dans `ToolExecutor`
- Envoi via Bot API Telegram
- Variables requises:
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID`

## API externe v1

Endpoints securises par header `X-API-Key`:

- `POST /api/v1/webhook/alert`
- `GET /api/v1/cameras`
- `GET /api/v1/stream/<cam_id>`
- `GET /api/v1/report/daily`

Variable requise:

- `SENTINEL_EXTERNAL_API_KEY`

## Validation tests

- Tests unitaires nouveaux:
  - `tests/test_alert_clip_recorder.py`
  - `tests/test_report_generator.py`
  - `tests/test_surveillance_zones.py`
- Extensions tests existants:
  - `tests/test_dashboard_api.py`
  - `tests/test_tools.py`
- Validation complete du projet: `177 passed`
