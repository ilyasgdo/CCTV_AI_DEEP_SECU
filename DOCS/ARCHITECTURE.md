# Architecture Sentinel-AI

## Vue globale

Sentinel-AI est organise en 5 pipelines coordonnes:

1. Perception CPU: camera -> detection YOLO -> tracking -> reconnaissance faciale
2. Cognitif LLM: prompt contextuel -> requete LLM -> parsing JSON action
3. Audio: STT (ecoute) et TTS (reponse vocale)
4. Effecteurs: outils (email, alarme, snapshot, event log)
5. Dashboard: API REST + stream MJPEG + websocket temps reel

## Flux principal (cycle periodique)

1. Capture continue de frames dans un thread dedie
2. Detection et tracking en boucle non bloquante
3. Toutes les `analysis_interval` secondes: snapshot + contexte -> LLM
4. Parsing de la reponse et execution des actions
5. Publication des evenements vers dashboard

## Resilience et mode degrade

- LLM indisponible: surveillance silencieuse
- Audio indisponible: mode texte
- Camera instable: reconnexion automatique dans `Camera`
- Dashboard optionnel via runtime flag

## Monitoring

Le module `SystemMonitor` expose:

- CPU / RAM / RSS
- FPS courant
- nombre de personnes detectees
- latence et taux d'erreur LLM
- nombre total d'evenements
- uptime
