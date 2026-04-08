# Configuration Reference - Sentinel-AI

## camera

- `source`: webcam index, RTSP URL, fichier video
- `width`, `height`, `fps`
- `buffer_size`
- `reconnect_delay`, `max_reconnect_attempts`

## llm

- `api_url`
- `model_name`
- `timeout`
- `analysis_interval`
- `max_retries`
- `temperature`
- `max_tokens`

## detection

- `model_path`
- `confidence`
- `iou_threshold`
- `input_size`
- `device`
- `skip_frames`

## face

- `similarity_threshold`
- `uncertain_threshold`
- `recalculate_interval`
- `margin_percent`
- `whitelist_dir`

Variable d'environnement associee:

- `SENTINEL_WHITELIST_ENCRYPTION_KEY`: cle Fernet optionnelle pour chiffrer
	les embeddings whitelist au repos.

## audio

- `tts_enabled`
- `stt_enabled`
- `tts_voice`
- `stt_model`
- `stt_language`
- `vad_enabled`
- `buffer_seconds`

## dashboard

- `host`
- `port`
- `debug`
- `secret_key`

## alerts

- `email_enabled`
- `email_to`
- `lingering_threshold`
- `clip_pre_seconds`
- `clip_post_seconds`

Variables d'environnement associees:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `SENTINEL_EXTERNAL_API_KEY`

## logging

- `level`
- `file_enabled`
- `log_dir`
- `max_file_size_mb`
- `backup_count`
