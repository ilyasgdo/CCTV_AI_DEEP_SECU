"""
CCTV AI DEEP SECU — Agent de Sécurité IA Conversationnel

Pipeline : Caméra (détections) + Micro (STT Whisper) → Ollama LLM → TTS Edge-TTS

L'agent observe la scène via les détections YOLO, écoute le micro,
pose des questions ou donne des ordres vocalement, comme un vrai
agent de sécurité posté dans l'entrée.
"""
import cv2
import time
import json
import threading
import tempfile
import wave
import subprocess
import asyncio
import os
import platform
import numpy as np
from typing import Optional, Dict, List

# Voix Edge-TTS — voix neuronale française professionnelle
EDGE_TTS_VOICE = "fr-FR-HenriNeural"  # Homme, autoritaire, professionnel


# ─── Configuration par défaut (peut être surchargée via config.py) ───
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_LISTEN_DURATION = 5       # secondes d'écoute micro
DEFAULT_TRIGGER_INTERVAL = 15     # secondes min entre interventions
DEFAULT_STT_MODEL = "base"        # whisper model size

SYSTEM_PROMPT = """Tu es un agent de sécurité IA posté à l'entrée d'un bâtiment surveillé par caméra.
Tu observes la scène via les détections de la caméra et tu communiques vocalement avec les personnes présentes.
Tu reçois un rapport détaillé comprenant une ANALYSE VISUELLE GLOBALE de la scène générée par un modèle de vision.
LA DESCRIPTION DE LA CAMÉRA VISION EST TA SOURCE DE VÉRITÉ PRINCIPALE.

RÈGLES STRICTES :
- Réponds TOUJOURS en français, 1 à 2 phrases MAXIMUM.
- Sois direct, professionnel et autoritaire quand nécessaire.
- PRÊTE UNE TRÈS GRANDE ATTENTION À LA DESCRIPTION VISUELLE. Si l'analyse visuelle mentionne un objet, c'est la vérité absolue.
- Si tu vois une arme ou un objet dangereux → ordonne de le poser immédiatement.
- Si une personne est inconnue depuis longtemps → demande-lui de s'identifier.
- Si quelqu'un court ou tombe → réagis rapidement (alerte, aide).
- Si une personne CONNUE lève les mains → c'est un signal amical pour te parler. Sois cordial et demande ce qu'elle souhaite.
- Si une personne INCONNUE lève les mains → c'est potentiellement une menace, demande-lui de s'identifier.
- Observe bien les OBJETS que la personne tient (selon l'Analyse Visuelle) et mentionne-les si pertinent.
- Commente l'activité de la personne (assise, debout, en mouvement, etc.).
- Ne répète JAMAIS la même phrase deux fois de suite.
- NE mentionne PAS que tu es une IA. Tu es "l'agent de sécurité".
- N'utilise PAS de markdown, d'astérisques ou de formatage. Parle naturellement.
- Ne fais pas de réflexion ou de raisonnement, donne uniquement ta réponse directe."""


class AISecurityGuard:
    """Agent de sécurité IA conversationnel."""

    def __init__(self, ollama_url: str = None, ollama_model: str = None,
                 stt_model: str = None, listen_duration: float = None,
                 trigger_interval: float = None, voice_enabled: bool = True):
        """
        Args:
            ollama_url: URL du serveur Ollama
            ollama_model: Modèle Ollama à utiliser
            stt_model: Taille du modèle Whisper (tiny, base, small, medium)
            listen_duration: Durée d'écoute micro en secondes
            trigger_interval: Intervalle min entre interventions (secondes)
            voice_enabled: Activer la voix TTS
        """
        # Config
        try:
            from src.config import (
                OLLAMA_URL, OLLAMA_MODEL,
                GUARD_STT_MODEL, GUARD_LISTEN_DURATION,
                GUARD_TRIGGER_INTERVAL,
                VISION_MODEL, VISION_ANALYSIS_INTERVAL, VISION_ENABLED,
            )
            self.ollama_url = ollama_url or OLLAMA_URL
            self.ollama_model = ollama_model or OLLAMA_MODEL
            self.stt_model_name = stt_model or GUARD_STT_MODEL
            self.listen_duration = listen_duration or GUARD_LISTEN_DURATION
            self.trigger_interval = trigger_interval or GUARD_TRIGGER_INTERVAL
        except ImportError:
            self.ollama_url = ollama_url or DEFAULT_OLLAMA_URL
            self.ollama_model = ollama_model or DEFAULT_OLLAMA_MODEL
            self.stt_model_name = stt_model or DEFAULT_STT_MODEL
            self.listen_duration = listen_duration or DEFAULT_LISTEN_DURATION
            self.trigger_interval = trigger_interval or DEFAULT_TRIGGER_INTERVAL

        self.voice_enabled = voice_enabled
        self.available = False

        # State
        self._last_intervention = 0.0
        self._last_response = ""
        self._current_response = ""
        self._responding = False
        self._listening = False
        self._conversation_history: List[Dict] = []
        self._last_scene = ""
        self._interaction_active = False

        # TTS (Edge-TTS)
        self._tts_available = False
        self._tts_lock = threading.Lock()
        self._tts_busy = False

        # STT (Whisper)
        self._whisper_model = None

        # Vision model (analyse de frames)
        self._vision_available = False
        self._vision_model = VISION_MODEL
        self._vision_interval = VISION_ANALYSIS_INTERVAL
        self._vision_enabled = VISION_ENABLED
        self._last_vision_analysis = 0
        self._vision_description = ""  # Dernière description visuelle
        self._current_frame = None      # Frame courante pour analyse
        self._vision_lock = threading.Lock()

        # Thread lock
        self._lock = threading.Lock()

        # Init components
        self._init_ollama()
        if self.available:
            self._init_tts()
            self._init_stt()
            self._init_vision()

    # ──────────────────────────────────────
    #  INITIALIZATION
    # ──────────────────────────────────────

    def _init_ollama(self):
        """Vérifie la connexion à Ollama."""
        try:
            import requests
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                # Check if our model is available (with or without :latest tag)
                model_base = self.ollama_model.split(":")[0]
                found = any(model_base in m for m in models)
                if found:
                    self.available = True
                    print(f"[AI-GUARD] Ollama connecté ✓ (modèle: {self.ollama_model})")
                else:
                    print(f"[AI-GUARD] ⚠ Modèle '{self.ollama_model}' non trouvé. "
                          f"Disponibles: {', '.join(models)}")
            else:
                print(f"[AI-GUARD] ⚠ Ollama inaccessible (status={r.status_code})")
        except Exception as e:
            print(f"[AI-GUARD] ⚠ Ollama non disponible ({e})")

    def _init_tts(self):
        """Initialise Edge-TTS (voix neuronale Microsoft)."""
        if not self.voice_enabled:
            return
        try:
            import edge_tts
            self._tts_available = True
            print(f"[AI-GUARD] TTS initialisé ✓ (Edge-TTS, voix: {EDGE_TTS_VOICE})")
        except ImportError:
            print("[AI-GUARD] ⚠ edge-tts non installé (pip install edge-tts)")
            self._tts_available = False

    def _init_stt(self):
        """Initialise Whisper STT."""
        try:
            from faster_whisper import WhisperModel
            self._whisper_model = WhisperModel(self.stt_model_name, device="auto", compute_type="int8")
            print(f"[AI-GUARD] Faster-Whisper STT initialisé ✓ (modèle: {self.stt_model_name})")
        except Exception as e:
            print(f"[AI-GUARD] ⚠ Faster-Whisper STT non disponible ({e})")
            self._whisper_model = None

    def _init_vision(self):
        """Vérifie que le modèle de vision est disponible sur Ollama."""
        if not self._vision_enabled:
            return
        try:
            import requests
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                model_base = self._vision_model.split(":")[0]
                found = any(model_base in m for m in models)
                if found:
                    self._vision_available = True
                    print(f"[AI-GUARD] Vision initialisée ✓ (modèle: {self._vision_model}, "
                          f"intervalle: {self._vision_interval}s)")
                else:
                    print(f"[AI-GUARD] ⚠ Modèle vision '{self._vision_model}' non trouvé. "
                          f"Installer avec: ollama pull {self._vision_model}")
        except Exception:
            print("[AI-GUARD] ⚠ Vision non disponible")

    def _analyze_frame_vision(self, frame) -> str:
        """Envoie une frame au modèle de vision Ollama et retourne la description."""
        import requests
        import base64
        import cv2

        try:
            # Réduire la résolution pour accélérer
            h, w = frame.shape[:2]
            scale = min(640 / w, 480 / h, 1.0)
            if scale < 1.0:
                resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                resized = frame

            # Encoder en JPEG puis base64
            _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            # Envoyer au modèle de vision
            r = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self._vision_model,
                    "prompt": (
                        "Décris précisément cette image de caméra de surveillance en français. "
                        "Mentionne : le nombre de personnes, ce qu'elles font, ce qu'elles tiennent "
                        "dans les mains, leur posture, leurs vêtements, et tout objet visible. "
                        "Sois bref et factuel, 2-3 phrases maximum."
                    ),
                    "images": [img_b64],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 150,
                    }
                },
                timeout=20
            )

            if r.status_code == 200:
                description = r.json().get("response", "").strip()
                if description:
                    print(f"  👁️ [VISION] {description}")
                    return description
        except Exception as e:
            print(f"[AI-GUARD] Erreur vision: {e}")

        return ""

    def update_frame(self, frame):
        """
        Appelé depuis la boucle principale pour fournir la frame courante.
        Lance l'analyse vision périodiquement dans un thread séparé.
        """
        if not self._vision_available or not self.available:
            return

        now = time.time()
        if now - self._last_vision_analysis >= self._vision_interval:
            self._last_vision_analysis = now
            # Copier la frame pour le thread
            frame_copy = frame.copy()

            def _do_vision():
                desc = self._analyze_frame_vision(frame_copy)
                if desc:
                    with self._vision_lock:
                        self._vision_description = desc

            t = threading.Thread(target=_do_vision, daemon=True)
            t.start()

    # ──────────────────────────────────────
    #  SCENE DESCRIPTION (Vision → Texte)
    # ──────────────────────────────────────

    def _analyze_body_position(self, det) -> str:
        """Analyse les keypoints pour décrire la position du corps."""
        try:
            kpts = det.keypoints_xy  # (17, 2) array
            if kpts is None or len(kpts) < 17:
                return ""

            details = []

            # Indices COCO 17 keypoints
            NOSE, L_SHOULDER, R_SHOULDER = 0, 5, 6
            L_ELBOW, R_ELBOW = 7, 8
            L_WRIST, R_WRIST = 9, 10
            L_HIP, R_HIP = 11, 12

            nose_y = kpts[NOSE][1]
            l_shoulder_y = kpts[L_SHOULDER][1]
            r_shoulder_y = kpts[R_SHOULDER][1]
            l_wrist_y = kpts[L_WRIST][1]
            r_wrist_y = kpts[R_WRIST][1]
            l_wrist_x = kpts[L_WRIST][0]
            r_wrist_x = kpts[R_WRIST][0]
            l_elbow_y = kpts[L_ELBOW][1]
            r_elbow_y = kpts[R_ELBOW][1]
            shoulder_mid_y = (l_shoulder_y + r_shoulder_y) / 2

            # Position des bras
            both_hands_up = (l_wrist_y < l_shoulder_y and
                             r_wrist_y < r_shoulder_y and
                             l_wrist_y > 0 and r_wrist_y > 0)
            left_hand_up = l_wrist_y < l_shoulder_y and l_wrist_y > 0
            right_hand_up = r_wrist_y < r_shoulder_y and r_wrist_y > 0

            if both_hands_up:
                if l_wrist_y < nose_y and r_wrist_y < nose_y:
                    details.append("les DEUX MAINS AU-DESSUS DE LA TÊTE")
                else:
                    details.append("les deux bras levés")
            elif left_hand_up:
                details.append("le bras gauche levé")
            elif right_hand_up:
                details.append("le bras droit levé")
            else:
                details.append("les bras le long du corps")

            # Mains rapprochées (tient un objet ?)
            if l_wrist_x > 0 and r_wrist_x > 0:
                wrist_dist = abs(l_wrist_x - r_wrist_x)
                shoulder_dist = abs(kpts[L_SHOULDER][0] - kpts[R_SHOULDER][0])
                if shoulder_dist > 0 and wrist_dist < shoulder_dist * 0.4:
                    if l_wrist_y > shoulder_mid_y:
                        details.append("les mains rapprochées devant le corps (tient peut-être un objet)")
                    else:
                        details.append("les mains rapprochées en hauteur")

            # Orientation : face à la caméra ou de profil
            shoulder_width = abs(kpts[L_SHOULDER][0] - kpts[R_SHOULDER][0])
            hip_width = abs(kpts[L_HIP][0] - kpts[R_HIP][0])
            if shoulder_width > 0:
                if shoulder_width > hip_width * 1.2:
                    details.append("fait face à la caméra")
                elif shoulder_width < hip_width * 0.6:
                    details.append("de profil par rapport à la caméra")

            return ", ".join(details)
        except Exception:
            return ""

    def _build_context(self, detections: list, person_stats: dict,
                       results: dict) -> str:
        """Construit une description textuelle détaillée de la scène pour Ollama."""
        # Traduction des actions en descriptions claires
        ACTION_DESC = {
            "immobile": "est immobile, debout sans bouger",
            "s'asseoir": "est assis(e)",
            "courir": "COURT RAPIDEMENT",
            "chute": "EST AU SOL / A CHUTÉ",
            "donner_un_coup": "FRAPPE / GESTE VIOLENT DÉTECTÉ",
            "mains_en_l_air": "a les MAINS EN L'AIR",
            "marcher": "marche / se déplace",
            "se_pencher": "se penche vers l'avant",
            "N/A": "en cours d'analyse",
        }

        lines = []
        now = time.strftime("%H:%M:%S")
        lines.append(f"[{now}] RAPPORT DÉTAILLÉ DE LA CAMÉRA DE SURVEILLANCE :")

        if not detections:
            lines.append("Aucune personne visible dans le champ de la caméra.")
            return "\n".join(lines)

        lines.append(f"Nombre de personnes détectées : {len(detections)}")

        for det in detections:
            tid = det.track_id
            stats = person_stats.get(tid, {})
            result = results.get(tid, {})

            name = stats.get("name", result.get("name", "INCONNU"))
            action_raw = stats.get("current_action", result.get("action", "N/A"))
            action_desc = ACTION_DESC.get(action_raw, action_raw)
            presence = stats.get("presence_time", 0)
            objects_pose = stats.get("pose_objects", [])
            objects_yolo = result.get("objects", [])
            is_loitering = result.get("loitering", False)

            # En-tête
            status = "IDENTIFIÉE" if name != "INCONNU" else "NON IDENTIFIÉE"
            desc_parts = [f"\n=== PERSONNE {status} : '{name}' (ID:{tid}) ==="]
            desc_parts.append(f"  Activité : {action_desc}")
            desc_parts.append(f"  Durée de présence : {int(presence)} secondes")

            # Analyse corporelle depuis les keypoints
            body_desc = self._analyze_body_position(det)
            if body_desc:
                desc_parts.append(f"  Position du corps : {body_desc}")

            # Objets détectés par YOLO
            all_objects = list(set(objects_pose + objects_yolo))
            if all_objects:
                # Nettoyer les emojis des labels pour le contexte texte
                clean_objs = []
                for obj in all_objects:
                    # Retirer emojis du début si présents
                    parts = obj.split(" ", 1)
                    clean_objs.append(parts[-1] if len(parts) > 1 else obj)
                obj_str = ", ".join(clean_objs)
                desc_parts.append(f"  OBJETS PORTÉS/VISIBLES : {obj_str}")
            else:
                desc_parts.append("  Aucun objet détecté dans les mains")

            # Alerte maraudage
            if is_loitering:
                desc_parts.append("  ⚠ ALERTE : MARAUDAGE / RÔDE DEPUIS TROP LONGTEMPS")

            lines.append("\n".join(desc_parts))

        # Ajouter la description de la caméra vision si disponible
        with self._vision_lock:
            if self._vision_description:
                lines.append(f"\n=== ANALYSE VISUELLE DE LA CAMÉRA ===")
                lines.append(f"  {self._vision_description}")

        return "\n".join(lines)

    # ──────────────────────────────────────
    #  OLLAMA LLM
    # ──────────────────────────────────────

    def _query_ollama(self, scene_context: str,
                      human_speech: str = None) -> Optional[str]:
        """Envoie le contexte à Ollama et retourne la réponse."""
        import requests

        # Build conversation messages
        # Pour les modèles qwen3, désactiver le mode thinking
        system_content = SYSTEM_PROMPT
        if "qwen" in self.ollama_model.lower():
            system_content += "\n/no_think"

        messages = [{"role": "system", "content": system_content}]

        # Add recent history (last 4 exchanges max)
        for msg in self._conversation_history[-8:]:
            messages.append(msg)

        # Build the user message
        user_content = f"SITUATION ACTUELLE :\n{scene_context}"
        if human_speech:
            user_content += f"\n\nLA PERSONNE DANS L'ENTRÉE DIT : \"{human_speech}\""
        else:
            user_content += ("\n\nAucune parole détectée. "
                             "Décide si tu dois intervenir ou rester silencieux. "
                             "Si la situation ne nécessite aucune intervention, "
                             "réponds uniquement : [SILENCE]")

        messages.append({"role": "user", "content": user_content})

        try:
            import json, re
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 200,
                    }
                },
                stream=True,
                timeout=15
            )
            if r.status_code == 200:
                full_response = ""
                buffer = ""
                self._tts_busy = True
                try:
                    for line in r.iter_lines():
                        if line:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            buffer += chunk
                            full_response += chunk
                            
                            # Si on rencontre une ponctuation finale ou la fin de génération
                            if any(p in chunk for p in ['.', '!', '?']) or data.get("done"):
                                clean_buf = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL).strip()
                                if clean_buf and clean_buf != "[SILENCE]":
                                    print(f"  🤖 [AI-GUARD] {clean_buf}")
                                    self._speak(clean_buf)
                                buffer = ""
                finally:
                    self._tts_busy = False

                response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
                if response and response != "[SILENCE]":
                    self._conversation_history.append({"role": "user", "content": user_content})
                    self._conversation_history.append({"role": "assistant", "content": response})
                    if len(self._conversation_history) > 16:
                        self._conversation_history = self._conversation_history[-12:]
                    return response
            return None
        except Exception as e:
            print(f"[AI-GUARD] Erreur Ollama: {e}")
            return None

    # ──────────────────────────────────────
    #  STT (Microphone → Texte)
    # ──────────────────────────────────────

    def _listen_microphone(self) -> Optional[str]:
        """Capture audio du micro et transcrit via Whisper."""
        if not self._whisper_model:
            return None

        try:
            import pyaudio

            RATE = 16000
            CHANNELS = 1
            FORMAT = pyaudio.paInt16
            CHUNK = 1024
            duration = self.listen_duration

            pa = pyaudio.PyAudio()
            stream = pa.open(format=FORMAT, channels=CHANNELS,
                             rate=RATE, input=True,
                             frames_per_buffer=CHUNK)

            self._listening = True
            print(f"  🎤 Écoute en cours ({duration}s)...")

            frames = []
            for _ in range(int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            pa.terminate()
            self._listening = False

            # Save to temp WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wf = wave.open(f.name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pa.get_sample_size(FORMAT) if hasattr(pa, 'get_sample_size') else 2)
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                temp_path = f.name

            # Transcribe with Faster-Whisper
            segments, _ = self._whisper_model.transcribe(
                temp_path, language="fr"
            )
            text = " ".join([segment.text for segment in segments]).strip()

            # Cleanup
            import os
            os.unlink(temp_path)

            if text and len(text) > 2:
                print(f"  🎤 Entendu : \"{text}\"")
                return text
            return None

        except Exception as e:
            self._listening = False
            print(f"[AI-GUARD] Erreur micro: {e}")
            return None

    # ──────────────────────────────────────
    #  TTS (Texte → Voix)
    # ──────────────────────────────────────

    def _speak(self, text: str):
        """Prononce le texte via Edge-TTS (bloquant)."""
        if not self._tts_available:
            print(f"  🔊 [AI-GUARD dit] {text}")
            return

        try:
            import edge_tts

            # Générer l'audio dans un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name

            # Edge-TTS est async, on crée un event loop dédié
            async def _generate():
                communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
                await communicate.save(temp_path)

            loop = asyncio.new_event_loop()
            loop.run_until_complete(_generate())
            loop.close()

            # Jouer l'audio
            with self._tts_lock:
                if platform.system() == "Darwin":  # macOS
                    subprocess.run(["afplay", temp_path],
                                   capture_output=True, timeout=15)
                elif platform.system() == "Windows":
                    subprocess.run(["powershell", "-c",
                                    f'(New-Object Media.SoundPlayer "{temp_path}").PlaySync()'],
                                   capture_output=True, timeout=15)
                else:  # Linux
                    subprocess.run(["aplay", temp_path],
                                   capture_output=True, timeout=15)

            # Cleanup
            os.unlink(temp_path)

        except Exception as e:
            print(f"[AI-GUARD] Erreur TTS: {e}")

    def _speak_async(self, text: str):
        """Prononce en thread séparé."""
        if self._tts_busy:
            return
        self._tts_busy = True

        def _do():
            try:
                self._speak(text)
            finally:
                self._tts_busy = False

        t = threading.Thread(target=_do, daemon=True)
        t.start()

    # ──────────────────────────────────────
    #  INTERACTION LOOP (Thread dédié)
    # ──────────────────────────────────────

    def _interaction_cycle(self, scene_context: str, reason: str = ""):
        """Un cycle d'interaction adapté au contexte.
        
        - known_handwave : écoute d'abord, puis Ollama répond
        - autres : Ollama analyse, parle, puis écoute
        """
        self._responding = True
        try:
            # === MODE MAIN LEVÉE (personne connue) → écouter d'abord ===
            if reason.startswith("known_handwave"):
                name = reason.split(":", 1)[-1] if ":" in reason else ""
                print(f"  🎙️ {name} veut parler — écoute active...")

                with self._lock:
                    self._current_response = f"🎙️ Écoute de {name}..."

                # 1. Écouter le micro directement
                speech = self._listen_microphone()

                if speech:
                    # 2. Envoyer la parole + contexte à Ollama
                    response = self._query_ollama(scene_context, speech)

                    if response:
                        self._last_response = response
                        with self._lock:
                            self._current_response = response
                        self._interaction_active = True
                    else:
                        self._interaction_active = False
                else:
                    print(f"  🎙️ Rien entendu.")
                    self._interaction_active = False
                return

            # === MODE NORMAL → Ollama analyse d'abord, parle, puis écoute ===
            response = self._query_ollama(scene_context, None)

            if response:
                if response.lower().strip() == self._last_response.lower().strip():
                    self._interaction_active = False
                    return

                self._last_response = response
                with self._lock:
                    self._current_response = response

                time.sleep(1.0)

                speech = self._listen_microphone()

                if speech:
                    followup = self._query_ollama(scene_context, speech)

                    if followup:
                        self._last_response = followup
                        with self._lock:
                            self._current_response = followup
                        self._interaction_active = True
                else:
                    self._interaction_active = False
            else:
                self._interaction_active = False

        except Exception as e:
            print(f"[AI-GUARD] Erreur cycle: {e}")
        finally:
            self._tts_busy = False
            self._responding = False

    # ──────────────────────────────────────
    #  UPDATE (appelé depuis la boucle main)
    # ──────────────────────────────────────

    def update(self, detections: list, person_stats: dict, results: dict):
        """
        Appelé à chaque frame. Décide si l'agent doit intervenir.

        Déclenche un cycle d'interaction dans un thread séparé si :
        - Il y a des personnes visibles
        - L'intervalle minimum est respecté
        - Le guard n'est pas déjà en train de répondre
        """
        if not self.available or self._responding:
            return

        now = time.time()

        # Vérifier si on doit intervenir
        should_intervene = False
        reason = ""

        # Si conversation active (la personne parle), répondre après 10s
        if self._interaction_active:
            if now - self._last_intervention >= 10.0:
                should_intervene = True
                reason = "conversation_active"

        elif not detections:
            return

        else:
            for det in detections:
                tid = det.track_id
                r = results.get(tid, {})
                stats = person_stats.get(tid, {})

                name = stats.get("name", r.get("name", "INCONNU"))
                action = r.get("action", "N/A")
                presence = stats.get("presence_time", 0)

                # ★ PRIORITÉ 1 : Personne CONNUE lève les mains → veut parler
                if (name != "INCONNU" and action == "mains_en_l_air"
                        and now - self._last_intervention >= 5.0):
                    should_intervene = True
                    reason = f"known_handwave:{name}"
                    print(f"  🙋 {name} lève les mains → conversation...")
                    break

                # Respecter le trigger_interval pour les autres cas
                if now - self._last_intervention < self.trigger_interval:
                    continue

                # ★ PRIORITÉ 2 : actions d'alerte
                from src.config import ALERT_ACTIONS
                if action in ALERT_ACTIONS:
                    should_intervene = True
                    reason = f"alert_action:{action}"
                    break

                # ★ PRIORITÉ 3 : Personne inconnue depuis longtemps
                if name == "INCONNU" and presence > 10:
                    should_intervene = True
                    reason = f"unknown_person:{tid}"
                    break

                # ★ PRIORITÉ 4 : Maraudage
                if r.get("loitering"):
                    should_intervene = True
                    reason = f"loitering:{tid}"
                    break

        if should_intervene:
            self._last_intervention = now
            scene = self._build_context(detections, person_stats, results)

            # Lancer le cycle dans un thread
            t = threading.Thread(
                target=self._interaction_cycle,
                args=(scene, reason),
                daemon=True
            )
            t.start()

    # ──────────────────────────────────────
    #  DRAW (Overlay sur la frame)
    # ──────────────────────────────────────

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Dessine l'état de l'agent sur la frame."""
        h, w = frame.shape[:2]

        if not self.available:
            return frame

        # Indicateur d'état en haut à droite
        status_color = (0, 255, 0)  # vert par défaut
        status_text = "AI GUARD"

        if self._listening:
            status_color = (0, 200, 255)  # orange
            status_text = "AI GUARD 🎤 ÉCOUTE..."
        elif self._responding:
            status_color = (255, 200, 0)  # cyan
            status_text = "AI GUARD 🤖 RÉFLEXION..."
        elif self._tts_busy:
            status_color = (0, 255, 255)  # jaune
            status_text = "AI GUARD 🔊 PARLE..."

        # Badge en haut à droite
        badge_w = 280
        badge_h = 30
        x1 = w - badge_w - 10
        y1 = 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + badge_w, y1 + badge_h),
                      status_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, status_text, (x1 + 10, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # Bulle de dialogue si réponse active
        with self._lock:
            response = self._current_response

        if response and (time.time() - self._last_intervention) < 12:
            # Bulle en bas au centre
            bubble_text = response[:120]  # Max 120 chars
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2

            # Word wrap
            words = bubble_text.split()
            lines = []
            current_line = ""
            for word in words:
                test = f"{current_line} {word}".strip()
                (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
                if tw > w - 100:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test
            if current_line:
                lines.append(current_line)

            line_h = 30
            box_h = len(lines) * line_h + 20
            box_w = min(w - 60, 800)
            bx = (w - box_w) // 2
            by = h - box_h - 170  # Au-dessus de la barre d'instructions

            # Fond semi-transparent
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (bx, by), (bx + box_w, by + box_h),
                          (40, 40, 40), -1)
            cv2.rectangle(overlay2, (bx, by), (bx + box_w, by + box_h),
                          status_color, 2)
            cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)

            # Icône
            cv2.putText(frame, "🤖", (bx + 10, by + 25),
                        font, 0.7, status_color, 2, cv2.LINE_AA)

            # Texte
            for i, line in enumerate(lines):
                cv2.putText(frame, line,
                            (bx + 45, by + 25 + i * line_h),
                            font, font_scale, (255, 255, 255),
                            thickness, cv2.LINE_AA)

        return frame

    # ──────────────────────────────────────
    #  STATS
    # ──────────────────────────────────────

    def get_stats(self) -> dict:
        """Retourne les stats du guard."""
        return {
            "available": self.available,
            "model": self.ollama_model,
            "responding": self._responding,
            "listening": self._listening,
            "tts_busy": self._tts_busy,
            "history_length": len(self._conversation_history),
            "last_response": self._last_response[:80] if self._last_response else "",
        }
