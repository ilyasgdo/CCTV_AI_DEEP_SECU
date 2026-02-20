"""
Système d'accueil et d'identification automatique.

Fonctionnalités :
  1. ACCUEIL : Quand une personne CONNUE entre → "Bonjour [Nom]" (voix + visuel)
  2. IDENTIFICATION : Quand un INCONNU est détecté > 10s → "Identifiez-vous" (voix + visuel)

Le système ne répète pas les annonces trop souvent (cooldown configurable).
"""
import cv2
import time
import math
import threading
import numpy as np
from typing import Dict, Set, Tuple, Optional


class IdentificationRequester:
    """
    Gère les accueils des personnes connues et les demandes d'identification
    pour les inconnus.
    """

    def __init__(self, delay: float = 10.0, cooldown: float = 30.0,
                 voice_enabled: bool = True):
        """
        Args:
            delay: Secondes avant de demander l'identification d'un inconnu
            cooldown: Secondes entre deux demandes pour la même personne
            voice_enabled: Activer les annonces vocales TTS
        """
        self.delay = delay
        self.cooldown = cooldown
        self.voice_enabled = voice_enabled

        # === IDENTIFICATION (inconnus) ===
        # {track_id: first_seen_as_unknown}
        self._unknown_since: Dict[int, float] = {}
        # {track_id: last_request_time}
        self._last_request: Dict[int, float] = {}
        # IDs en alerte d'identification
        self._active_alerts: Dict[int, float] = {}

        # === ACCUEIL (personnes connues) ===
        # {track_id: True} pour les personnes déjà accueillies
        self._greeted: Dict[int, bool] = {}
        # Messages d'accueil actifs {track_id: (name, start_time)}
        self._greeting_active: Dict[int, Tuple[str, float]] = {}
        # Noms déjà accueillis dans cette session (éviter doublons)
        self._greeted_names: Set[str] = set()

        # TTS engine
        self._tts_engine = None
        self._tts_lock = threading.Lock()
        self._tts_busy = False

        if voice_enabled:
            self._init_tts()

        print(f"[ID-REQUEST] Initialisé (délai={delay}s, cooldown={cooldown}s, "
              f"voix={'ON' if voice_enabled else 'OFF'})")

    def _init_tts(self):
        """Initialise le moteur TTS."""
        try:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            voices = self._tts_engine.getProperty('voices')
            for v in voices:
                if 'french' in v.name.lower() or 'fr' in v.id.lower():
                    self._tts_engine.setProperty('voice', v.id)
                    break
            self._tts_engine.setProperty('rate', 160)
            self._tts_engine.setProperty('volume', 1.0)
            print("[ID-REQUEST] TTS initialisé ✓")
        except Exception as e:
            print(f"[ID-REQUEST] ⚠ TTS non disponible ({e})")
            self._tts_engine = None

    def _speak_async(self, text: str):
        """Annonce vocale dans un thread séparé."""
        if not self._tts_engine or self._tts_busy:
            return

        def _speak():
            self._tts_busy = True
            try:
                with self._tts_lock:
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
            except Exception:
                pass
            finally:
                self._tts_busy = False

        t = threading.Thread(target=_speak, daemon=True)
        t.start()

    def update(self, detections: list, face_matcher) -> list:
        """
        Vérifie chaque détection :
          - Personne connue → accueil vocal "Bonjour [Nom]"
          - Personne inconnue > delay → demande d'identification
        
        Args:
            detections: Liste de PersonDetection
            face_matcher: Le FaceMatcher pour obtenir les noms
            
        Returns:
            Liste de track_ids nécessitant identification
        """
        now = time.time()
        requesting_ids = []
        active_unknown_ids = set()

        for det in detections:
            tid = det.track_id
            name = getattr(det, 'name', None) or face_matcher.get_name(tid)

            if name and name != "INCONNU":
                # ===== PERSONNE CONNUE : ACCUEIL =====
                if tid not in self._greeted and name not in self._greeted_names:
                    self._greeted[tid] = True
                    self._greeted_names.add(name)
                    self._greeting_active[tid] = (name, now)

                    # Annonce vocale d'accueil
                    if self.voice_enabled:
                        self._speak_async(f"Bonjour {name}. Bienvenue.")

                    print(f"  👋 BONJOUR {name} ! (ID:{tid})")

                # Retirer des listes d'inconnus si la personne est maintenant identifiée
                if tid in self._unknown_since:
                    del self._unknown_since[tid]
                if tid in self._active_alerts:
                    del self._active_alerts[tid]

            else:
                # ===== PERSONNE INCONNUE =====
                active_unknown_ids.add(tid)

                if tid not in self._unknown_since:
                    self._unknown_since[tid] = now

                elapsed = now - self._unknown_since[tid]
                if elapsed >= self.delay:
                    last = self._last_request.get(tid, 0)
                    if now - last >= self.cooldown:
                        requesting_ids.append(tid)
                        self._last_request[tid] = now
                        self._active_alerts[tid] = now

                        if self.voice_enabled:
                            self._speak_async(
                                "Attention. Personne non identifiée détectée. "
                                "Veuillez vous identifier face à la caméra."
                            )

                        print(f"  🔊 IDENTIFICATION REQUISE — ID:{tid} "
                              f"(inconnu depuis {elapsed:.0f}s)")

        # Nettoyer les IDs perdus
        lost = set(self._unknown_since.keys()) - {d.track_id for d in detections}
        for tid in lost:
            self._unknown_since.pop(tid, None)

        # Désactiver les alertes d'identification après 10s de clignotement
        expired = [tid for tid, t in self._active_alerts.items() if now - t > 10.0]
        for tid in expired:
            del self._active_alerts[tid]

        # Désactiver les accueils après 5s d'affichage
        expired_greet = [tid for tid, (_, t) in self._greeting_active.items()
                         if now - t > 5.0]
        for tid in expired_greet:
            del self._greeting_active[tid]

        return requesting_ids

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Dessine les bandeaux d'accueil (vert) et d'identification (rouge).
        """
        now = time.time()

        for det in detections:
            tid = det.track_id

            # ===== BANDEAU D'ACCUEIL (vert) =====
            if tid in self._greeting_active:
                name, start_time = self._greeting_active[tid]
                elapsed = now - start_time

                x1, y1, x2, y2 = det.bbox.astype(int)
                bw = x2 - x1

                # Effet d'apparition (fade in)
                alpha = min(1.0, elapsed / 0.5)

                banner_h = 50
                banner_y = max(0, y1 - banner_h - 10)

                # Fond vert semi-transparent
                overlay = frame.copy()
                cv2.rectangle(overlay,
                              (x1 - 15, banner_y),
                              (x2 + 15, banner_y + banner_h),
                              (0, 150, 0), -1)
                cv2.addWeighted(overlay, 0.75 * alpha, frame, 1 - 0.75 * alpha,
                                0, frame)

                # Bordure verte
                cv2.rectangle(frame,
                              (x1 - 15, banner_y),
                              (x2 + 15, banner_y + banner_h),
                              (0, 255, 0), 2)

                # Texte "👋 Bonjour [Nom] !"
                text = f"Bonjour {name} !"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.7
                (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
                tx = x1 + (bw - tw) // 2
                ty = banner_y + (banner_h + th) // 2

                # Ombre
                cv2.putText(frame, text, (tx + 1, ty + 1),
                            font, scale, (0, 0, 0), 3, cv2.LINE_AA)
                # Texte blanc
                cv2.putText(frame, text, (tx, ty),
                            font, scale, (255, 255, 255), 2, cv2.LINE_AA)

                # Cadre vert autour de la personne
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                              (0, 255, 0), 3)

            # ===== BANDEAU D'IDENTIFICATION (rouge clignotant) =====
            if tid in self._active_alerts:
                blink = math.sin(now * 6) > 0
                if not blink:
                    continue

                x1, y1, x2, y2 = det.bbox.astype(int)
                bw = x2 - x1

                banner_h = 40
                banner_y = max(0, y1 - banner_h - 5)

                overlay = frame.copy()
                cv2.rectangle(overlay,
                              (x1 - 10, banner_y),
                              (x2 + 10, banner_y + banner_h),
                              (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

                cv2.rectangle(frame,
                              (x1 - 10, banner_y),
                              (x2 + 10, banner_y + banner_h),
                              (0, 0, 255), 2)

                text = "IDENTIFIEZ-VOUS"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
                tx = x1 + (bw - tw) // 2
                ty = banner_y + (banner_h + th) // 2

                cv2.putText(frame, text, (tx + 1, ty + 1),
                            font, scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, text, (tx, ty),
                            font, scale, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                              (0, 0, 255), 3)

                icon_y = y2 + 25
                cv2.putText(frame, "! INCONNU !", (x1, icon_y),
                            font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # === Grand bandeau en bas si alerte identification active ===
        if self._active_alerts:
            blink_full = math.sin(now * 4) > -0.3
            if blink_full:
                h, w = frame.shape[:2]
                bar_h = 50
                bar_y = h - bar_h

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, bar_y), (w, h), (0, 0, 180), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                text = "PERSONNE(S) NON IDENTIFIEE(S) — VEUILLEZ VOUS IDENTIFIER"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
                tx = (w - tw) // 2
                ty = bar_y + (bar_h + th) // 2

                cv2.putText(frame, text, (tx, ty),
                            font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # === Bandeau vert en haut si accueil actif ===
        if self._greeting_active:
            h, w = frame.shape[:2]
            names = [n for n, _ in self._greeting_active.values()]
            bar_h = 45

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 46), (w, 46 + bar_h), (0, 120, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            text = f"Bienvenue : {', '.join(names)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
            tx = (w - tw) // 2
            ty = 46 + (bar_h + th) // 2

            cv2.putText(frame, text, (tx, ty),
                        font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def has_active_alerts(self) -> bool:
        return len(self._active_alerts) > 0

    def cleanup_lost_ids(self, active_ids: set):
        lost = set(self._unknown_since.keys()) - active_ids
        for tid in lost:
            self._unknown_since.pop(tid, None)
        lost2 = set(self._active_alerts.keys()) - active_ids
        for tid in lost2:
            self._active_alerts.pop(tid, None)

    def get_stats(self) -> dict:
        return {
            "tracked_unknowns": len(self._unknown_since),
            "active_alerts": len(self._active_alerts),
            "greeted_persons": list(self._greeted_names),
            "tts_available": self._tts_engine is not None,
        }
