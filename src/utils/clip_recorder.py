"""
Enregistreur de clips vidéo automatique sur alerte.

Maintient un buffer circulaire des dernières frames.
Quand une alerte se déclenche, sauvegarde les N secondes
avant + après l'événement dans un fichier vidéo MP4.

Usage :
    recorder = ClipRecorder(fps=25, buffer_seconds=5, post_seconds=5)
    recorder.add_frame(frame)  # Chaque frame
    recorder.trigger_alert("chute", track_id=3, name="Thomas")  # Sur alerte
"""
import cv2
import time
import threading
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime


class ClipRecorder:
    """
    Enregistreur de clips vidéo sur alerte avec buffer circulaire.
    
    Maintient les N dernières secondes en mémoire.
    Lors d'une alerte, sauvegarde pre + post événement.
    """

    def __init__(self, fps: int = 25, buffer_seconds: int = 5,
                 post_seconds: int = 5, output_dir: str = None,
                 max_clips: int = 100):
        """
        Args:
            fps: FPS de la vidéo source
            buffer_seconds: Secondes à garder AVANT l'alerte
            post_seconds: Secondes à enregistrer APRÈS l'alerte
            output_dir: Dossier de sortie des clips
            max_clips: Nombre maximum de clips gardés (rotation)
        """
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.post_seconds = post_seconds
        self.max_clips = max_clips

        # Buffer circulaire des dernières frames
        buffer_size = fps * buffer_seconds
        self._buffer = deque(maxlen=buffer_size)

        # Dossier de sortie
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent.parent / "data" / "clips"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # État d'enregistrement
        self._recording = False
        self._record_start: float = 0
        self._record_frames: list = []
        self._record_info: dict = {}
        self._record_lock = threading.Lock()

        # Statistiques
        self.total_clips = 0
        self.total_alerts = 0

        # Cooldown pour éviter les doublons (5s min entre 2 clips)
        self._last_trigger: float = 0
        self._cooldown = 5.0

        # Frame dimensions (détectées automatiquement)
        self._frame_size: Optional[Tuple[int, int]] = None

        print(f"[CLIP] Enregistreur initialisé (buffer={buffer_seconds}s, "
              f"post={post_seconds}s, dir={self.output_dir})")

    def add_frame(self, frame: np.ndarray):
        """
        Ajoute une frame au buffer circulaire.
        Si un enregistrement est en cours, ajoute aussi aux frames post-alerte.
        """
        if frame is None:
            return

        # Détecter la taille
        if self._frame_size is None:
            self._frame_size = (frame.shape[1], frame.shape[0])

        # Ajouter au buffer circulaire (pré-alerte)
        self._buffer.append(frame.copy())

        # Si on enregistre les frames post-alerte
        if self._recording:
            with self._record_lock:
                self._record_frames.append(frame.copy())
                elapsed = time.time() - self._record_start
                if elapsed >= self.post_seconds:
                    # Fin de la période post-alerte → sauvegarder
                    self._save_clip_async()

    def trigger_alert(self, alert_type: str, track_id: int = 0,
                      name: str = "INCONNU", confidence: float = 0.0):
        """
        Déclenche l'enregistrement d'un clip vidéo.
        
        Args:
            alert_type: Type d'alerte (chute, donner_un_coup, MARAUDAGE, etc.)
            track_id: ID de la personne
            name: Nom de la personne
            confidence: Score de confiance
        """
        now = time.time()

        # Cooldown : pas deux clips en 5s
        if now - self._last_trigger < self._cooldown:
            return
        self._last_trigger = now

        # Ne pas interrompre un enregistrement en cours
        if self._recording:
            return

        self.total_alerts += 1

        # Copier le buffer pré-alerte
        pre_frames = list(self._buffer)

        # Démarrer l'enregistrement post-alerte
        with self._record_lock:
            self._recording = True
            self._record_start = now
            self._record_frames = pre_frames  # Commence avec le pré-alerte
            self._record_info = {
                "alert_type": alert_type,
                "track_id": track_id,
                "name": name,
                "confidence": confidence,
                "timestamp": datetime.now(),
            }

        print(f"  🎬 CLIP: Enregistrement déclenché ({alert_type} — {name})")

    def _save_clip_async(self):
        """Lance la sauvegarde du clip dans un thread séparé."""
        self._recording = False
        frames = self._record_frames.copy()
        info = self._record_info.copy()
        self._record_frames = []
        self._record_info = {}

        t = threading.Thread(target=self._save_clip, args=(frames, info), daemon=True)
        t.start()

    def _save_clip(self, frames: list, info: dict):
        """Sauvegarde les frames en fichier MP4."""
        if not frames or self._frame_size is None:
            return

        # Nom du fichier : YYYYMMDD_HHMMSS_type_nom.mp4
        ts = info["timestamp"]
        alert_type = info["alert_type"].replace(" ", "_")
        name = info.get("name", "inconnu").replace(" ", "_")
        filename = f"{ts.strftime('%Y%m%d_%H%M%S')}_{alert_type}_{name}.mp4"
        filepath = self.output_dir / filename

        try:
            # Encoder en MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(filepath),
                fourcc,
                self.fps,
                self._frame_size
            )

            # Ajouter un overlay d'information sur les premières frames
            for i, frame in enumerate(frames):
                # Barre d'info en haut
                display = frame.copy()
                h, w = display.shape[:2]

                # Fond semi-transparent en haut
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 150), -1)
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

                # Texte d'alerte
                ts_str = ts.strftime('%d/%m/%Y %H:%M:%S')
                cv2.putText(display,
                            f"ALERTE: {info['alert_type']} | {info['name']} | {ts_str}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2, cv2.LINE_AA)

                # Indicateur REC clignotant
                if (i // (self.fps // 2)) % 2 == 0:
                    cv2.circle(display, (w - 25, 18), 8, (0, 0, 255), -1)
                    cv2.putText(display, "REC", (w - 70, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)

                writer.write(display)

            writer.release()
            self.total_clips += 1

            duration = len(frames) / max(1, self.fps)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  💾 CLIP SAUVEGARDÉ: {filename} "
                  f"({duration:.1f}s, {size_mb:.1f} Mo)")

            # Rotation : supprimer les anciens clips si trop nombreux
            self._rotate_clips()

        except Exception as e:
            print(f"  ⚠ ERREUR clip: {e}")

    def _rotate_clips(self):
        """Supprime les clips les plus anciens si on dépasse max_clips."""
        clips = sorted(self.output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        while len(clips) > self.max_clips:
            oldest = clips.pop(0)
            oldest.unlink()
            print(f"  🗑️ Ancien clip supprimé: {oldest.name}")

    def get_clips_list(self) -> list:
        """Retourne la liste des clips sauvegardés."""
        clips = []
        for f in sorted(self.output_dir.glob("*.mp4"),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            clips.append({
                "filename": f.name,
                "path": str(f),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "date": datetime.fromtimestamp(f.stat().st_mtime).strftime(
                    '%d/%m/%Y %H:%M:%S'),
            })
        return clips[:50]

    def get_stats(self) -> dict:
        return {
            "total_clips": self.total_clips,
            "total_alerts_triggered": self.total_alerts,
            "buffer_size": len(self._buffer),
            "recording": self._recording,
            "clips_on_disk": len(list(self.output_dir.glob("*.mp4"))),
            "output_dir": str(self.output_dir),
        }
