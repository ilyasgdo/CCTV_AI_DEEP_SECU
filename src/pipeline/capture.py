"""
Thread de capture vidéo.
Lit la vidéo en continu et place les frames dans une file d'attente.
"""
import cv2
import time
import threading
from queue import Queue
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import VIDEO_WIDTH, VIDEO_HEIGHT, TARGET_FPS


class VideoCapture:
    """
    Capture vidéo asynchrone.

    Lit en continu depuis une source (webcam, fichier, RTSP)
    et garde toujours la frame la plus récente disponible.
    """

    def __init__(self, source=0, queue_size: int = 2):
        """
        Args:
            source: 0 (webcam), chemin fichier, ou URL RTSP
            queue_size: Taille de la file (petit = faible latence)
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source vidéo : {source}")

        # Configurer la résolution si c'est une webcam
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        self.queue = Queue(maxsize=queue_size)
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.fps = 0.0

        # Info vidéo
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS

        print(f"[CAPTURE] Source : {source}")
        print(f"[CAPTURE] Résolution : {self.width}x{self.height}")
        print(f"[CAPTURE] FPS source : {self.source_fps}")

    def start(self):
        """Démarre le thread de capture."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("[CAPTURE] Thread démarré")
        return self

    def _capture_loop(self):
        """Boucle de capture (exécutée dans un thread séparé)."""
        fps_start = time.time()
        fps_counter = 0

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                if isinstance(self.source, str):
                    # Replay vidéo
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.running = False
                    break

            # Si la queue est pleine, supprimer la vieille frame
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Exception:
                    pass

            self.queue.put(frame)
            self.frame_count += 1

            # Calcul FPS
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                self.fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

    def read(self):
        """
        Lit la frame la plus récente.

        Returns:
            (success: bool, frame: np.ndarray or None)
        """
        if not self.queue.empty():
            return True, self.queue.get()
        return False, None

    def stop(self):
        """Arrête le thread de capture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.cap.release()
        print("[CAPTURE] Thread arrêté")

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
