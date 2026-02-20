"""
Gestionnaire multi-caméras avec détection automatique.

Détecte automatiquement les caméras disponibles et les gère.
Supporte 1-8 caméras avec rendu mosaïque.

Usage :
    manager = MultiCameraManager()
    manager.auto_detect()  # Détecte les caméras
    manager.start_all()    # Démarre la capture
    mosaic = manager.get_mosaic()  # Image mosaïque
"""
import cv2
import time
import threading
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class CameraFeed:
    """Flux d'une caméra individuelle."""

    def __init__(self, source, name: str = "Camera", width: int = 640,
                 height: int = 480):
        self.source = source
        self.name = name
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self.fps = 0.0
        self.frame_count = 0
        self.is_connected = False
        self._last_frame_time = 0
        self._fps_counter = 0
        self._fps_timer = time.time()

    def start(self) -> bool:
        """Démarre la capture."""
        try:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                return False

            # Configurer la résolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            self.is_connected = True
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True
        except Exception:
            return False

    def _capture_loop(self):
        """Boucle de capture dans un thread."""
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self.frame_count += 1
                    self._fps_counter += 1

                    # Calculer FPS
                    now = time.time()
                    if now - self._fps_timer >= 1.0:
                        self.fps = self._fps_counter / (now - self._fps_timer)
                        self._fps_counter = 0
                        self._fps_timer = now
            else:
                self.is_connected = False
                time.sleep(0.1)
                # Tenter de reconnecter
                try:
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self.source)
                    if self._cap.isOpened():
                        self.is_connected = True
                except Exception:
                    pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Lit la dernière frame."""
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

    def stop(self):
        """Arrête la capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        self.is_connected = False


class MultiCameraManager:
    """
    Gestionnaire multi-caméras avec détection automatique.
    """

    def __init__(self, max_cameras: int = 8):
        self.max_cameras = max_cameras
        self.cameras: List[CameraFeed] = []
        self._mosaic_size = (1920, 1080)
        print(f"[MULTI-CAM] Gestionnaire initialisé (max {max_cameras} caméras)")

    def auto_detect(self) -> int:
        """
        Détecte automatiquement les caméras disponibles.
        Teste les indices 0 à max_cameras.
        
        Returns:
            Nombre de caméras détectées
        """
        print("[MULTI-CAM] Détection automatique des caméras...")
        detected = 0

        for idx in range(self.max_cameras):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        name = f"Camera {idx}"
                        cam = CameraFeed(source=idx, name=name, width=w, height=h)
                        self.cameras.append(cam)
                        detected += 1
                        print(f"  ✅ Camera {idx} détectée ({w}x{h})")
                cap.release()
            except Exception:
                continue

        print(f"[MULTI-CAM] {detected} caméra(s) détectée(s)")
        return detected

    def add_camera(self, source, name: str = None) -> bool:
        """Ajoute manuellement une caméra (index, URL RTSP, fichier)."""
        if len(self.cameras) >= self.max_cameras:
            print(f"[MULTI-CAM] ⚠ Maximum {self.max_cameras} caméras atteint")
            return False

        if name is None:
            name = f"Camera {len(self.cameras)}"

        cam = CameraFeed(source=source, name=name)
        self.cameras.append(cam)
        print(f"  ✅ Camera ajoutée: {name} (source={source})")
        return True

    def start_all(self):
        """Démarre toutes les caméras."""
        for cam in self.cameras:
            if cam.start():
                print(f"  ▶ {cam.name} démarrée")
            else:
                print(f"  ⚠ {cam.name} — échec de démarrage")

    def stop_all(self):
        """Arrête toutes les caméras."""
        for cam in self.cameras:
            cam.stop()

    def read_all(self) -> Dict[int, np.ndarray]:
        """Lit les frames de toutes les caméras."""
        frames = {}
        for i, cam in enumerate(self.cameras):
            ret, frame = cam.read()
            if ret and frame is not None:
                frames[i] = frame
        return frames

    def get_mosaic(self, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Crée une mosaïque de toutes les caméras.
        
        Args:
            target_size: (width, height) de l'image finale
            
        Returns:
            Image mosaïque np.ndarray
        """
        if target_size is None:
            target_size = self._mosaic_size

        n = len(self.cameras)
        if n == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # Calculer la grille
        if n == 1:
            cols, rows = 1, 1
        elif n == 2:
            cols, rows = 2, 1
        elif n <= 4:
            cols, rows = 2, 2
        elif n <= 6:
            cols, rows = 3, 2
        else:
            cols, rows = 4, 2

        cell_w = target_size[0] // cols
        cell_h = target_size[1] // rows

        mosaic = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        for i, cam in enumerate(self.cameras):
            row = i // cols
            col = i % cols

            if row >= rows:
                break

            x = col * cell_w
            y = row * cell_h

            ret, frame = cam.read()
            if ret and frame is not None:
                # Redimensionner
                resized = cv2.resize(frame, (cell_w, cell_h))
                mosaic[y:y + cell_h, x:x + cell_w] = resized

                # Label de la caméra
                label_bg = mosaic.copy()
                cv2.rectangle(label_bg, (x, y), (x + cell_w, y + 28),
                              (0, 0, 0), -1)
                cv2.addWeighted(label_bg, 0.6, mosaic, 0.4, 0, mosaic)

                # Nom + FPS
                cv2.putText(mosaic, f"{cam.name} | {cam.fps:.0f} FPS",
                            (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Indicateur connecté
                color = (0, 255, 0) if cam.is_connected else (0, 0, 255)
                cv2.circle(mosaic, (x + cell_w - 15, y + 14), 6, color, -1)

            else:
                # Caméra déconnectée
                cv2.rectangle(mosaic, (x, y), (x + cell_w, y + cell_h),
                              (30, 30, 30), -1)
                cv2.putText(mosaic, f"{cam.name} - DECONNECTEE",
                            (x + 20, y + cell_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 200), 2)

            # Bordure entre cellules
            cv2.rectangle(mosaic, (x, y), (x + cell_w - 1, y + cell_h - 1),
                          (50, 50, 50), 1)

        return mosaic

    def get_camera_count(self) -> int:
        return len(self.cameras)

    def get_active_count(self) -> int:
        return sum(1 for c in self.cameras if c.is_connected)

    def get_stats(self) -> dict:
        return {
            "total_cameras": len(self.cameras),
            "active_cameras": self.get_active_count(),
            "cameras": [
                {
                    "name": cam.name,
                    "source": str(cam.source),
                    "connected": cam.is_connected,
                    "fps": round(cam.fps, 1),
                    "frames": cam.frame_count,
                }
                for cam in self.cameras
            ]
        }
