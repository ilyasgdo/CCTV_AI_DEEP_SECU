"""
Benchmark de performance du pipeline YOLO-Pose.
Mesure le temps d'inférence moyen et le FPS théorique max.
"""
import time
import torch
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline.detector import PoseDetector


def benchmark(source=0, num_frames: int = 100):
    """
    Lance un benchmark du détecteur YOLO-Pose.

    Args:
        source: Source vidéo (0 = webcam)
        num_frames: Nombre de frames à tester
    """
    print("=" * 50)
    print("  BENCHMARK YOLO-POSE")
    print("=" * 50)

    # GPU info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"\n  GPU: {gpu.name}")
        print(f"  VRAM: {gpu.total_mem / 1024**2:.0f} MB")
    else:
        print("\n  ⚠ Pas de GPU détecté, benchmark sur CPU")

    # Init
    detector = PoseDetector()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir : {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Résolution: {width}x{height}")
    print(f"  Frames à tester: {num_frames}")

    # Warmup (5 frames)
    print("\n  Warmup...", end=" ")
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            detector.detect(frame)
    print("OK")

    # Benchmark
    print(f"  Benchmark en cours ({num_frames} frames)...")
    times = []
    det_counts = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        elapsed = time.perf_counter() - t0

        times.append(elapsed)
        det_counts.append(len(detections))

        if (i + 1) % 25 == 0:
            avg_ms = sum(times[-25:]) / 25 * 1000
            print(f"    Frame {i+1}/{num_frames} — {avg_ms:.1f} ms/frame")

    cap.release()

    # Résultats
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    avg_dets = sum(det_counts) / len(det_counts)

    print(f"\n{'='*50}")
    print(f"  RÉSULTATS BENCHMARK")
    print(f"{'='*50}")
    print(f"  Temps moyen  : {avg_time*1000:.1f} ms/frame")
    print(f"  Temps min    : {min_time*1000:.1f} ms/frame")
    print(f"  Temps max    : {max_time*1000:.1f} ms/frame")
    print(f"  FPS théorique: {1/avg_time:.0f}")
    print(f"  Détections   : {avg_dets:.1f} moy/frame")

    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  GPU Mem used : {mem_used:.0f} MB")
        print(f"  GPU Mem rsv  : {mem_reserved:.0f} MB")

    print(f"{'='*50}")

    # Verdict
    fps_max = 1 / avg_time
    if fps_max >= 30:
        print("  ✅ Performance EXCELLENTE (≥30 FPS)")
    elif fps_max >= 25:
        print("  ✅ Performance BONNE (≥25 FPS)")
    elif fps_max >= 15:
        print("  ⚠ Performance MOYENNE — réduire la résolution")
    else:
        print("  ❌ Performance FAIBLE — vérifier GPU / config")


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    benchmark(source)
