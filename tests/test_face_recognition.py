"""
Test complet de la reconnaissance faciale intégrée au détecteur.
"""
import cv2
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline.detector import PoseDetector
from src.face_recognition.matcher import FaceMatcher
from src.utils.drawing import draw_detections, draw_fps


def test_face_recognition(source=0):
    """Test la reconnaissance faciale en temps réel."""

    print("[TEST] Initialisation...")
    detector = PoseDetector()
    matcher = FaceMatcher()
    matcher.load_whitelist()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir : {source}")
        return False

    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0

    print("[TEST] Démarrage (Appuyez sur 'q' pour quitter)")

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # Détection
        detections = detector.detect(frame)

        # Reconnaissance faciale
        for det in detections:
            if matcher.should_scan(det.track_id, detector.frame_count):
                # Extraire la zone de la tête (tiers supérieur de la bbox)
                head = det.head_bbox.astype(int)
                x1, y1, x2, y2 = head
                # Sécuriser les bornes
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 > x1 + 20 and y2 > y1 + 20:  # Min 20px
                    face_crop = frame[y1:y2, x1:x2]
                    name, score = matcher.identify(
                        face_crop, det.track_id, detector.frame_count
                    )
                    det.name = name
            else:
                det.name = matcher.get_name(det.track_id)

        # Nettoyer les IDs perdus
        active_ids = {d.track_id for d in detections}
        matcher.cleanup_lost_ids(active_ids)

        # Affichage
        annotated = draw_detections(frame, detections)
        annotated = draw_fps(annotated, current_fps)

        # FPS
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        cv2.imshow("CCTV AI - Test Face Recognition", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    stats = matcher.get_stats()
    print(f"\n[TEST] Statistiques reconnaissance faciale :")
    print(f"  Personnes identifiées : {stats['identified_persons']}")
    return True


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    success = test_face_recognition(source)
    print(f"\n{'✅ TEST RÉUSSI' if success else '❌ TEST ÉCHOUÉ'}")
