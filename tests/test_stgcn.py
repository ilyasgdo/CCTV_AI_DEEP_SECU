"""
Test complet du système ST-GCN intégré au pipeline.
"""
import cv2
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline.detector import PoseDetector
from src.behavior.action_classifier import ActionClassifier
from src.behavior.loitering_detector import LoiteringDetector
from src.utils.drawing import draw_detections, draw_fps, draw_alert


def test_stgcn_pipeline(source=0):
    """Test le pipeline complet avec ST-GCN et détection de maraudage."""

    print("[TEST ST-GCN] Initialisation...")
    detector = PoseDetector()
    classifier = ActionClassifier(device="cuda")
    loitering = LoiteringDetector(timeout=30)  # 30s pour le test

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir : {source}")
        return False

    # Zones de maraudage
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    loitering.set_default_zones(w, h)

    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0

    print("[TEST ST-GCN] Démarrage ('q' pour quitter)")

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # 1. Détection
        detections = detector.detect(frame)
        active_ids = {d.track_id for d in detections}

        # 2. Mise à jour des buffers + classification
        for det in detections:
            classifier.update(det.track_id, det.keypoints_xy)

        predictions = classifier.classify(detector.frame_count)

        # 3. Appliquer les actions aux détections
        for det in detections:
            det.action = classifier.get_action(det.track_id)

            # 4. Vérifier le maraudage
            loiter_result = loitering.update(det.track_id, det.center)
            if loiter_result:
                det.action = f"MARAUDAGE ({loiter_result[1]:.0f}s)"

        # 5. Vérifier les alertes ST-GCN
        alerts = classifier.check_alerts()

        # 6. Affichage
        frame = loitering.draw_zones(frame)
        annotated = draw_detections(frame, detections)
        annotated = draw_fps(annotated, current_fps)

        for alert_tid, alert_action, alert_conf in alerts:
            draw_alert(annotated,
                       f"{alert_action} (ID:{alert_tid} - {alert_conf:.0%})")

        # Nettoyage
        classifier.cleanup_lost_ids(active_ids)
        loitering.cleanup_lost_ids(active_ids)

        # FPS
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()

        cv2.imshow("CCTV AI - Test ST-GCN", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[TEST ST-GCN] Stats classificateur : {classifier.get_stats()}")
    print(f"[TEST ST-GCN] Stats maraudage : {loitering.get_stats()}")
    return True


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    success = test_stgcn_pipeline(source)
    print(f"\n{'✅ TEST RÉUSSI' if success else '❌ TEST ÉCHOUÉ'}")
