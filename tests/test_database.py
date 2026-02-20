"""
Test d'intégration : Détection + BDD.
Vérifie que les entrées/sorties sont correctement enregistrées.
"""
import cv2
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline.detector import PoseDetector
from src.database.db_manager import DatabaseManager
from src.utils.drawing import draw_detections, draw_fps


def test_with_database(source=0):
    """Test le pipeline avec enregistrement en base de données."""

    detector = PoseDetector()
    db = DatabaseManager()  # Crée cctv_records.db

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir : {source}")
        return False

    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0

    print("[TEST DB] Démarrage ('q' pour quitter, 's' pour voir les stats)")

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        detections = detector.detect(frame)

        # Enregistrer les présences
        for det in detections:
            db.update_presence(det.track_id, det.name)

        # Vérifier les sorties (toutes les 30 frames)
        if detector.frame_count % 30 == 0:
            db.check_exits()

        # Affichage
        annotated = draw_detections(frame, detections)
        annotated = draw_fps(annotated, current_fps)

        # Stats en overlay
        stats = db.get_stats()
        cv2.putText(annotated, f"DB: {stats['currently_present']} present(s)",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()

        cv2.imshow("CCTV AI - Test Database", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Afficher les stats dans la console
            print(f"\n{'='*40}")
            print(f"Stats : {db.get_stats()}")
            print(f"Historique récent :")
            for r in db.get_history(10):
                print(f"  {r}")
            print(f"{'='*40}\n")

    cap.release()
    cv2.destroyAllWindows()

    # Résumé final
    print(f"\n{'='*50}")
    print(f"RÉSUMÉ BASE DE DONNÉES :")
    stats = db.get_stats()
    print(f"  Total enregistrements : {stats['total_records']}")
    print(f"  Actuellement présents : {stats['currently_present']}")
    print(f"  Alertes totales       : {stats['total_alerts']}")

    print(f"\nHistorique complet :")
    for record in db.get_history():
        duration = record.get('duration_s')
        dur_str = f"{duration:.0f}s" if duration else "N/A"
        print(f"  ID:{record['track_id']} | {record['name']} | "
              f"Entrée: {record['entry_time']} | "
              f"Status: {record['status']} | "
              f"Durée: {dur_str}")

    db.close()
    return True


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    success = test_with_database(source)
    print(f"\n{'✅ TEST RÉUSSI' if success else '❌ TEST ÉCHOUÉ'}")
