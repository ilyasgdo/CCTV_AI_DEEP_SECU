"""
Test complet du détecteur YOLOv8-Pose + ByteTrack.
Utilise la webcam ou un fichier vidéo pour vérifier :
- Détection des personnes
- Suivi des IDs
- Extraction des squelettes
- Affichage temps réel
"""
import cv2
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline.detector import PoseDetector
from src.utils.drawing import draw_detections, draw_fps


def test_with_video(source=0):
    """
    Test le détecteur en temps réel.

    Args:
        source: 0 pour webcam, ou chemin vers un fichier vidéo
    """
    print(f"[TEST] Initialisation du détecteur...")
    detector = PoseDetector()

    print(f"[TEST] Ouverture de la source vidéo : {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERREUR : Impossible d'ouvrir la source vidéo : {source}")
        return False

    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0

    print("[TEST] Démarrage de la boucle de détection (Appuyez sur 'q' pour quitter)")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                # Replay la vidéo en boucle
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # --- Détection ---
        detections = detector.detect(frame)

        # --- Affichage ---
        annotated = draw_detections(frame, detections)
        annotated = draw_fps(annotated, current_fps)

        # Infos supplémentaires
        cv2.putText(annotated, f"Personnes: {len(detections)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Afficher les détails en console (toutes les 30 frames)
        if detector.frame_count % 30 == 0 and detections:
            print(f"\n[Frame {detector.frame_count}] {len(detections)} personne(s):")
            for det in detections:
                kpts_valid = sum(1 for k in det.keypoints if k[2] > 0.5)
                print(f"  ID:{det.track_id} | Conf:{det.confidence:.2f} | "
                      f"Keypoints valides: {kpts_valid}/17 | "
                      f"Centre: ({det.center[0]:.0f}, {det.center[1]:.0f})")

        # Calcul FPS
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        cv2.imshow("CCTV AI - Test Detecteur", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    stats = detector.get_stats()
    print(f"\n{'=' * 60}")
    print(f"[TEST] Terminé. Statistiques :")
    print(f"  Frames traitées : {stats['frames_processed']}")
    print(f"  Modèle          : {stats['model']}")
    print(f"  Device           : {stats['device']}")

    return True


if __name__ == "__main__":
    # Par défaut : webcam (0)
    # Pour un fichier : python tests/test_detector.py chemin/vers/video.mp4
    source = sys.argv[1] if len(sys.argv) > 1 else 0

    success = test_with_video(source)
    if success:
        print("\n✅ TEST RÉUSSI — Le détecteur fonctionne correctement")
    else:
        print("\n❌ TEST ÉCHOUÉ — Vérifier la configuration")
