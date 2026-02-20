"""
CCTV AI DEEP SECU — Point d'Entrée Principal

Pipeline complet :
  Thread 1 : Capture vidéo (OpenCV)
  Thread 2 : Détection + Suivi (YOLOv8-Pose + ByteTrack)
  Thread 3 : Analyse (ST-GCN + InsightFace + SQLite) — cadence réduite

Usage :
  python src/main.py                          # Webcam
  python src/main.py --source video.mp4       # Fichier vidéo
  python src/main.py --source rtsp://...      # Flux RTSP
"""
import cv2
import time
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import TARGET_FPS
from src.pipeline.capture import VideoCapture
from src.pipeline.detector import PoseDetector
from src.pipeline.analyzer import Analyzer
from src.utils.drawing import draw_detections, draw_fps, draw_alert


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="CCTV AI DEEP SECU — Système de Vidéosurveillance Intelligente"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Source vidéo : 0 (webcam), chemin fichier, ou URL RTSP"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Désactiver l'affichage (mode headless)"
    )
    parser.add_argument(
        "--no-face", action="store_true",
        help="Désactiver la reconnaissance faciale"
    )
    parser.add_argument(
        "--no-stgcn", action="store_true",
        help="Désactiver l'analyse ST-GCN"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Source vidéo
    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 60)
    print("  CCTV AI DEEP SECU — Démarrage du système")
    print("=" * 60)

    # === INITIALISATION ===
    print("\n[1/3] Initialisation de la capture vidéo...")
    capture = VideoCapture(source)
    capture.start()

    print("\n[2/3] Initialisation du détecteur YOLOv8-Pose...")
    detector = PoseDetector()

    print("\n[3/3] Initialisation de l'analyseur (ST-GCN + InsightFace + DB)...")
    analyzer = Analyzer(
        frame_width=capture.width,
        frame_height=capture.height
    )
    analyzer.start()

    print("\n" + "=" * 60)
    print("  ✅ SYSTÈME PRÊT — Appuyez sur 'q' pour quitter")
    print("  📊 Appuyez sur 's' pour les statistiques")
    print("=" * 60 + "\n")

    # === BOUCLE PRINCIPALE ===
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0

    try:
        while True:
            # 1. Lire la frame
            success, frame = capture.read()
            if not success:
                time.sleep(0.001)
                continue

            # 2. Détection YOLOv8-Pose (chaque frame)
            detections = detector.detect(frame)

            # 3. Analyse (ST-GCN + InsightFace à cadence réduite)
            analyzer.process(detections, frame, detector.frame_count)

            # 4. Appliquer les résultats aux détections
            analyzer.apply_to_detections(detections)

            # 5. Affichage
            if not args.no_display:
                # Dessiner les zones de maraudage
                frame = analyzer.loitering.draw_zones(frame)

                # Dessiner les détections
                annotated = draw_detections(frame, detections)
                annotated = draw_fps(annotated, current_fps)

                # Infos supplémentaires
                db_stats = analyzer.db.get_stats()
                cv2.putText(annotated,
                           f"Personnes: {len(detections)} | "
                           f"DB: {db_stats['currently_present']} present(s) | "
                           f"Alertes: {db_stats['total_alerts']}",
                           (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                # Alertes en cours
                results = analyzer.get_results()
                for tid, r in results.items():
                    if r.get("loitering"):
                        draw_alert(annotated,
                                  f"MARAUDAGE ID:{tid} ({r['loitering'][1]:.0f}s)")

                cv2.imshow("CCTV AI DEEP SECU", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print(f"\n{'='*50}")
                    stats = analyzer.get_stats()
                    print(f"📊 STATISTIQUES SYSTÈME :")
                    print(f"  FPS          : {current_fps:.1f}")
                    print(f"  Frames       : {detector.frame_count}")
                    print(f"  Détections   : {len(detections)}")
                    print(f"  BDD          : {stats['database']}")
                    print(f"  Face Matcher : {stats['face_matcher']}")
                    print(f"  Classifier   : {stats['classifier']}")
                    print(f"  Maraudage    : {stats['loitering']}")
                    print(f"{'='*50}\n")

            # Calcul FPS
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

    except KeyboardInterrupt:
        print("\n[SYSTEM] Interruption utilisateur...")

    finally:
        # === ARRÊT PROPRE ===
        print("\n[SYSTEM] Arrêt du système...")
        capture.stop()
        analyzer.stop()
        cv2.destroyAllWindows()

        print(f"\n{'='*50}")
        print(f"  SESSION TERMINÉE")
        print(f"  Frames traitées : {detector.frame_count}")
        print(f"  FPS moyen       : {current_fps:.1f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
