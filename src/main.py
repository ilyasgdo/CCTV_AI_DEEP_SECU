"""
CCTV AI DEEP SECU — Point d'Entrée Principal

Pipeline complet :
  Thread 1 : Capture vidéo (OpenCV)
  Thread 2 : Détection + Suivi (YOLOv8-Pose + ByteTrack)
  Thread 3 : Analyse (Actions + InsightFace + Objets + SQLite)
  Thread 4 : Dashboard Web (Flask)

Features :
  - Détection de personnes + squelette 17 keypoints
  - Reconnaissance faciale (InsightFace)
  - Analyse d'actions (géométrique)
  - Détection d'objets portés (YOLOv8n)
  - Comptage entrées/sorties
  - Heatmap de mouvement
  - Dashboard web temps réel

Usage :
  python src/main.py                          # Webcam
  python src/main.py --source video.mp4       # Fichier vidéo
  python src/main.py --source rtsp://...      # Flux RTSP
"""
import cv2
import time
import argparse
import sys
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import TARGET_FPS
from src.pipeline.capture import VideoCapture
from src.pipeline.detector import PoseDetector
from src.pipeline.analyzer import Analyzer
from src.behavior.people_counter import PeopleCounter
from src.behavior.identification_request import IdentificationRequester
from src.utils.heatmap import MovementHeatmap
from src.utils.drawing import (
    draw_detections, draw_fps, draw_alert,
    draw_status_bar, draw_side_panel, format_duration
)


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
        "--no-panel", action="store_true",
        help="Désactiver le panneau latéral de stats"
    )
    parser.add_argument(
        "--no-face", action="store_true",
        help="Désactiver la reconnaissance faciale"
    )
    parser.add_argument(
        "--no-stgcn", action="store_true",
        help="Désactiver l'analyse d'actions"
    )
    parser.add_argument(
        "--dashboard", action="store_true", default=True,
        help="Activer le dashboard web (défaut: activé)"
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Désactiver le dashboard web"
    )
    parser.add_argument(
        "--dashboard-port", type=int, default=5555,
        help="Port du dashboard web (défaut: 5555)"
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
    print("\n[1/5] Initialisation de la capture vidéo...")
    capture = VideoCapture(source)
    capture.start()

    print("\n[2/5] Initialisation du détecteur YOLOv8-Pose...")
    detector = PoseDetector()

    print("\n[3/5] Initialisation de l'analyseur (Actions + InsightFace + Objets + DB)...")
    analyzer = Analyzer(
        frame_width=capture.width,
        frame_height=capture.height
    )
    analyzer.start()

    print("\n[4/5] Initialisation du compteur de personnes...")
    counter = PeopleCounter(
        frame_width=capture.width,
        frame_height=capture.height
    )

    print("\n[5/6] Initialisation de la heatmap...")
    heatmap = MovementHeatmap(
        width=capture.width,
        height=capture.height
    )

    print("\n[6/6] Initialisation de l'identification des inconnus...")
    id_requester = IdentificationRequester(
        delay=10.0,      # 10s avant de demander
        cooldown=30.0,   # 30s entre deux demandes
        voice_enabled=True
    )

    # === State partagé pour le dashboard ===
    shared_state = {
        "frame": None,
        "fps": 0.0,
        "detections_count": 0,
        "person_stats": {},
        "counter_stats": {},
        "heatmap_stats": {},
        "analyzer_stats": {},
        "db_stats": {},
        "alerts": [],
    }

    # === Dashboard Web ===
    dashboard_thread = None
    if not args.no_dashboard:
        try:
            from src.dashboard.server import create_dashboard, run_dashboard
            app = create_dashboard(shared_state)
            dashboard_thread = threading.Thread(
                target=run_dashboard,
                args=(app, args.dashboard_port),
                daemon=True
            )
            dashboard_thread.start()
            print(f"\n  🌐 Dashboard : http://localhost:{args.dashboard_port}")
        except ImportError as e:
            print(f"\n  ⚠ Dashboard non disponible ({e}). Installer flask : pip install flask")
        except Exception as e:
            print(f"\n  ⚠ Erreur dashboard : {e}")

    print("\n" + "=" * 60)
    print("  ✅ SYSTÈME PRÊT — Appuyez sur 'q' pour quitter")
    print("  📊 [S] Stats  [P] Panel  [H] Heatmap  [C] Compteur  [I] Identification")
    if not args.no_dashboard:
        print(f"  🌐 Dashboard : http://localhost:{args.dashboard_port}")
    print("=" * 60 + "\n")

    # === BOUCLE PRINCIPALE ===
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0
    show_panel = not args.no_panel
    show_heatmap = False
    show_counter = True
    show_id_request = True

    try:
        while True:
            # 1. Lire la frame
            success, frame = capture.read()
            if not success:
                time.sleep(0.001)
                continue

            # 2. Détection YOLOv8-Pose (chaque frame)
            detections = detector.detect(frame)

            # 3. Analyse (Actions + InsightFace + Objets, cadence réduite)
            analyzer.process(detections, frame, detector.frame_count)

            # 4. Appliquer les résultats aux détections
            analyzer.apply_to_detections(detections)

            # 5. Comptage des personnes
            counter.update(detections)

            # 6. Mise à jour heatmap
            heatmap.update(detections, detector.frame_count)

            # 7. Vérifier les inconnus (demande d'identification)
            if show_id_request:
                id_requester.update(detections, analyzer.face_matcher)

            # 7. Affichage
            if not args.no_display:
                person_stats = analyzer.get_person_stats()

                # Dessiner les zones de maraudage
                annotated = analyzer.loitering.draw_zones(frame)

                # Dessiner le compteur de personnes
                if show_counter:
                    annotated = counter.draw(annotated)

                # Dessiner les détections avec tags et stats
                annotated = draw_detections(annotated, detections,
                                            person_stats=person_stats)

                # Heatmap overlay
                if show_heatmap:
                    annotated = heatmap.draw(annotated, alpha=0.35)

                # FPS
                annotated = draw_fps(annotated, current_fps)

                # Barre de statut enrichie
                db_stats = analyzer.db.get_stats()
                counter_stats = counter.get_stats()
                annotated = draw_status_bar(annotated, len(detections), db_stats,
                                            counter_stats=counter_stats)

                # Alertes en cours
                results = analyzer.get_results()
                alert_y = 70
                for tid, r in results.items():
                    if r.get("loitering"):
                        name = r.get("name", "INCONNU")
                        draw_alert(annotated,
                                   f"MARAUDAGE {name} ({r['loitering'][1]:.0f}s)",
                                   position=(annotated.shape[1] // 2 - 200, alert_y))
                        alert_y += 50

                # Identification request overlay (clignotant)
                if show_id_request:
                    annotated = id_requester.draw(annotated, detections)

                # Panneau latéral
                if show_panel:
                    annotated = draw_side_panel(annotated, person_stats)

                cv2.imshow("CCTV AI DEEP SECU", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    show_panel = not show_panel
                    print(f"  [PANEL] {'Activé' if show_panel else 'Désactivé'}")
                elif key == ord('h'):
                    show_heatmap = not show_heatmap
                    print(f"  [HEATMAP] {'Activé' if show_heatmap else 'Désactivé'}")
                elif key == ord('c'):
                    show_counter = not show_counter
                    print(f"  [COUNTER] {'Activé' if show_counter else 'Désactivé'}")
                elif key == ord('r'):
                    heatmap.reset()
                elif key == ord('i'):
                    show_id_request = not show_id_request
                    print(f"  [ID-REQUEST] {'Activé' if show_id_request else 'Désactivé'}")
                elif key == ord('s'):
                    print(f"\n{'='*60}")
                    stats = analyzer.get_stats()
                    cs = counter.get_stats()
                    hs = heatmap.get_stats()

                    print(f"📊 STATISTIQUES SYSTÈME :")
                    print(f"  FPS          : {current_fps:.1f}")
                    print(f"  Frames       : {detector.frame_count}")
                    print(f"  Détections   : {len(detections)}")
                    print(f"  BDD          : {stats['database']}")
                    print(f"  Face Matcher : {stats['face_matcher']}")
                    print(f"  Classifier   : {stats['classifier']}")
                    print(f"  Maraudage    : {stats['loitering']}")

                    print(f"\n🚪 COMPTEUR :")
                    print(f"  Entrées : {cs['total_entries']}")
                    print(f"  Sorties : {cs['total_exits']}")
                    print(f"  Présents: {cs['present']}")

                    print(f"\n🗺️  HEATMAP :")
                    print(f"  Points   : {hs['total_points']}")
                    print(f"  Intensité: {hs['max_intensity']:.1f}")

                    print(f"\n👤 STATS PAR PERSONNE :")
                    for tid, ps in person_stats.items():
                        name = ps.get("name", "INCONNU")
                        presence = ps.get("presence_time", 0)
                        action = ps.get("current_action", "N/A")
                        objects = ps.get("pose_objects", [])
                        actions = ps.get("action_durations", {})

                        print(f"\n  ID:{tid} — {name}")
                        print(f"    Présence : {format_duration(presence)}")
                        print(f"    Action   : {action}")
                        if actions:
                            for act, dur in sorted(actions.items(),
                                                   key=lambda x: x[1], reverse=True):
                                if act != "N/A" and dur > 0.5:
                                    print(f"    → {act}: {format_duration(dur)}")
                        if objects:
                            print(f"    Objets   : {', '.join(objects)}")

                    print(f"{'='*60}\n")

            # === Dashboard: mettre à jour l'état partagé ===
            shared_state["frame"] = frame.copy() if frame is not None else None
            shared_state["fps"] = current_fps
            shared_state["detections_count"] = len(detections)
            shared_state["person_stats"] = person_stats if not args.no_display else analyzer.get_person_stats()
            shared_state["counter_stats"] = counter.get_stats()
            shared_state["heatmap_stats"] = heatmap.get_stats()
            shared_state["db_stats"] = db_stats if not args.no_display else analyzer.db.get_stats()
            shared_state["analyzer_stats"] = analyzer.get_stats()

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

        # Exporter la heatmap
        heatmap.save(str(Path(__file__).parent.parent / "data" / "heatmap_session.png"))

        # Résumé final
        person_stats = analyzer.get_person_stats()
        cs = counter.get_stats()
        print(f"\n{'='*60}")
        print(f"  SESSION TERMINÉE")
        print(f"  Frames traitées : {detector.frame_count}")
        print(f"  FPS moyen       : {current_fps:.1f}")
        print(f"  Entrées         : {cs['total_entries']}")
        print(f"  Sorties         : {cs['total_exits']}")
        print(f"  Personnes vues  : {cs['total_entries']}")
        if person_stats:
            print(f"\n  RÉSUMÉ PAR PERSONNE :")
            for tid, ps in person_stats.items():
                name = ps.get("name", "INCONNU")
                presence = ps.get("presence_time", 0)
                actions = ps.get("action_durations", {})
                print(f"    {name} (ID:{tid}) — {format_duration(presence)}")
                for act, dur in sorted(actions.items(),
                                       key=lambda x: x[1], reverse=True):
                    if act != "N/A" and dur > 0.5:
                        print(f"      → {act}: {format_duration(dur)}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
