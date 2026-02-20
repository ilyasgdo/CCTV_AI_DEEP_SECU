"""
CCTV AI DEEP SECU — Point d'Entrée Principal

Pipeline complet :
  Thread 1 : Capture vidéo (OpenCV) — Multi-caméras
  Thread 2 : Détection + Suivi (YOLO11-Pose + ByteTrack)
  Thread 3 : Analyse (Actions + InsightFace + Objets + SQLite)
  Thread 4 : Dashboard Web (Flask)
  Thread 5 : Enregistrement clips d'alertes

Features :
  - Détection de personnes + squelette 17 keypoints
  - Reconnaissance faciale (InsightFace)
  - Analyse d'actions (géométrique)
  - Détection d'objets portés (YOLO26n)
  - Comptage entrées/sorties
  - Heatmap de mouvement
  - Accueil vocal + Identification des inconnus
  - Enregistrement clips vidéo sur alerte
  - Rapports PDF automatiques
  - Multi-caméras avec mosaïque
  - Dashboard web temps réel

Usage :
  python src/main.py                          # Webcam (auto-détection)
  python src/main.py --source 0              # Webcam spécifique
  python src/main.py --source video.mp4      # Fichier vidéo
  python src/main.py --source rtsp://...     # Flux RTSP
  python src/main.py --multi-cam             # Multi-caméras auto-detect
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
from src.utils.clip_recorder import ClipRecorder
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
        "--multi-cam", action="store_true",
        help="Mode multi-caméras : détecte et affiche toutes les caméras"
    )
    parser.add_argument("--no-display", action="store_true",
                        help="Mode headless (pas d'affichage)")
    parser.add_argument("--no-panel", action="store_true",
                        help="Désactiver le panneau latéral")
    parser.add_argument("--no-face", action="store_true",
                        help="Désactiver la reconnaissance faciale")
    parser.add_argument("--no-stgcn", action="store_true",
                        help="Désactiver l'analyse d'actions")
    parser.add_argument("--dashboard", action="store_true", default=True,
                        help="Activer le dashboard web (défaut: activé)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Désactiver le dashboard web")
    parser.add_argument("--dashboard-port", type=int, default=5555,
                        help="Port du dashboard web (défaut: 5555)")
    parser.add_argument("--no-clips", action="store_true",
                        help="Désactiver l'enregistrement de clips")
    parser.add_argument("--report", action="store_true",
                        help="Générer un rapport PDF à la fin de la session")
    return parser.parse_args()


# === MODE MULTI-CAMÉRAS ===
def run_multi_camera():
    """Lance le mode multi-caméras (mosaïque)."""
    from src.pipeline.multi_camera import MultiCameraManager

    print("=" * 60)
    print("  CCTV AI DEEP SECU — Mode Multi-Caméras")
    print("=" * 60)

    manager = MultiCameraManager(max_cameras=8)
    n = manager.auto_detect()

    if n == 0:
        print("\n  ⚠ Aucune caméra détectée !")
        return

    manager.start_all()
    print(f"\n  ✅ {n} caméra(s) active(s) — Appuyez sur 'q' pour quitter")
    print("  📊 [M] Mode mosaïque  [1-8] Caméra plein écran  [Q] Quitter")

    fullscreen_cam = None  # None = mosaïque, int = index caméra plein écran

    try:
        while True:
            if fullscreen_cam is not None:
                # Mode plein écran d'une caméra
                ret, frame = manager.cameras[fullscreen_cam].read()
                if ret:
                    cam = manager.cameras[fullscreen_cam]
                    cv2.putText(frame, f"{cam.name} | {cam.fps:.0f} FPS",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                    cv2.imshow("CCTV AI DEEP SECU — Multi-Cam", frame)
            else:
                # Mode mosaïque
                mosaic = manager.get_mosaic()
                cv2.imshow("CCTV AI DEEP SECU — Multi-Cam", mosaic)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                fullscreen_cam = None
                print("  [MODE] Mosaïque")
            elif ord('1') <= key <= ord('8'):
                idx = key - ord('1')
                if idx < len(manager.cameras):
                    fullscreen_cam = idx
                    print(f"  [MODE] Plein écran — {manager.cameras[idx].name}")

    finally:
        manager.stop_all()
        cv2.destroyAllWindows()
        print("\n[MULTI-CAM] Arrêté.")


# === MODE PRINCIPAL (1 CAMÉRA) ===
def main():
    args = parse_args()

    # Mode multi-cam
    if args.multi_cam:
        run_multi_camera()
        return

    # Source vidéo
    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 60)
    print("  CCTV AI DEEP SECU — Démarrage du système")
    print("=" * 60)

    # === INITIALISATION ===
    steps = 8
    step = 1

    print(f"\n[{step}/{steps}] Initialisation de la capture vidéo...")
    capture = VideoCapture(source)
    capture.start()
    step += 1

    print(f"\n[{step}/{steps}] Initialisation du détecteur YOLO11-Pose...")
    detector = PoseDetector()
    step += 1

    print(f"\n[{step}/{steps}] Initialisation de l'analyseur (Actions + InsightFace + Objets + DB)...")
    analyzer = Analyzer(
        frame_width=capture.width,
        frame_height=capture.height
    )
    analyzer.start()
    step += 1

    print(f"\n[{step}/{steps}] Initialisation du compteur de personnes...")
    counter = PeopleCounter(
        frame_width=capture.width,
        frame_height=capture.height
    )
    step += 1

    print(f"\n[{step}/{steps}] Initialisation de la heatmap...")
    heatmap = MovementHeatmap(
        width=capture.width,
        height=capture.height
    )
    step += 1

    print(f"\n[{step}/{steps}] Initialisation de l'identification / accueil...")
    id_requester = IdentificationRequester(
        delay=10.0, cooldown=30.0, voice_enabled=True
    )
    step += 1

    # Clip recorder
    clip_recorder = None
    if not args.no_clips:
        print(f"\n[{step}/{steps}] Initialisation de l'enregistreur de clips...")
        clip_recorder = ClipRecorder(
            fps=25, buffer_seconds=5, post_seconds=5
        )
    step += 1

    # PDF Report generator (lazy)
    report_gen = None
    if args.report:
        try:
            from src.reports.pdf_report import ReportGenerator
            report_gen = ReportGenerator()
            print(f"\n[{step}/{steps}] Génération de rapport PDF activée")
        except ImportError:
            print(f"\n[{step}/{steps}] ⚠ fpdf2 non installé (pip install fpdf2)")

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
        "clips": [],
    }

    # === Dashboard Web ===
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
            print(f"\n  ⚠ Dashboard non disponible ({e})")
        except Exception as e:
            print(f"\n  ⚠ Erreur dashboard : {e}")

    print("\n" + "=" * 60)
    print("  ✅ SYSTÈME PRÊT — Appuyez sur 'q' pour quitter")
    print("  📊 [S] Stats  [P] Panel  [H] Heatmap  [C] Compteur")
    print("  🔊 [I] Identification  [R] Reset heatmap")
    if clip_recorder:
        print("  🎬 Clips d'alertes : ACTIVÉ (auto)")
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

    # Tracking des alertes pour les clips
    session_alerts = []

    try:
        while True:
            # 1. Lire la frame
            success, frame = capture.read()
            if not success:
                time.sleep(0.001)
                continue

            # 2. Ajouter la frame au buffer de clips
            if clip_recorder:
                clip_recorder.add_frame(frame)

            # 3. Détection YOLO11-Pose
            detections = detector.detect(frame)

            # 4. Analyse (Actions + InsightFace + Objets, cadence réduite)
            analyzer.process(detections, frame, detector.frame_count)

            # 5. Appliquer les résultats aux détections
            analyzer.apply_to_detections(detections)

            # 6. Comptage des personnes
            counter.update(detections)

            # 7. Mise à jour heatmap
            heatmap.update(detections, detector.frame_count)

            # 8. Vérifier les inconnus (demande d'identification)
            if show_id_request:
                id_requester.update(detections, analyzer.face_matcher)

            # 9. Vérifier les alertes → déclencher le clip
            if clip_recorder:
                results = analyzer.get_results()
                for tid, r in results.items():
                    # Alertes d'action
                    alert_action = r.get("action")
                    alert_conf = r.get("action_confidence", 0)
                    if alert_action and alert_conf >= 0.8:
                        from src.config import ALERT_ACTIONS
                        if alert_action in ALERT_ACTIONS:
                            name = r.get("name", "INCONNU")
                            clip_recorder.trigger_alert(
                                alert_type=alert_action,
                                track_id=tid,
                                name=name,
                                confidence=alert_conf
                            )
                            session_alerts.append({
                                "timestamp": time.strftime('%H:%M:%S'),
                                "type": alert_action,
                                "name": name,
                                "confidence": alert_conf,
                            })

                    # Alerte maraudage
                    if r.get("loitering"):
                        name = r.get("name", "INCONNU")
                        clip_recorder.trigger_alert(
                            alert_type="maraudage",
                            track_id=tid,
                            name=name,
                            confidence=1.0
                        )

            # 10. Affichage
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

                # Indicateur REC si clip en cours
                if clip_recorder and clip_recorder._recording:
                    h, w = annotated.shape[:2]
                    if int(time.time() * 2) % 2 == 0:
                        cv2.circle(annotated, (w - 30, 65), 10, (0, 0, 255), -1)
                        cv2.putText(annotated, "REC", (w - 80, 72),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 0, 255), 2, cv2.LINE_AA)

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

                    if clip_recorder:
                        clip_stats = clip_recorder.get_stats()
                        print(f"\n🎬 CLIPS :")
                        print(f"  Clips sauvegardés : {clip_stats['total_clips']}")
                        print(f"  Alertes captées   : {clip_stats['total_alerts_triggered']}")
                        print(f"  Buffer            : {clip_stats['buffer_size']} frames")

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
            if clip_recorder:
                shared_state["clips"] = clip_recorder.get_clips_list()

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

        # Générer le rapport PDF
        if args.report or report_gen:
            try:
                from src.reports.pdf_report import ReportGenerator
                if report_gen is None:
                    report_gen = ReportGenerator()

                person_stats = analyzer.get_person_stats()
                cs = counter.get_stats()

                session_data = {
                    "fps": current_fps,
                    "frames": detector.frame_count,
                    "total_persons": len(person_stats),
                    "total_alerts": len(session_alerts),
                    "total_clips": clip_recorder.total_clips if clip_recorder else 0,
                    "person_stats": person_stats,
                    "counter_stats": cs,
                    "alerts": session_alerts,
                }

                report_path = report_gen.generate(session_data)
                print(f"  📄 Rapport PDF : {report_path}")

            except Exception as e:
                print(f"  ⚠ Erreur rapport PDF : {e}")

        # Résumé final
        person_stats = analyzer.get_person_stats()
        cs = counter.get_stats()
        print(f"\n{'='*60}")
        print(f"  SESSION TERMINÉE")
        print(f"  Frames traitées : {detector.frame_count}")
        print(f"  FPS moyen       : {current_fps:.1f}")
        print(f"  Entrées         : {cs['total_entries']}")
        print(f"  Sorties         : {cs['total_exits']}")
        if clip_recorder:
            print(f"  Clips sauvés    : {clip_recorder.total_clips}")
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
