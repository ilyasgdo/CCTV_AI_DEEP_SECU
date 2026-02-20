"""
CCTV AI DEEP SECU — Outil de Capture de Visages pour la Whitelist

Permet de capturer facilement les visages des personnes autorisées
directement depuis la webcam ou un flux vidéo.

Usage :
    python tools/whitelist_capture.py                  # Webcam
    python tools/whitelist_capture.py --source URL     # Flux IP
"""
import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import WHITELIST_DIR


def draw_instructions(frame, name_mode=False, current_name="", captures=None):
    """Dessine les instructions sur la frame."""
    h, w = frame.shape[:2]
    
    # Fond semi-transparent en bas
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 160), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    if name_mode:
        # Mode saisie de nom
        cv2.putText(frame, "ENTRER LE NOM DE LA PERSONNE",
                    (20, h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"Nom : {current_name}_",
                    (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "[ENTREE] Valider  |  [ECHAP] Annuler",
                    (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        # Mode normal
        cv2.putText(frame, "[C] Capturer un visage  |  [B] Construire whitelist  |  [Q] Quitter",
                    (20, h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        cv2.putText(frame, "Cadrez le visage de la personne, puis appuyez sur 'C'",
                    (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Liste des captures effectuées
        if captures:
            names_summary = {}
            for name in captures:
                names_summary[name] = names_summary.get(name, 0) + 1
            text = " | ".join([f"{name}: {count} photo(s)" for name, count in names_summary.items()])
            cv2.putText(frame, f"Captures : {text}",
                        (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Aucune capture pour l'instant",
                        (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
    
    return frame


def draw_face_boxes(frame, face_cascade):
    """Dessine les boîtes de détection de visages."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        # Boîte verte avec coins arrondis
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "VISAGE", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame, faces


def draw_success(frame, name, count):
    """Affiche un message de succès."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 100, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    cv2.putText(frame, "CAPTURE REUSSIE !",
                (w//4 + 30, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"{name} - Photo #{count}",
                (w//4 + 30, h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def whitelist_capture(source=0):
    """Outil interactif de capture de visages."""
    
    print("=" * 60)
    print("  CCTV AI DEEP SECU — Capture de Visages (Whitelist)")
    print("=" * 60)
    print()
    print("  COMMANDES :")
    print("    [C]  Capturer le visage visible")
    print("    [B]  Construire la whitelist (quand termine)")
    print("    [Q]  Quitter")
    print()
    
    # Ouvrir la source vidéo
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  ERREUR : Impossible d'ouvrir la source : {source}")
        return
    
    # Détecteur de visages OpenCV (léger, pour la preview uniquement)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Créer le dossier whitelist
    whitelist_dir = Path(WHITELIST_DIR)
    whitelist_dir.mkdir(parents=True, exist_ok=True)
    
    captures = []  # Liste des noms capturés
    name_mode = False
    current_name = ""
    pending_frame = None
    show_success = False
    success_time = 0
    success_name = ""
    success_count = 0
    
    # Compter les photos existantes par nom
    existing_counts = {}
    for f in whitelist_dir.glob("*.jpg"):
        name_part = f.stem.rsplit("_", 1)[0].lower()
        existing_counts[name_part] = existing_counts.get(name_part, 0) + 1
    for f in whitelist_dir.glob("*.jpeg"):
        name_part = f.stem.rsplit("_", 1)[0].lower()
        existing_counts[name_part] = existing_counts.get(name_part, 0) + 1
    for f in whitelist_dir.glob("*.png"):
        name_part = f.stem.rsplit("_", 1)[0].lower()
        existing_counts[name_part] = existing_counts.get(name_part, 0) + 1
    
    print(f"  Photos existantes : {sum(existing_counts.values())} "
          f"({len(existing_counts)} personne(s))")
    print(f"  Dossier : {whitelist_dir}")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str) and not source.startswith(("http", "rtsp")):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        display = frame.copy()
        
        # Message de succès temporaire
        if show_success and (time.time() - success_time) < 1.5:
            display = draw_success(display, success_name, success_count)
            cv2.imshow("CCTV AI - Capture Whitelist", display)
            cv2.waitKey(1)
            continue
        else:
            show_success = False
        
        if name_mode:
            # Mode saisie de nom : montrer la frame capturée
            display = pending_frame.copy() if pending_frame is not None else display
            display = draw_instructions(display, name_mode=True, 
                                        current_name=current_name)
            cv2.imshow("CCTV AI - Capture Whitelist", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ECHAP → annuler
                name_mode = False
                current_name = ""
                pending_frame = None
                print("  Capture annulee")
            elif key == 13:  # ENTREE → valider
                if current_name.strip():
                    # Sauvegarder la photo
                    name_lower = current_name.strip().lower()
                    count = existing_counts.get(name_lower, 0) + 1
                    existing_counts[name_lower] = count
                    
                    filename = f"{name_lower}_{count}.jpg"
                    filepath = whitelist_dir / filename
                    cv2.imwrite(str(filepath), pending_frame)
                    
                    captures.append(current_name.strip().capitalize())
                    print(f"  Sauvegarde : {filepath}")
                    
                    # Afficher succès
                    show_success = True
                    success_time = time.time()
                    success_name = current_name.strip().capitalize()
                    success_count = count
                    
                    name_mode = False
                    current_name = ""
                    pending_frame = None
                else:
                    print("  Nom vide, veuillez entrer un nom")
            elif key == 8:  # BACKSPACE
                current_name = current_name[:-1]
            elif 32 <= key <= 126:  # Caractère imprimable
                current_name += chr(key)
        else:
            # Mode normal : preview avec détection de visages
            display, faces = draw_face_boxes(display, face_cascade)
            
            # Nombre de visages
            if len(faces) > 0:
                cv2.putText(display, f"{len(faces)} visage(s) detecte(s)",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Aucun visage detecte",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            display = draw_instructions(display, captures=captures)
            cv2.imshow("CCTV AI - Capture Whitelist", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                if len(faces) > 0:
                    # Capturer : prendre la plus grande face
                    # Sauvegarder la frame complète (InsightFace détectera le visage)
                    pending_frame = frame.copy()
                    name_mode = True
                    current_name = ""
                    print("  Visage capture ! Entrez le nom...")
                else:
                    print("  Aucun visage detecte ! Positionnez-vous face a la camera.")
            elif key == ord('b'):
                # Construire la whitelist
                cap.release()
                cv2.destroyAllWindows()
                print()
                print("  Construction de la whitelist...")
                print("  (Cela peut prendre 30-60 secondes)")
                print()
                
                try:
                    from src.face_recognition.encoder import FaceEncoder
                    encoder = FaceEncoder()
                    whitelist = encoder.build_whitelist()
                    if whitelist:
                        encoder.save_whitelist(whitelist)
                        print()
                        print("  " + "=" * 50)
                        print(f"  WHITELIST CONSTRUITE : {len(whitelist)} personne(s)")
                        for name in whitelist:
                            print(f"    → {name}")
                        print("  " + "=" * 50)
                    else:
                        print("  Aucun visage encode. Verifiez les photos.")
                except Exception as e:
                    print(f"  ERREUR lors de la construction : {e}")
                
                input("\n  Appuyez sur ENTREE pour continuer...")
                return
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Résumé
    if captures:
        print()
        print("  " + "=" * 50)
        print(f"  CAPTURES EFFECTUEES : {len(captures)}")
        names_count = {}
        for name in captures:
            names_count[name] = names_count.get(name, 0) + 1
        for name, count in names_count.items():
            print(f"    → {name} : {count} photo(s)")
        print()
        print("  Pour construire la whitelist, relancez et appuyez sur [B]")
        print("  Ou executez :")
        print('    python tools/whitelist_capture.py --build')
        print("  " + "=" * 50)


def build_only():
    """Construit la whitelist sans ouvrir la caméra."""
    print("=" * 60)
    print("  Construction de la Whitelist")
    print("=" * 60)
    
    from src.face_recognition.encoder import FaceEncoder
    encoder = FaceEncoder()
    whitelist = encoder.build_whitelist()
    if whitelist:
        encoder.save_whitelist(whitelist)
        print()
        print(f"  WHITELIST : {len(whitelist)} personne(s)")
        for name in whitelist:
            print(f"    → {name}")
    else:
        print("  Aucun visage. Ajoutez des photos dans data/whitelist_photos/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Capture de visages pour whitelist")
    parser.add_argument("--source", default="0", help="Source video (0=webcam, URL...)")
    parser.add_argument("--build", action="store_true", help="Construire la whitelist sans camera")
    args = parser.parse_args()
    
    if args.build:
        build_only()
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        whitelist_capture(source)
