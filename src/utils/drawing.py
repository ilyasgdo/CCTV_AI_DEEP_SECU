"""
Module d'affichage : dessine les boîtes, squelettes, tags et statistiques par personne.
"""
import cv2
import numpy as np
from typing import List, Dict
import time

# Connexions du squelette COCO pour dessiner les os
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),       # Nez → Yeux
    (1, 3), (2, 4),       # Yeux → Oreilles
    (5, 6),                # Épaule gauche → Épaule droite
    (5, 7), (7, 9),       # Épaule G → Coude G → Poignet G
    (6, 8), (8, 10),      # Épaule D → Coude D → Poignet D
    (5, 11), (6, 12),     # Épaules → Hanches
    (11, 12),              # Hanche G → Hanche D
    (11, 13), (13, 15),   # Hanche G → Genou G → Cheville G
    (12, 14), (14, 16),   # Hanche D → Genou D → Cheville D
]

# Couleurs pour différents IDs (palette de 20 couleurs)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (128, 128, 255), (255, 128, 128), (128, 255, 128), (255, 128, 255),
    (128, 255, 255), (255, 255, 128), (192, 0, 0), (0, 192, 0),
]

# Couleurs par type de tag
TAG_COLORS = {
    "identifie":  (0, 180, 0),      # Vert — personne reconnue
    "inconnu":    (0, 140, 255),     # Orange — personne inconnue
    "alerte":     (0, 0, 220),       # Rouge — alerte active
    "maraudage":  (0, 80, 255),      # Orange foncé — maraudage
}


def get_color(track_id: int) -> tuple:
    """Retourne une couleur unique basée sur l'ID de suivi."""
    return COLORS[track_id % len(COLORS)]


def format_duration(seconds: float) -> str:
    """Formate une durée en texte lisible."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}min{s:02d}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def draw_tag(frame, text, x, y, bg_color, text_color=(255, 255, 255), scale=0.45):
    """Dessine un tag arrondi avec texte."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad_x, pad_y = 6, 4
    # Fond arrondi
    cv2.rectangle(frame, (x, y - th - pad_y * 2), (x + tw + pad_x * 2, y), bg_color, -1)
    cv2.putText(frame, text, (x + pad_x, y - pad_y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)
    return tw + pad_x * 2 + 4  # Retourne la largeur du tag + marge


def draw_detections(frame: np.ndarray, detections: list,
                    person_stats: Dict[int, dict] = None,
                    draw_skeleton: bool = True,
                    draw_bbox: bool = True,
                    draw_label: bool = True) -> np.ndarray:
    """
    Dessine les détections avec tags et statistiques enrichis.

    Args:
        frame: Image BGR
        detections: Liste de PersonDetection
        person_stats: Dictionnaire {track_id: {presence_time, action_history, ...}}
        draw_skeleton: Dessiner le squelette
        draw_bbox: Dessiner la boîte englobante
        draw_label: Dessiner le label (ID + nom + action + stats)

    Returns:
        Frame annotée
    """
    annotated = frame.copy()

    for det in detections:
        color = get_color(det.track_id)
        is_known = det.name != "INCONNU" and det.name != "N/A"
        has_alert = "MARAUDAGE" in det.action or det.action in ["chute", "donner_un_coup"]

        # Couleur de la boîte selon le statut
        if has_alert:
            box_color = TAG_COLORS["alerte"]
        elif is_known:
            box_color = TAG_COLORS["identifie"]
        else:
            box_color = TAG_COLORS["inconnu"]

        # --- Boîte englobante ---
        if draw_bbox:
            x1, y1, x2, y2 = det.bbox.astype(int)
            # Boîte avec coins
            thickness = 3 if has_alert else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            # Coins décoratifs
            corner_len = min(20, (x2 - x1) // 4)
            for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                    (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(annotated, (cx, cy), (cx + dx * corner_len, cy), box_color, 3)
                cv2.line(annotated, (cx, cy), (cx, cy + dy * corner_len), box_color, 3)

        # --- Tags et labels ---
        if draw_label:
            x1, y1 = det.bbox[:2].astype(int)
            x2 = int(det.bbox[2])
            tag_y = y1 - 8

            # Ligne 1 : Nom + ID
            name_text = det.name if is_known else "INCONNU"
            id_text = f"ID:{det.track_id}"

            # Fond principal
            main_label = f"  {name_text}  "
            (mlw, mlh), _ = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, tag_y - mlh - 8), (x1 + mlw + 60, tag_y),
                          box_color, -1)
            cv2.putText(annotated, main_label, (x1, tag_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            # ID en petit
            cv2.putText(annotated, id_text, (x1 + mlw + 4, tag_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            # Ligne 2 : Tags sous la boîte
            x2_box = int(det.bbox[2])
            y2_box = int(det.bbox[3])
            tag_x = x1
            tag_y_bottom = y2_box + 16

            # Tag action
            if det.action != "N/A":
                action_color = TAG_COLORS["alerte"] if has_alert else (100, 100, 100)
                w = draw_tag(annotated, det.action.upper(), tag_x, tag_y_bottom,
                             action_color)
                tag_x += w

            # Tags de stats si disponibles
            if person_stats and det.track_id in person_stats:
                stats = person_stats[det.track_id]

                # Tag temps de présence
                if "presence_time" in stats:
                    dur = format_duration(stats["presence_time"])
                    w = draw_tag(annotated, f"⏱ {dur}", tag_x, tag_y_bottom,
                                 (80, 80, 80))
                    tag_x += w

                # Ligne 3 : Action dominante + objets détectés
                tag_y_line3 = tag_y_bottom + 18

                # Top action avec durée
                if "top_action" in stats and stats["top_action"]:
                    action_name, action_dur = stats["top_action"]
                    if action_name != "N/A" and action_dur > 1:
                        dur_str = format_duration(action_dur)
                        w = draw_tag(annotated, f"{action_name} {dur_str}",
                                     x1, tag_y_line3, (120, 60, 0))
                        tag_x_l3 = x1 + w
                    else:
                        tag_x_l3 = x1
                else:
                    tag_x_l3 = x1

                # Objets détectés (basés sur la pose)
                if "pose_objects" in stats:
                    for obj in stats["pose_objects"]:
                        w = draw_tag(annotated, obj, tag_x_l3, tag_y_line3,
                                     (150, 80, 0))
                        tag_x_l3 += w

        # --- Squelette ---
        if draw_skeleton:
            kpts = det.keypoints  # (17, 3)

            # Dessiner les connexions d'abord (sous les points)
            for (a, b) in SKELETON_CONNECTIONS:
                xa, ya, ca = kpts[a]
                xb, yb, cb = kpts[b]
                if ca > 0.5 and cb > 0.5:
                    cv2.line(annotated, (int(xa), int(ya)), (int(xb), int(yb)),
                             color, 2, cv2.LINE_AA)

            # Dessiner les points
            for j in range(17):
                x, y, conf = kpts[j]
                if conf > 0.5:
                    cv2.circle(annotated, (int(x), int(y)), 4, color, -1)
                    cv2.circle(annotated, (int(x), int(y)), 4, (255, 255, 255), 1)

    return annotated


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Affiche le FPS en haut à gauche avec fond."""
    text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (8, 8), (tw + 16, th + 16), (0, 0, 0), -1)
    color = (0, 255, 0) if fps >= 25 else (0, 200, 255) if fps >= 15 else (0, 0, 255)
    cv2.putText(frame, text, (12, th + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def draw_status_bar(frame: np.ndarray, num_persons: int, db_stats: dict,
                    person_stats: Dict[int, dict] = None,
                    counter_stats: dict = None) -> np.ndarray:
    """Dessine la barre de statut en haut de l'écran."""
    h, w = frame.shape[:2]

    # Fond semi-transparent en haut
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Infos
    present = db_stats.get('currently_present', 0)
    alerts = db_stats.get('total_alerts', 0)

    x = 120
    cv2.putText(frame, f"Personnes: {num_persons}", (x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    x += 170

    # Compteur entrées/sorties
    if counter_stats:
        entries = counter_stats.get("total_entries", 0)
        exits = counter_stats.get("total_exits", 0)
        cv2.putText(frame, f"| IN:{entries}", (x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA)
        x += 90
        cv2.putText(frame, f"OUT:{exits}", (x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1, cv2.LINE_AA)
        x += 90

    cv2.putText(frame, f"| BDD: {present}", (x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    x += 120

    alert_color = (0, 0, 255) if alerts > 0 else (100, 100, 100)
    cv2.putText(frame, f"| Alertes: {alerts}", (x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, alert_color, 1, cv2.LINE_AA)

    return frame


def draw_alert(frame: np.ndarray, message: str,
               position: tuple = None) -> np.ndarray:
    """Affiche une alerte rouge clignotante."""
    h, w = frame.shape[:2]
    if position is None:
        position = (w // 2 - 200, 70)

    # Fond rouge semi-transparent
    overlay = frame.copy()
    (tw, th), _ = cv2.getTextSize(f"⚠ {message}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(overlay, (position[0] - 10, position[1] - 30),
                  (position[0] + tw + 20, position[1] + 10), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    cv2.putText(frame, f"! {message}", position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def draw_side_panel(frame: np.ndarray, person_stats: Dict[int, dict],
                    panel_width: int = 280) -> np.ndarray:
    """
    Dessine un panneau latéral avec les statistiques détaillées par personne.
    """
    h, w = frame.shape[:2]

    # Créer le panneau
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    # Titre
    cv2.putText(panel, "SUIVI EN DIRECT", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.line(panel, (10, 40), (panel_width - 10, 40), (80, 80, 80), 1)

    y_offset = 60

    if not person_stats:
        cv2.putText(panel, "Aucune personne", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    else:
        for tid, stats in sorted(person_stats.items()):
            if y_offset > h - 40:
                break

            name = stats.get("name", "INCONNU")
            is_known = name != "INCONNU"
            presence = stats.get("presence_time", 0)

            # Nom + ID
            name_color = (0, 255, 0) if is_known else (0, 140, 255)
            cv2.putText(panel, f"{name}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, name_color, 1, cv2.LINE_AA)
            cv2.putText(panel, f"ID:{tid}", (panel_width - 60, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
            y_offset += 20

            # Temps de présence
            dur = format_duration(presence)
            cv2.putText(panel, f"  Presence: {dur}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 18

            # Action en cours
            action = stats.get("current_action", "N/A")
            if action != "N/A":
                action_color = (0, 0, 255) if action in ["chute", "donner_un_coup"] else (180, 180, 180)
                cv2.putText(panel, f"  Action: {action}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, action_color, 1)
                y_offset += 18

            # Historique des actions avec durées
            action_hist = stats.get("action_durations", {})
            if action_hist:
                # Trier par durée décroissante
                sorted_actions = sorted(action_hist.items(), key=lambda x: x[1], reverse=True)
                for act_name, act_dur in sorted_actions[:3]:  # Top 3
                    if act_name == "N/A" or act_dur < 1:
                        continue
                    dur_str = format_duration(act_dur)
                    # Barre de proportion
                    bar_width = min(int(act_dur / max(presence, 1) * 120), 120)
                    cv2.rectangle(panel, (15, y_offset - 10),
                                  (15 + bar_width, y_offset + 2), (60, 60, 60), -1)
                    cv2.putText(panel, f"  {act_name}: {dur_str}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
                    y_offset += 16

            # Objets détectés
            objects = stats.get("pose_objects", [])
            if objects:
                obj_text = ", ".join(objects)
                cv2.putText(panel, f"  Objets: {obj_text}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
                y_offset += 16

            # Séparateur
            y_offset += 6
            cv2.line(panel, (10, y_offset), (panel_width - 10, y_offset), (60, 60, 60), 1)
            y_offset += 12

    # Combiner frame + panel
    combined = np.hstack([frame, panel])
    return combined
