"""
Générateur de rapports PDF automatiques.

Crée un rapport PDF professionnel avec :
  - Résumé de la session (durée, personnes, alertes)
  - Graphiques (fréquentation par heure, types d'alertes)
  - Tableau des personnes détectées
  - Clips vidéo associés aux alertes
  - Statistiques du compteur entrées/sorties

Usage :
    gen = ReportGenerator()
    gen.generate(session_data, output_path="rapport.pdf")
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import tempfile

# Ajouter le répertoire racine au path
sys.path.append(str(Path(__file__).parent.parent.parent))


class ReportGenerator:
    """Génère des rapports PDF professionnels."""

    def __init__(self, output_dir: str = None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent.parent / "data" / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, session_data: dict, output_path: str = None) -> str:
        """
        Génère un rapport PDF complet.
        
        Args:
            session_data: Dict contenant toutes les stats de la session
            output_path: Chemin de sortie (auto-généré si None)
            
        Returns:
            Chemin du fichier PDF généré
        """
        try:
            from fpdf import FPDF
        except ImportError:
            print("[REPORT] ⚠ fpdf2 non installé. pip install fpdf2")
            return ""

        if output_path is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = str(self.output_dir / f"rapport_{ts}.pdf")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # === PAGE 1 : Couverture + Résumé ===
        pdf.add_page()
        self._draw_header(pdf)
        self._draw_summary(pdf, session_data)
        self._draw_counter_stats(pdf, session_data)

        # === PAGE 2 : Personnes détectées ===
        pdf.add_page()
        self._draw_header(pdf, "Personnes Détectées")
        self._draw_persons_table(pdf, session_data)

        # === PAGE 3 : Alertes ===
        pdf.add_page()
        self._draw_header(pdf, "Journal des Alertes")
        self._draw_alerts_section(pdf, session_data)

        # === PAGE 4 : Graphiques ===
        chart_path = self._generate_charts(session_data)
        if chart_path:
            pdf.add_page()
            self._draw_header(pdf, "Analyse Graphique")
            try:
                pdf.image(chart_path, x=10, y=50, w=190)
            except Exception:
                pdf.set_font('Helvetica', '', 10)
                pdf.cell(0, 10, "Graphiques non disponibles", ln=True)
            # Nettoyer le fichier temp
            try:
                os.unlink(chart_path)
            except Exception:
                pass

        # === Pied de page ===
        pdf.set_y(-30)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(128, 128, 128)
        ts_now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        pdf.cell(0, 10, f"CCTV AI DEEP SECU - Rapport genere le {ts_now}", align='C')

        # Sauvegarder
        pdf.output(output_path)
        print(f"  📄 RAPPORT PDF: {output_path}")
        return output_path

    def _draw_header(self, pdf, subtitle: str = "Rapport de Session"):
        """Dessine l'en-tête du rapport."""
        # Bande de titre
        pdf.set_fill_color(20, 25, 50)
        pdf.rect(0, 0, 210, 40, 'F')

        pdf.set_font('Helvetica', 'B', 22)
        pdf.set_text_color(0, 180, 255)
        pdf.set_xy(10, 8)
        pdf.cell(0, 12, "CCTV AI DEEP SECU", ln=True)

        pdf.set_font('Helvetica', '', 12)
        pdf.set_text_color(200, 200, 200)
        pdf.set_x(10)
        pdf.cell(0, 8, subtitle)

        # Date
        pdf.set_font('Helvetica', '', 9)
        pdf.set_xy(140, 15)
        pdf.cell(0, 8, datetime.now().strftime('%d/%m/%Y %H:%M'))

        pdf.set_y(48)
        pdf.set_text_color(0, 0, 0)

    def _draw_summary(self, pdf, data: dict):
        """Section résumé."""
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(0, 100, 200)
        pdf.cell(0, 10, "Resume de la Session", ln=True)
        pdf.ln(3)

        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(50, 50, 50)

        fps = data.get("fps", 0)
        frames = data.get("frames", 0)
        duration = frames / max(fps, 1) if fps > 0 else 0
        dur_str = f"{int(duration // 60)}m {int(duration % 60)}s"

        items = [
            ("Duree de session", dur_str),
            ("FPS moyen", f"{fps:.1f}"),
            ("Frames traitees", str(frames)),
            ("Personnes detectees", str(data.get("total_persons", 0))),
            ("Alertes declenchees", str(data.get("total_alerts", 0))),
            ("Clips enregistres", str(data.get("total_clips", 0))),
        ]

        for label, value in items:
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(90, 8, f"  {label}:", border=0)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(0, 8, value, ln=True)

        pdf.ln(5)

    def _draw_counter_stats(self, pdf, data: dict):
        """Section compteur entrées/sorties."""
        counter = data.get("counter_stats", {})
        if not counter:
            return

        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(0, 100, 200)
        pdf.cell(0, 10, "Compteur Entrees / Sorties", ln=True)
        pdf.ln(3)

        pdf.set_text_color(50, 50, 50)

        # Tableau
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(230, 245, 255)
        pdf.cell(63, 8, "Entrees", border=1, fill=True, align='C')
        pdf.cell(63, 8, "Sorties", border=1, fill=True, align='C')
        pdf.cell(63, 8, "Presents", border=1, fill=True, align='C')
        pdf.ln()

        pdf.set_font('Helvetica', 'B', 14)
        entries = counter.get("total_entries", 0)
        exits = counter.get("total_exits", 0)
        present = counter.get("present", 0)

        pdf.set_text_color(0, 150, 0)
        pdf.cell(63, 12, str(entries), border=1, align='C')
        pdf.set_text_color(200, 0, 0)
        pdf.cell(63, 12, str(exits), border=1, align='C')
        pdf.set_text_color(0, 100, 200)
        pdf.cell(63, 12, str(present), border=1, align='C')
        pdf.ln(15)

        pdf.set_text_color(0, 0, 0)

    def _draw_persons_table(self, pdf, data: dict):
        """Tableau des personnes détectées."""
        persons = data.get("person_stats", {})
        if not persons:
            pdf.set_font('Helvetica', '', 11)
            pdf.cell(0, 10, "Aucune personne detectee.", ln=True)
            return

        # En-têtes
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(20, 25, 50)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(15, 8, "ID", border=1, fill=True, align='C')
        pdf.cell(35, 8, "Nom", border=1, fill=True, align='C')
        pdf.cell(30, 8, "Presence", border=1, fill=True, align='C')
        pdf.cell(35, 8, "Action", border=1, fill=True, align='C')
        pdf.cell(35, 8, "Action Top", border=1, fill=True, align='C')
        pdf.cell(40, 8, "Objets", border=1, fill=True, align='C')
        pdf.ln()

        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Helvetica', '', 9)

        for tid, ps in persons.items():
            name = ps.get("name", "INCONNU")
            presence = ps.get("presence_time", 0)
            mins = int(presence // 60)
            secs = int(presence % 60)
            pres_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            action = ps.get("current_action", "N/A")
            top_action = ps.get("top_action", "N/A")
            objects = ", ".join(ps.get("pose_objects", []))[:20]

            # Couleur alternée
            if int(str(tid)) % 2 == 0:
                pdf.set_fill_color(245, 245, 255)
            else:
                pdf.set_fill_color(255, 255, 255)

            pdf.cell(15, 7, str(tid), border=1, fill=True, align='C')
            pdf.cell(35, 7, name[:15], border=1, fill=True)
            pdf.cell(30, 7, pres_str, border=1, fill=True, align='C')
            pdf.cell(35, 7, action[:15], border=1, fill=True)
            pdf.cell(35, 7, str(top_action)[:15] if top_action else "N/A",
                     border=1, fill=True)
            pdf.cell(40, 7, objects if objects else "-", border=1, fill=True)
            pdf.ln()

    def _draw_alerts_section(self, pdf, data: dict):
        """Section des alertes."""
        alerts = data.get("alerts", [])
        total = data.get("total_alerts", 0)

        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 8, f"Total alertes: {total}", ln=True)
        pdf.ln(3)

        if not alerts:
            pdf.set_font('Helvetica', 'I', 10)
            pdf.cell(0, 8, "Aucune alerte pendant cette session.", ln=True)
            return

        # En-têtes tableau
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(150, 0, 0)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(40, 8, "Heure", border=1, fill=True, align='C')
        pdf.cell(45, 8, "Type", border=1, fill=True, align='C')
        pdf.cell(35, 8, "Personne", border=1, fill=True, align='C')
        pdf.cell(30, 8, "Confiance", border=1, fill=True, align='C')
        pdf.cell(40, 8, "Clip", border=1, fill=True, align='C')
        pdf.ln()

        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Helvetica', '', 9)

        for alert in alerts[-30:]:
            ts = alert.get("timestamp", "")
            atype = alert.get("type", "")
            name = alert.get("name", "INCONNU")
            conf = alert.get("confidence", 0)
            clip = alert.get("clip", "-")

            pdf.cell(40, 7, str(ts)[:19], border=1)
            pdf.cell(45, 7, atype[:20], border=1)
            pdf.cell(35, 7, name[:15], border=1)
            pdf.cell(30, 7, f"{conf:.0%}" if isinstance(conf, float) else str(conf),
                     border=1, align='C')
            pdf.cell(40, 7, str(clip)[:20] if clip else "-", border=1)
            pdf.ln()

    def _generate_charts(self, data: dict) -> Optional[str]:
        """Génère les graphiques avec matplotlib."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle('CCTV AI DEEP SECU — Analyse', fontsize=14, fontweight='bold')

        # Graphique 1 : Répartition des actions
        persons = data.get("person_stats", {})
        action_totals = {}
        for ps in persons.values():
            for action, dur in ps.get("action_durations", {}).items():
                if action != "N/A" and dur > 0.5:
                    action_totals[action] = action_totals.get(action, 0) + dur

        if action_totals:
            actions = list(action_totals.keys())
            durations = list(action_totals.values())
            colors = ['#00d4ff', '#00ff88', '#ffd700', '#ff4757',
                      '#b388ff', '#ff8c42', '#4ecdc4', '#95e1d3']
            ax1 = axes[0]
            ax1.barh(actions, durations, color=colors[:len(actions)])
            ax1.set_xlabel('Duree (secondes)')
            ax1.set_title('Repartition des Actions')
            ax1.invert_yaxis()
        else:
            axes[0].text(0.5, 0.5, 'Pas de donnees', ha='center', va='center')
            axes[0].set_title('Repartition des Actions')

        # Graphique 2 : Compteur
        counter = data.get("counter_stats", {})
        entries = counter.get("total_entries", 0)
        exits = counter.get("total_exits", 0)
        present = counter.get("present", 0)

        ax2 = axes[1]
        categories = ['Entrees', 'Sorties', 'Presents']
        values = [entries, exits, present]
        bar_colors = ['#00ff88', '#ff4757', '#00d4ff']
        bars = ax2.bar(categories, values, color=bar_colors, width=0.5)
        ax2.set_title('Compteur Entrees / Sorties')
        ax2.set_ylabel('Nombre')

        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
                     str(val), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # Sauvegarder en image temporaire
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return tmp.name

    def get_reports_list(self) -> list:
        """Retourne la liste des rapports générés."""
        reports = []
        for f in sorted(self.output_dir.glob("*.pdf"),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            reports.append({
                "filename": f.name,
                "path": str(f),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "date": datetime.fromtimestamp(f.stat().st_mtime).strftime(
                    '%d/%m/%Y %H:%M:%S'),
            })
        return reports[:20]
