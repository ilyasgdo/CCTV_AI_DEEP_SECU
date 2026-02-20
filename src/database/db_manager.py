"""
Module de gestion de la base de données SQLite.
Gère les enregistrements de présence et les alertes.
"""
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import DB_PATH, PERSON_LOST_TIMEOUT


class DatabaseManager:
    """
    Gestionnaire de base de données pour le système CCTV.
    Thread-safe grâce à check_same_thread=False.
    """

    def __init__(self, db_path: str = None):
        """
        Initialise la connexion et crée les tables.

        Args:
            db_path: Chemin vers le fichier SQLite (ou ':memory:' pour les tests)
        """
        if db_path is None:
            db_path = str(DB_PATH)

        # Créer le dossier parent si nécessaire (sauf pour :memory:)
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Accès par nom de colonne
        self.cursor = self.conn.cursor()

        self._create_tables()

        # Cache des personnes actuellement présentes
        self._active_tracks: Dict[int, int] = {}  # {track_id: record_id}
        self._last_seen: Dict[int, float] = {}     # {track_id: timestamp}

        print(f"[DB] Base de données initialisée : {db_path}")

    def _create_tables(self):
        """Crée les tables si elles n'existent pas."""
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS presence_records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id    INTEGER NOT NULL,
                name        TEXT DEFAULT 'INCONNU',
                entry_time  TIMESTAMP NOT NULL,
                exit_time   TIMESTAMP,
                duration_s  REAL,
                status      TEXT DEFAULT 'PRESENT',
                alert_flag  TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_track_id ON presence_records(track_id);
            CREATE INDEX IF NOT EXISTS idx_status ON presence_records(status);
            CREATE INDEX IF NOT EXISTS idx_entry_time ON presence_records(entry_time);

            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id    INTEGER NOT NULL,
                name        TEXT DEFAULT 'INCONNU',
                alert_type  TEXT NOT NULL,
                confidence  REAL,
                timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                frame_num   INTEGER,
                acknowledged INTEGER DEFAULT 0
            );
        """)
        self.conn.commit()

    # ==========================================
    # GESTION DES PRÉSENCES
    # ==========================================

    def register_entry(self, track_id: int, name: str = "INCONNU") -> int:
        """
        Enregistre l'entrée d'une nouvelle personne.

        Args:
            track_id: ID ByteTrack
            name: Nom identifié

        Returns:
            ID de l'enregistrement créé
        """
        now = datetime.now()
        self.cursor.execute("""
            INSERT INTO presence_records (track_id, name, entry_time, status)
            VALUES (?, ?, ?, 'PRESENT')
        """, (track_id, name, now))
        self.conn.commit()

        record_id = self.cursor.lastrowid
        self._active_tracks[track_id] = record_id
        self._last_seen[track_id] = time.time()

        print(f"  📥 Entrée : ID:{track_id} ({name}) à {now.strftime('%H:%M:%S')}")
        return record_id

    def register_exit(self, track_id: int):
        """
        Enregistre la sortie d'une personne (plus détectée depuis PERSON_LOST_TIMEOUT).
        """
        if track_id not in self._active_tracks:
            return

        record_id = self._active_tracks[track_id]
        now = datetime.now()

        # Calculer la durée
        self.cursor.execute(
            "SELECT entry_time FROM presence_records WHERE id = ?",
            (record_id,)
        )
        row = self.cursor.fetchone()
        if row:
            entry_time = datetime.fromisoformat(row["entry_time"])
            duration = (now - entry_time).total_seconds()

            self.cursor.execute("""
                UPDATE presence_records
                SET exit_time = ?, duration_s = ?, status = 'PARTI'
                WHERE id = ?
            """, (now, duration, record_id))
            self.conn.commit()

            print(f"  📤 Sortie : ID:{track_id} après {duration:.0f}s")

        # Nettoyer le cache
        del self._active_tracks[track_id]
        if track_id in self._last_seen:
            del self._last_seen[track_id]

    def update_presence(self, track_id: int, name: str = None):
        """
        Met à jour le timestamp de dernière vue pour un track_id.
        Appelé à chaque frame pour chaque personne détectée.
        """
        self._last_seen[track_id] = time.time()

        # Nouvelle personne ?
        if track_id not in self._active_tracks:
            self.register_entry(track_id, name or "INCONNU")
        elif name and name != "INCONNU":
            # Mettre à jour le nom si identifié
            record_id = self._active_tracks[track_id]
            self.cursor.execute(
                "UPDATE presence_records SET name = ? WHERE id = ? AND name = 'INCONNU'",
                (name, record_id)
            )
            self.conn.commit()

    def check_exits(self):
        """
        Vérifie si des personnes ont disparu depuis plus de PERSON_LOST_TIMEOUT.
        À appeler régulièrement (ex: toutes les 30 frames).
        """
        now = time.time()
        lost_ids = []

        for track_id, last_seen in self._last_seen.items():
            if (now - last_seen) > PERSON_LOST_TIMEOUT:
                lost_ids.append(track_id)

        for track_id in lost_ids:
            self.register_exit(track_id)

    def update_name(self, track_id: int, name: str):
        """Met à jour le nom d'une personne identifiée."""
        if track_id in self._active_tracks:
            record_id = self._active_tracks[track_id]
            self.cursor.execute(
                "UPDATE presence_records SET name = ? WHERE id = ?",
                (name, record_id)
            )
            self.conn.commit()

    # ==========================================
    # GESTION DES ALERTES
    # ==========================================

    def log_alert(self, track_id: int, alert_type: str,
                  confidence: float = 0.0, name: str = "INCONNU",
                  frame_num: int = 0) -> int:
        """
        Enregistre une alerte dans la base de données.

        Args:
            track_id: ID de la personne
            alert_type: Type d'alerte (CHUTE, MARAUDAGE, COUP...)
            confidence: Score de confiance
            name: Nom de la personne
            frame_num: Numéro de frame
        """
        self.cursor.execute("""
            INSERT INTO alerts (track_id, name, alert_type, confidence, frame_num)
            VALUES (?, ?, ?, ?, ?)
        """, (track_id, name, alert_type, confidence, frame_num))
        self.conn.commit()

        # Aussi marquer l'enregistrement de présence
        if track_id in self._active_tracks:
            record_id = self._active_tracks[track_id]
            self.cursor.execute(
                "UPDATE presence_records SET alert_flag = ? WHERE id = ?",
                (alert_type, record_id)
            )
            self.conn.commit()

        print(f"  🚨 ALERTE {alert_type} : ID:{track_id} ({name}) "
              f"[confiance: {confidence:.1%}]")
        return self.cursor.lastrowid

    # ==========================================
    # REQUÊTES DE CONSULTATION
    # ==========================================

    def get_currently_present(self) -> List[dict]:
        """Retourne la liste des personnes actuellement présentes."""
        self.cursor.execute("""
            SELECT track_id, name, entry_time
            FROM presence_records
            WHERE status = 'PRESENT'
            ORDER BY entry_time DESC
        """)
        return [dict(row) for row in self.cursor.fetchall()]

    def get_history(self, limit: int = 50) -> List[dict]:
        """Retourne l'historique des présences."""
        self.cursor.execute("""
            SELECT * FROM presence_records
            ORDER BY entry_time DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in self.cursor.fetchall()]

    def get_alerts(self, limit: int = 20,
                   unacknowledged_only: bool = False) -> List[dict]:
        """Retourne les alertes récentes."""
        query = "SELECT * FROM alerts"
        if unacknowledged_only:
            query += " WHERE acknowledged = 0"
        query += " ORDER BY timestamp DESC LIMIT ?"

        self.cursor.execute(query, (limit,))
        return [dict(row) for row in self.cursor.fetchall()]

    def acknowledge_alert(self, alert_id: int):
        """Acquitte une alerte."""
        self.cursor.execute(
            "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
            (alert_id,)
        )
        self.conn.commit()

    def get_time_spent(self, name: str) -> float:
        """Retourne le temps total passé par une personne (en secondes)."""
        self.cursor.execute("""
            SELECT SUM(duration_s) as total
            FROM presence_records
            WHERE name = ? AND duration_s IS NOT NULL
        """, (name,))
        row = self.cursor.fetchone()
        return row["total"] or 0.0

    def get_stats(self) -> dict:
        """Retourne les statistiques globales."""
        self.cursor.execute("SELECT COUNT(*) as total FROM presence_records")
        total = self.cursor.fetchone()["total"]

        self.cursor.execute(
            "SELECT COUNT(*) as present FROM presence_records WHERE status = 'PRESENT'"
        )
        present = self.cursor.fetchone()["present"]

        self.cursor.execute("SELECT COUNT(*) as alerts FROM alerts")
        alerts = self.cursor.fetchone()["alerts"]

        return {
            "total_records": total,
            "currently_present": present,
            "total_alerts": alerts,
            "active_tracks": len(self._active_tracks)
        }

    def close(self):
        """Ferme la connexion à la base de données."""
        # Marquer toutes les personnes encore présentes comme parties
        for track_id in list(self._active_tracks.keys()):
            self.register_exit(track_id)
        self.conn.close()
        print("[DB] Connexion fermée.")
