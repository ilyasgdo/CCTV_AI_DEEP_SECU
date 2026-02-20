# 🗄️ Étape 3 — L'Historique et les Temps (SQLite)

## 📋 Summary (À lire AVANT de commencer)

**Objectif** : Créer le système de **base de données SQLite** qui archive automatiquement les entrées/sorties de chaque personne détectée, avec calcul du temps passé. Cette brique est essentielle pour le maraudage (Étape 4) et l'historique consultable.

**Durée estimée** : 1-2 heures

**Prérequis** :
- ✅ Étape 0 et Étape 1 validées
- ✅ Le détecteur retourne des `track_id` stables
- ✅ (Optionnel) Étape 2 validée pour avoir les noms

**Ce que vous aurez à la fin** :
- ✅ Base de données SQLite avec schéma propre
- ✅ Module CRUD pour enregistrer entrées/sorties
- ✅ Logique temporelle : détection d'arrivée et de départ
- ✅ Requêtes de consultation (qui est là, historique, temps passé)
- ✅ Test complet validé

---

## 📝 Étapes Détaillées

### 3.1 — Concevoir le Schéma de la Base de Données

**Schéma SQL :**

```sql
-- Table principale : enregistrements de présence
CREATE TABLE IF NOT EXISTS presence_records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id    INTEGER NOT NULL,          -- ID ByteTrack
    name        TEXT DEFAULT 'INCONNU',     -- Nom identifié (InsightFace)
    entry_time  TIMESTAMP NOT NULL,         -- Heure d'apparition
    exit_time   TIMESTAMP,                  -- Heure de disparition (NULL si encore là)
    duration_s  REAL,                       -- Durée en secondes (calculée à la sortie)
    status      TEXT DEFAULT 'PRESENT',     -- PRESENT / PARTI
    alert_flag  TEXT,                       -- Type d'alerte si applicable
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour les requêtes fréquentes
CREATE INDEX IF NOT EXISTS idx_track_id ON presence_records(track_id);
CREATE INDEX IF NOT EXISTS idx_status ON presence_records(status);
CREATE INDEX IF NOT EXISTS idx_entry_time ON presence_records(entry_time);

-- Table des alertes
CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id    INTEGER NOT NULL,
    name        TEXT DEFAULT 'INCONNU',
    alert_type  TEXT NOT NULL,              -- CHUTE, MARAUDAGE, COUP, etc.
    confidence  REAL,                       -- Score de confiance ST-GCN
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    frame_num   INTEGER,                    -- Numéro de frame
    acknowledged INTEGER DEFAULT 0          -- 0 = non lu, 1 = acquitté
);
```

**Diagramme relationnel :**
```
┌─────────────────────────┐       ┌──────────────────────────┐
│   presence_records      │       │       alerts             │
├─────────────────────────┤       ├──────────────────────────┤
│ id (PK)                 │       │ id (PK)                  │
│ track_id                │──────▸│ track_id                 │
│ name                    │       │ name                     │
│ entry_time              │       │ alert_type               │
│ exit_time               │       │ confidence               │
│ duration_s              │       │ timestamp                │
│ status                  │       │ frame_num                │
│ alert_flag              │       │ acknowledged             │
│ created_at              │       └──────────────────────────┘
└─────────────────────────┘
```

---

### 3.2 — Créer le Module de Gestion de Base de Données (`src/database/db_manager.py`)

**Actions :**

Créer `src/database/db_manager.py` :

```python
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
            db_path: Chemin vers le fichier SQLite
        """
        if db_path is None:
            db_path = str(DB_PATH)
        
        # Créer le dossier parent si nécessaire
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
        self.conn.close()
        print("[DB] Connexion fermée.")
```

**✅ Critère de validation 3.2** :
```python
python -c "
from src.database.db_manager import DatabaseManager
import time

db = DatabaseManager(':memory:')  # Base en mémoire pour le test

# Test entrée
db.register_entry(1, 'Thomas')
db.register_entry(2, 'INCONNU')

# Test présence
present = db.get_currently_present()
assert len(present) == 2, f'Attendu 2, trouvé {len(present)}'
print(f'✅ Personnes présentes : {len(present)}')

# Test mise à jour nom
db.update_name(2, 'Sarah')

# Test sortie
db.register_exit(1)

# Test historique
history = db.get_history()
assert len(history) == 2
thomas = [r for r in history if r['name'] == 'Thomas'][0]
assert thomas['status'] == 'PARTI'
assert thomas['duration_s'] is not None
print(f'✅ Historique OK ({len(history)} enregistrements)')

# Test alerte
db.log_alert(2, 'CHUTE', 0.93, 'Sarah', frame_num=150)
alerts = db.get_alerts()
assert len(alerts) == 1
print(f'✅ Alertes OK ({len(alerts)} alerte)')

# Stats
stats = db.get_stats()
print(f'✅ Stats : {stats}')

db.close()
print('\\n✅ TOUS LES TESTS DB PASSENT')
"
```

---

### 3.3 — Intégrer la Base de Données dans le Pipeline

**Actions :**

Créer le test d'intégration `tests/test_database.py` :

```python
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
        print(f"  ID:{record['track_id']} | {record['name']} | "
              f"Entrée: {record['entry_time']} | "
              f"Status: {record['status']} | "
              f"Durée: {record.get('duration_s', 'N/A')}s")
    
    db.close()
    return True


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    success = test_with_database(source)
    print(f"\n{'✅ TEST RÉUSSI' if success else '❌ TEST ÉCHOUÉ'}")
```

**✅ Critère de validation 3.3** :
```powershell
python tests/test_database.py

# DOIT :
# - Créer le fichier src/database/cctv_records.db
# - Afficher "DB: X present(s)" sur la vidéo
# - Enregistrer les entrées quand de nouvelles personnes apparaissent
# - Appuyer sur 's' affiche les stats et l'historique dans la console
```

---

## ✅ Checklist de Validation Finale — Étape 3

| # | Critère | Commande/Action | Status |
|---|---------|-----------------|--------|
| 3.1 | Schéma SQL défini | `presence_records` + `alerts` + 3 index | ✅ |
| 3.2 | Module `db_manager.py` testé | Tests in-memory complets (CRUD+alertes) | ✅ |
| 3.3 | Intégration pipeline + BDD | Script `test_database.py` prêt | ✅ |

**Vérifications fonctionnelles obligatoires :**
- [ ] Nouvelle personne détectée → ligne créée avec `PRESENT`
- [ ] Personne disparue > 5 min → ligne mise à jour avec `PARTI` + durée
- [ ] Le nom est mis à jour quand InsightFace identifie la personne
- [ ] Les alertes sont correctement enregistrées
- [ ] Les requêtes de consultation retournent des données cohérentes
- [ ] Le fichier `cctv_records.db` est créé et consultable

> [!TIP]
> Pour inspecter la base de données manuellement :
> ```powershell
> # Installer sqlite3 ou utiliser DB Browser for SQLite
> sqlite3 src/database/cctv_records.db "SELECT * FROM presence_records;"
> ```

---

**⬅️ Étape précédente : [etape_2.md](etape_2.md)**
**➡️ Étape suivante : [etape_4.md](etape_4.md) — Analyse Comportementale (ST-GCN)**
